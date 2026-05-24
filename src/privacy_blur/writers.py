"""Video writer backends.

Two implementations behind a common `write(frame) / close()` interface:

* `OpenCVWriter`   - legacy `cv2.VideoWriter` with the `mp4v` software codec.
                     Kept as a no-dependency fallback for machines without FFmpeg
                     or without an NVIDIA GPU.

* `FFmpegWriter`   - pipes raw BGR frames into a subprocess `ffmpeg` and lets it
                     encode with hardware NVENC (`h264_nvenc` / `hevc_nvenc`) or
                     CPU `libx264`. This is the fast path on machines with an
                     NVIDIA GPU and an FFmpeg build that includes NVENC.

`make_writer(args, W, H, fps)` is the factory used by `cli.py`. It also
auto-falls-back to mp4v with a clear warning if FFmpeg or NVENC is missing,
so the CLI never crashes silently mid-video because of a missing dependency.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Optional

import cv2
import numpy as np


class _BaseWriter:
    def write(self, frame: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class OpenCVWriter(_BaseWriter):
    """Software MPEG-4 writer using OpenCV. Slow on 4K, but zero external deps."""

    def __init__(self, path: str, w: int, h: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open: {path}")
        self._expected_shape = (h, w, 3)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class FFmpegWriter(_BaseWriter):
    """Pipe raw BGR frames to a subprocess `ffmpeg` for hardware encoding.

    Designed for `h264_nvenc` / `hevc_nvenc` on NVIDIA GPUs, but also works with
    `libx264` as a CPU fallback. We deliberately use a one-way stdin pipe and
    let ffmpeg's stderr print to the console so encoding errors are visible.

    `transpose_out` (None | "cw" | "ccw") inserts an output-side rotation filter
    so we can give ffmpeg a rotated frame (e.g. portrait, as produced by an
    FFmpegReader with --sensor) and have it encoded in the original landscape
    orientation. ffmpeg's `transpose` filter is much faster than `cv2.rotate`.
    """

    def __init__(
        self,
        path: str,
        w: int,
        h: int,
        fps: float,
        codec: str = "h264_nvenc",
        preset: str = "p4",
        cq: int = 23,
        transpose_out: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
    ):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH. Install it or use --encoder mp4v.")

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        is_nvenc = codec.endswith("_nvenc")
        rc_args = ["-rc", "vbr", "-cq", str(cq), "-b:v", "0"] if is_nvenc else ["-crf", str(cq)]
        preset_args = ["-preset", preset] if is_nvenc else ["-preset", "veryfast"]

        vf_args: list[str] = []
        if transpose_out == "cw":
            vf_args = ["-vf", "transpose=1"]
        elif transpose_out == "ccw":
            vf_args = ["-vf", "transpose=2"]
        elif transpose_out is not None:
            raise ValueError(f"transpose_out must be None|'cw'|'ccw', got {transpose_out!r}")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{fps:.6f}",
            "-i", "-",
            "-an",
            *vf_args,
            "-c:v", codec,
            *preset_args,
            *rc_args,
            "-pix_fmt", "yuv420p",
            *(extra_args or []),
            path,
        ]

        self._cmd = cmd
        self._expected_shape = (h, w, 3)
        self._path = path
        self._codec = codec

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        time.sleep(0.15)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"ffmpeg failed to start (exit code {self._proc.poll()}). "
                f"Command was:\n  {' '.join(cmd)}"
            )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self._expected_shape:
            raise RuntimeError(
                f"Frame shape {frame.shape} does not match writer shape {self._expected_shape}. "
                "If you use --sensor, make sure the frame is rotated back before writing."
            )
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            raise RuntimeError(
                f"ffmpeg pipe closed unexpectedly while writing to {self._path}. "
                f"Encoder '{self._codec}' likely failed; rerun with `--encoder mp4v` to confirm."
            ) from e

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        ret = self._proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with non-zero status {ret} writing {self._path}")


def _ffmpeg_has_encoder(name: str) -> bool:
    """Return True if the installed ffmpeg lists `name` in its encoders."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return False
    return name in out.stdout


_ENCODER_CODEC = {
    "nvenc": "h264_nvenc",
    "hevc_nvenc": "hevc_nvenc",
    "x264": "libx264",
    "mp4v": None,
}


def make_writer(args, w: int, h: int, fps: float,
                transpose_out: Optional[str] = None) -> _BaseWriter:
    """Build the right writer for `args.encoder`, with graceful fallback.

    Falls back to OpenCV's `mp4v` writer (with a clear stderr warning) if the
    user picked an FFmpeg encoder but `ffmpeg` or the specific encoder is
    missing at runtime. This means an outdated PATH never crashes a long job.

    `transpose_out` is forwarded to FFmpegWriter for the rotated-input case
    (e.g. when FFmpegReader has already rotated frames CCW and we need ffmpeg
    to rotate them back CW before encoding).
    """
    requested = getattr(args, "encoder", "nvenc")
    cq = getattr(args, "cq", 23)
    preset = getattr(args, "preset", "p4")

    if requested == "mp4v":
        if transpose_out is not None:
            print(
                "[WARN] --encoder mp4v cannot apply an output-side rotation. "
                "The OpenCV writer will store frames in their current orientation. "
                "Use --encoder nvenc (or any ffmpeg encoder) if you need ffmpeg-side rotation.",
                file=sys.stderr,
            )
        print("[INFO] Writer: OpenCV mp4v (software, slow)")
        return OpenCVWriter(args.output, w, h, fps)

    codec = _ENCODER_CODEC.get(requested)
    if codec is None:
        raise ValueError(f"Unknown encoder: {requested}")

    if not _ffmpeg_has_encoder(codec):
        print(
            f"[WARN] ffmpeg does not expose encoder '{codec}'. "
            "Falling back to OpenCV mp4v. Install a full ffmpeg build with NVENC "
            "(e.g. `winget install Gyan.FFmpeg`) to get the fast path.",
            file=sys.stderr,
        )
        return OpenCVWriter(args.output, w, h, fps)

    rot_str = f", transpose_out={transpose_out}" if transpose_out else ""
    print(f"[INFO] Writer: ffmpeg {codec} (preset={preset}, cq={cq}{rot_str})")
    return FFmpegWriter(args.output, w, h, fps,
                        codec=codec, preset=preset, cq=cq,
                        transpose_out=transpose_out)


class FFmpegReader:
    """Decode + optionally rotate the input video with ffmpeg, pipe raw BGR to Python.

    Moves the two biggest CPU costs in the previous pipeline (4K H.264 decode
    and a 4K `cv2.rotate`) into ffmpeg, where decoding can use NVDEC and the
    transpose filter is a highly optimised C path. Python receives ready-to-use
    BGR frames in the post-rotation shape.

    `transpose_in` (None | "cw" | "ccw") applies the rotation inside ffmpeg.
    Use "ccw" to mirror `--sensor` (which was `cv2.ROTATE_90_COUNTERCLOCKWISE`).

    `hwaccel` toggles NVDEC. Defaults to True; falls back to software decode
    silently if the input codec isn't NVDEC-capable.
    """

    def __init__(self, path: str, transpose_in: Optional[str] = None, hwaccel: bool = True):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH. Use --reader opencv instead.")

        # Probe input dimensions via OpenCV (cheap, doesn't decode the stream).
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input: {path}")
        self.W_in  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H_in  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps   = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()

        if transpose_in in ("cw", "ccw"):
            self.W_out, self.H_out = self.H_in, self.W_in
        elif transpose_in is None:
            self.W_out, self.H_out = self.W_in, self.H_in
        else:
            raise ValueError(f"transpose_in must be None|'cw'|'ccw', got {transpose_in!r}")

        vf_args: list[str] = []
        if transpose_in == "ccw":
            vf_args = ["-vf", "transpose=2"]
        elif transpose_in == "cw":
            vf_args = ["-vf", "transpose=1"]

        hw_args = ["-hwaccel", "cuda"] if hwaccel else []

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            *hw_args,
            "-i", path,
            *vf_args,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-",
        ]
        self._cmd = cmd
        self._shape = (self.H_out, self.W_out, 3)
        self._bytes_per_frame = self.H_out * self.W_out * 3
        self._path = path

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            bufsize=self._bytes_per_frame * 2,
        )

        time.sleep(0.15)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"ffmpeg reader failed to start (exit code {self._proc.poll()}). "
                f"Command was:\n  {' '.join(cmd)}"
            )

    def read(self) -> Optional[np.ndarray]:
        raw = self._proc.stdout.read(self._bytes_per_frame)
        if not raw or len(raw) < self._bytes_per_frame:
            return None
        # np.frombuffer over a bytes object is read-only. The downstream
        # blur ops mutate the frame in place, so we copy once into a writable
        # numpy array. The memcpy is ~5-10 ms for a 4K frame, still tiny
        # compared to the 108 ms rotation it replaces.
        return np.frombuffer(raw, dtype=np.uint8).reshape(self._shape).copy()

    def close(self) -> None:
        if self._proc.stdout and not self._proc.stdout.closed:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
