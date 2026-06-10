import argparse, cv2, time
from .detectors import PlateDetectorYOLO, FaceDetectorHaar, FaceDetectorYOLO
from .blur_ops import gaussian_inplace, pixelate_inplace
from .utils import choose_device, expand_box, nms_merge
from .writers import make_writer, FFmpegReader
from .pipeline import ThreadedReader, ThreadedWriter


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Register all flags shared by the single-file CLI and the batch CLI.

    The single-file CLI adds `--input`/`--output` on top of these; the batch
    CLI adds `--work-dir` plus batch-specific flags. Keeping these in one
    function ensures behavior stays identical between the two entry points.
    """
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="inference device")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--scale", type=float, default=1.35, help="bbox expansion factor")
    ap.add_argument("--show", action="store_true", help="preview window")
    ap.add_argument("--blur-plates", action="store_true", default=True, help="blur license plates")
    ap.add_argument("--no-blur-plates", dest="blur_plates", action="store_false")
    ap.add_argument("--blur-faces", action="store_true", default=True, help="blur faces")
    ap.add_argument("--no-blur-faces", dest="blur_faces", action="store_false")
    ap.add_argument("--face-detector", choices=["haar","yolo"], default="haar", help="face backend")
    ap.add_argument("--face-yolo-weights", default="yolov8n-face.pt", help="YOLO face weights (file or hub id)")
    ap.add_argument("--face-task", choices=["pose", "detect"], default="pose",
                    help="Ultralytics task for the face model. 'pose' for yolov8-face (default), 'detect' for plain face detectors. Required to be set correctly when loading .engine files.")
    ap.add_argument("--plate-task", choices=["detect", "pose"], default="detect",
                    help="Ultralytics task for the plate model. 'detect' for standard plate detectors (default).")
    ap.add_argument("--method", choices=["gaussian","pixelate"], default="gaussian", help="blur method")
    ap.add_argument("--plate-weights", default="models/license_plate_detector.pt", help="Path to YOLO license plate model (.pt)")
    ap.add_argument("--blur-strength", type=float, default=3.0, help="Lower = stronger blur, higher = weaker blur (divisor for bbox size)")
    ap.add_argument("--sensor", action="store_true", help="Rotate frames 90 degrees CCW for processing, then back before saving")
    ap.add_argument("--encoder", choices=["nvenc", "hevc_nvenc", "x264", "mp4v"], default="nvenc",
                    help="Video encoder. 'nvenc' = H.264 on NVIDIA GPU (fast). 'hevc_nvenc' = HEVC on GPU (smaller files). 'x264' = CPU H.264. 'mp4v' = legacy OpenCV fallback.")
    ap.add_argument("--cq", type=int, default=23, help="NVENC constant quality (lower = better quality, larger file). 18-28 is typical.")
    ap.add_argument("--preset", default="p4", help="NVENC preset p1 (fastest) to p7 (best quality). p4 = balanced.")
    ap.add_argument("--reader", choices=["opencv", "ffmpeg"], default="opencv",
                    help="Input decoder. 'opencv' uses cv2.VideoCapture (default, fastest on this hardware/codec mix). 'ffmpeg' pipes from an ffmpeg subprocess with NVDEC + transpose filter; useful for codecs OpenCV decodes poorly but typically slower because of the CPU-side NV12->BGR24 conversion.")
    ap.add_argument("--no-hwaccel", action="store_true",
                    help="Disable -hwaccel cuda in the ffmpeg reader (use only if NVDEC misbehaves on a specific codec).")
    ap.add_argument("--threads", action="store_true", default=True,
                    help="Run decode/rotate in a background reader thread and encoding in a background writer thread so they overlap with GPU detection. On by default.")
    ap.add_argument("--no-threads", dest="threads", action="store_false",
                    help="Disable the threaded pipeline (single-threaded loop). Useful for debugging or to compare against the baseline.")
    ap.add_argument("--queue-size", type=int, default=4,
                    help="Max frames buffered between threads. 2-8 is sensible; higher uses more RAM.")


def build_detectors(args, device: str):
    """Construct plate + face detectors from `args`. Returns (plate, face).

    Either may be None when the corresponding `--no-blur-*` flag is set. The
    batch runner calls this once and passes the result into every
    `process_video()` invocation so TRT engines aren't reloaded per file.
    """
    plate_detector = PlateDetectorYOLO(args.plate_weights, device, args.conf, args.imgsz,
                                        task=args.plate_task) \
                 if args.blur_plates else None

    if args.blur_faces:
        face_detector = FaceDetectorHaar() if args.face_detector=="haar" else FaceDetectorYOLO(
            args.face_yolo_weights, device, args.conf, args.imgsz, task=args.face_task
        )
    else:
        face_detector = None
    return plate_detector, face_detector


def process_video(args, plate_detector, face_detector) -> dict:
    """Run the blur pipeline on `args.input` -> `args.output`.

    Constructs the reader/writer/threads from `args`, but takes pre-built
    detectors so callers (the batch runner) can reuse them across files. On
    success the function returns a stats dict with `frames`, `elapsed_s`,
    `fps`, `n_boxes_total`, and a `per_frame_ms` breakdown. Errors raise
    instead of `sys.exit()` so the batch runner can record them in
    `results.json` and keep going.
    """
    use_ffmpeg_reader = args.reader == "ffmpeg" and not str(args.input).isdigit()
    ffmpeg_reader = None  # FFmpegReader instance when use_ffmpeg_reader else None
    cap = None            # cv2.VideoCapture instance when not use_ffmpeg_reader else None

    # `processed_orientation` describes the orientation Python sees during the
    # detect/blur loop. When --sensor is on we want detection to see the
    # rotated (portrait) orientation. We achieve that either by asking the
    # ffmpeg reader to apply transpose=2, or by rotating in the (threaded) reader.
    if use_ffmpeg_reader:
        transpose_in  = "ccw" if args.sensor else None
        transpose_out = "cw"  if args.sensor else None
        ffmpeg_reader = FFmpegReader(args.input, transpose_in=transpose_in,
                                     hwaccel=not args.no_hwaccel)
        W, H, fps = ffmpeg_reader.W_out, ffmpeg_reader.H_out, ffmpeg_reader.fps
        print(f"[INFO] Reader: ffmpeg {'(NVDEC' if not args.no_hwaccel else '(software'}"
              f"{', transpose_in=' + transpose_in if transpose_in else ''})"
              f"  in={ffmpeg_reader.W_in}x{ffmpeg_reader.H_in}  out={W}x{H}@{fps:.2f}")
        reader_rotate, writer_rotate = None, None
        writer_inner = make_writer(args, W, H, fps, transpose_out=transpose_out)
    else:
        cap = cv2.VideoCapture(int(args.input) if str(args.input).isdigit() else args.input)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input: {args.input}")
        W_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if args.sensor:
            reader_rotate, writer_rotate = "ccw", "cw"
            W, H = H_raw, W_raw  # post-rotation dimensions
        else:
            reader_rotate, writer_rotate = None, None
            W, H = W_raw, H_raw
        print(f"[INFO] Reader: opencv (cv2.VideoCapture)  raw={W_raw}x{H_raw}  "
              f"processed={W}x{H}@{fps:.2f}")
        writer_inner = make_writer(args, W_raw, H_raw, fps)

    def _cv2_read():
        ok, frame = cap.read()
        return frame if ok else None

    raw_read = ffmpeg_reader.read if use_ffmpeg_reader else _cv2_read

    if args.threads:
        reader = ThreadedReader(raw_read, rotate=reader_rotate, maxsize=args.queue_size)
        writer = ThreadedWriter(writer_inner, rotate=writer_rotate, maxsize=args.queue_size)
        print(f"[INFO] Pipeline: threaded reader + threaded writer (queue size {args.queue_size})")
    else:
        class _DirectReader:
            def __init__(self, read_fn, rotate):
                self._fn = read_fn; self._rot = rotate
            def read(self):
                f = self._fn()
                if f is None: return None
                if self._rot == "ccw": return cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if self._rot == "cw":  return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
                return f
            def close(self): pass

        class _DirectWriter:
            def __init__(self, w, rotate): self._w = w; self._rot = rotate
            def write(self, f):
                if self._rot == "ccw": f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._rot == "cw": f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
                self._w.write(f)
            def close(self): self._w.close()

        reader = _DirectReader(raw_read, reader_rotate)
        writer = _DirectWriter(writer_inner, writer_rotate)
        print("[INFO] Pipeline: single-threaded loop")

    start_t = time.perf_counter()
    frames = 0
    n_boxes_total = 0
    t_read = t_plate = t_face = t_nms = t_blur = t_write = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            frame = reader.read()
            t_read += time.perf_counter() - t0
            if frame is None: break
            frames += 1

            boxes = []
            if plate_detector:
                t0 = time.perf_counter()
                boxes += plate_detector(frame)
                t_plate += time.perf_counter() - t0
            if face_detector:
                t0 = time.perf_counter()
                boxes += face_detector(frame)
                t_face += time.perf_counter() - t0

            t0 = time.perf_counter()
            boxes = nms_merge(boxes, iou_thresh=0.5)
            t_nms += time.perf_counter() - t0
            n_boxes_total += len(boxes)

            h_proc, w_proc = frame.shape[:2]

            t0 = time.perf_counter()
            for (x1,y1,x2,y2) in boxes:
                x1,y1,x2,y2 = expand_box(x1,y1,x2,y2, args.scale, w_proc, h_proc)
                if args.method == "gaussian":
                    gaussian_inplace(frame, x1,y1,x2,y2, args.blur_strength)
                else:
                    pixelate_inplace(frame, x1,y1,x2,y2)
            t_blur += time.perf_counter() - t0

            t0 = time.perf_counter()
            writer.write(frame)
            t_write += time.perf_counter() - t0
            if args.show:
                cv2.imshow("privacy-blur", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        # Always tear IO down even on error, so files aren't left half-written
        # and ffmpeg subprocesses don't linger.
        reader.close()
        writer.close()
        if cap is not None:
            cap.release()
        if ffmpeg_reader is not None:
            ffmpeg_reader.close()
        if args.show: cv2.destroyAllWindows()

    elapsed_s = max(1e-9, time.perf_counter() - start_t)
    fps_out = frames / elapsed_s
    n = max(1, frames)
    per_frame_ms = {
        "read":  t_read  / n * 1000,
        "plate": t_plate / n * 1000,
        "face":  t_face  / n * 1000,
        "nms":   t_nms   / n * 1000,
        "blur":  t_blur  / n * 1000,
        "write": t_write / n * 1000,
    }
    print(f"[STATS] Frames: {frames} | Time: {elapsed_s:.2f}s | Speed: {fps_out:.2f} fps")
    print(
        f"[PROFILE] per-frame ms  "
        f"read={per_frame_ms['read']:6.1f}  "
        f"plate={per_frame_ms['plate']:6.1f}  "
        f"face={per_frame_ms['face']:6.1f}  "
        f"nms={per_frame_ms['nms']:6.1f}  "
        f"blur={per_frame_ms['blur']:6.1f}  "
        f"write={per_frame_ms['write']:6.1f}  "
        f"boxes/frame={n_boxes_total/n:.2f}"
    )
    print(f"[OK] Saved: {args.output}")

    return {
        "frames": frames,
        "elapsed_s": elapsed_s,
        "fps": fps_out,
        "n_boxes_total": n_boxes_total,
        "per_frame_ms": per_frame_ms,
    }


def main():
    ap = argparse.ArgumentParser("Blur license plates and/or faces in video")
    ap.add_argument("--input", required=True, help="video path or camera index, e.g. 0")
    ap.add_argument("--output", default="out_blurred.mp4", help="output video path")
    add_common_args(ap)
    args = ap.parse_args()

    device = choose_device(args.device)
    print(f"[INFO] Using device: {device}")

    plate_detector, face_detector = build_detectors(args, device)
    process_video(args, plate_detector, face_detector)
