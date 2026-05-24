"""Background reader/writer threads to overlap CPU and GPU work.

The single-threaded loop in cli.py was dominated by two serial CPU costs
(4K H.264 decode at ~74 ms/frame and cv2.rotate at ~108 ms/frame) running
back-to-back with the GPU detection (~100 ms/frame combined for the two
TensorRT engines). Because Python ran everything sequentially, the GPU
was idle while the CPU decoded, the CPU was idle while the GPU detected,
and the writer was idle until both were done.

These two classes let those stages run in parallel:

* `ThreadedReader` wraps a frame source (a `cv2.VideoCapture` or our
  `FFmpegReader`) plus the optional `--sensor` rotation. A background
  thread pulls frames, rotates them, and pushes them onto a bounded
  queue. The main thread just calls `.read()`.

* `ThreadedWriter` wraps any writer with the `write/close` interface.
  Frames pushed via `.write()` are queued, and a background thread drains
  the queue into the underlying writer (which itself often pipes to an
  external ffmpeg encoder).

Bounded queues keep memory in check: a few 4K BGR frames at ~35 MiB each
is several hundred MiB if we let the queue grow unbounded.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import cv2
import numpy as np


_SENTINEL = object()


class ThreadedReader:
    """Background-decode + (optionally) rotate frames into a bounded queue.

    `read_fn` must return either a BGR numpy array (`np.ndarray`) for the next
    frame or None when the source is exhausted. We treat None as end-of-stream.

    `rotate` ∈ {None, "ccw", "cw"} applies a `cv2.rotate` in the reader thread
    so the main thread receives frames already in the correct orientation for
    detection. Putting the rotation here is what makes threading pay off.
    """

    def __init__(self, read_fn: Callable[[], Optional[np.ndarray]],
                 rotate: Optional[str] = None, maxsize: int = 4):
        if rotate not in (None, "ccw", "cw"):
            raise ValueError(f"rotate must be None|'ccw'|'cw', got {rotate!r}")
        self._read_fn = read_fn
        self._rotate = rotate
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._exc: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._loop, name="ThreadedReader",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._read_fn()
                if frame is None:
                    break
                if self._rotate == "ccw":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._rotate == "cw":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # `put` blocks when the consumer is slow, which back-pressures
                # the decoder and caps memory use.
                self._queue.put(frame)
        except BaseException as e:
            self._exc = e
        finally:
            self._queue.put(_SENTINEL)

    def read(self) -> Optional[np.ndarray]:
        item = self._queue.get()
        if item is _SENTINEL:
            if self._exc is not None:
                raise self._exc
            return None
        return item  # type: ignore[return-value]

    def close(self) -> None:
        self._stop.set()
        # Drain the queue so the producer can finish if it's blocked on put().
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)


class ThreadedWriter:
    """Background-write frames so the main loop doesn't block on the encoder.

    Wraps any object with `write(frame)` and `close()` (our `FFmpegWriter` or
    `OpenCVWriter`). The main thread calls `.write(frame)` which enqueues; a
    background thread drains the queue into the underlying writer.

    `rotate` ∈ {None, "ccw", "cw"} applies a `cv2.rotate` inside the writer
    thread before handing the frame to the underlying writer. This is useful
    when the main loop processes a rotated frame but the output needs to be in
    the original orientation, and the writer itself can't apply rotation (e.g.
    `OpenCVWriter`, or anyone running without an ffmpeg-side `transpose=`).
    """

    def __init__(self, writer, rotate: Optional[str] = None, maxsize: int = 4):
        if rotate not in (None, "ccw", "cw"):
            raise ValueError(f"rotate must be None|'ccw'|'cw', got {rotate!r}")
        self._writer = writer
        self._rotate = rotate
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=maxsize)
        self._exc: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._loop, name="ThreadedWriter",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                frame = item  # type: ignore[assignment]
                if self._rotate == "ccw":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._rotate == "cw":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                self._writer.write(frame)
        except BaseException as e:
            self._exc = e

    def write(self, frame: np.ndarray) -> None:
        if self._exc is not None:
            raise self._exc
        self._queue.put(frame)

    def close(self) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._exc is not None:
            raise self._exc
        self._writer.close()
