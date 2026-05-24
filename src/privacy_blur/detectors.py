from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


def _warmup_and_log(model: "YOLO", name: str, device: str, imgsz: int, half: bool) -> None:
    """Run one silent dummy inference to warm up the model kernels.

    We deliberately use `verbose=False` because Ultralytics' verbose printer
    crashes on some TensorRT engines whose post-processed class indices don't
    line up with their `names` dict on out-of-distribution inputs (a zeros
    dummy frame). That's a logging-only bug; real inference returns correct
    `boxes.xyxy` regardless. The engine-load banner (`Loading X for TensorRT
    inference...`, `[TRT] [I] Loaded engine size: ...`) is printed by Ultralytics
    and TensorRT themselves at construction time, so we still see device info.
    """
    print(f"[INFO] {name} warmup: device={device}, imgsz={imgsz}, half={half}")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        model(dummy, device=device, imgsz=imgsz, half=half, verbose=False)
    except Exception as e:  # never let a warmup failure kill a long video job
        print(f"[WARN] {name} warmup raised {type(e).__name__}: {e}. Continuing anyway.")


class PlateDetectorYOLO:
    def __init__(self, weights: str, device: str="cpu", conf: float=0.35, imgsz: int=960,
                 task: str = "detect"):
        # `task` is critical for TensorRT .engine files because the export format
        # does not carry Ultralytics task metadata. For .pt files Ultralytics
        # infers it correctly; passing it explicitly is harmless.
        self.model = YOLO(weights, task=task)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        # FP16 is only safe on CUDA; CPU/MPS must stay FP32.
        self.half = device not in ("cpu", "mps")
        _warmup_and_log(self.model, "PlateDetectorYOLO", self.device, self.imgsz, self.half)

    def __call__(self, frame):
        res = self.model(frame, device=self.device, conf=self.conf, imgsz=self.imgsz,
                         half=self.half, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        return [tuple(map(int, xy)) for xy in res.boxes.xyxy.cpu().numpy()]


class FaceDetectorHaar:
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)

    def __call__(self, frame) -> List[Tuple[int,int,int,int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

class FaceDetectorYOLO:
    def __init__(self, weights: str = "yolov8n-face.pt", device:str="cpu", conf:float=0.35, imgsz:int=960,
                 task: str = "pose"):
        # Default task is "pose" because the popular akanametov/yolov8-face model
        # is actually a YOLOv8-pose model (face bbox + 5 keypoints). When loaded
        # from a .engine file, Ultralytics cannot infer this and defaults to
        # "detect", which silently misinterprets keypoint coords as class scores
        # and floods the pipeline with 300 garbage detections per frame.
        # Users with a pure-detection face model should pass task="detect".
        self.model = YOLO(weights, task=task)
        self.device = device
        self.conf = max(0.25, conf)
        self.imgsz = imgsz
        self.half = device not in ("cpu", "mps")
        _warmup_and_log(self.model, "FaceDetectorYOLO", self.device, self.imgsz, self.half)

    def __call__(self, frame) -> List[Tuple[int,int,int,int]]:
        res = self.model(frame, device=self.device, conf=self.conf, imgsz=self.imgsz,
                         half=self.half, verbose=False)[0]
        if res.boxes is None or len(res.boxes)==0: return []
        return [tuple(map(int, xy)) for xy in res.boxes.xyxy.cpu().numpy()]
