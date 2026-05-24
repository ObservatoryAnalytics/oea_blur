# video-privacy-blur

Blur **license plates** and/or **faces** in videos with a single command.
Uses Ultralytics YOLO for detection and OpenCV for blurring.

## Features
- CPU / GPU / Apple MPS selection (`--device cpu|cuda|mps|auto`)
- Toggle plates and/or faces independently
- Two anonymization styles: **gaussian** blur or **pixelation**
- Works on files or webcams
- Minimal deps, no manual plate weights required

A sample test video showing blurred faces and license plates can be viewed here:
[Privacy Blur Demo YouTube](https://youtu.be/2lWWn4ubX78)

![alt text](image.png)


## Models

To run the tool, you need to download the following YOLOv8 models and place them in the `models/` folder:

- **Face detection model:** `yolov8n-face.pt`
  - Downloaded from [akanametov/yolo-face](https://github.com/akanametov/yolo-face)
  - Used to detect human faces with YOLOv8.
  - Can be replaced with any YOLOv8 face model you prefer.

- **License plate detection model:** `license_plate_detector.pt`
  - Downloaded from [Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)
  - Used to detect vehicle license plates with YOLOv8.
  - Can be swapped with any YOLOv8 license plate detector model you have trained or obtained.


## Quickstart

```bash
git clone git@github.com:MengWoods/video-privacy-blur.git
cd video-privacy-blur
# Crate a virtual Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Enable the privacy-blur CLI in editable mode
pip install -e .
```

### Run

```bash
# Plates and faces, Gaussian blur
privacy-blur --plate-weights models/license_plate_detector.pt \
             --face-detector yolo \
             --face-yolo-weights models/yolov8n-face.pt \
             --input input.mp4 --output out.mp4 --device auto --show

# Only plates
privacy-blur --plate-weights models/license_plate_detector.pt \
             --no-blur-faces \
             --input input.mp4 --output plates_only.mp4
```

### Examples

```bash
# Plates only, gaussian blur (default)
privacy-blur --input dashcam.mp4 --no-blur-faces

# Faces only, pixelate on GPU
privacy-blur --input street.mp4 --no-blur-plates --method pixelate --device cuda

# Use YOLO face model instead of Haar (better on varied angles)
privacy-blur --input crowd.mp4 --face-detector yolo --face-yolo-weights models/yolov8n-face.pt


### CLI Options

```bash
--input: path to video or camera index (0, 1, …)
--output: output video path (default out_blurred.mp4)
--device: auto (default), cpu, cuda, or mps
--imgsz: YOLO inference size (default 960)
--conf: YOLO confidence threshold (default 0.35)
--scale: expand boxes to ensure full coverage (default 1.35)
--show: show preview window
--blur-plates / --no-blur-plates: toggle plate blur
--blur-faces / --no-blur-faces: toggle face blur
--face-detector: haar (default) or yolo
--face-yolo-weights: local file or Ultralytics hub id
--plate-weights: local file for license plate model
--method: gaussian (default) or pixelate
--blur-strength: blur kernel divisor for gaussian (lower = stronger blur)
--pixelate-blocks: block count for pixelation (lower = stronger pixelation)
```

### Notes

- **GPU:** Only YOLO models use GPU. Haar face detection is CPU-only.
- **Weights:**
  - Plates: supply `--plate-weights` (e.g., `models/license_plate_detector.pt`).
  - Faces (YOLO): supply `--face-yolo-weights` (e.g., `models/yolov8n-face.pt`).
- **Performance:**
  - Lower `--imgsz` for faster speed; raise for accuracy.
  - Increase `--conf` to reduce false positives.
  - Tweak `--scale` if edges of plates/faces are not fully covered.
- **Privacy:**
  - Pixelation generally offers stronger privacy than light Gaussian blur.
  - For maximum anonymity, use large pixel blocks (`--pixelate-blocks 8–12`) or low blur divisor (`--blur-strength 4–6`).





## Performance progression (test_1.mkv: 239 frames @ 7 fps, 4K)

GTX 1660 Ti, imgsz=1280, --sensor (CCW rotation for upright detection):

| Stage | Time | fps | Speedup |
|---|---|---|---|
| Original (mp4v + FP32) | 156.60 s | 1.53 | 1.00x |
| + FP16 (`half=True`)              | 94.11 s  | 2.54 | 1.66x |
| + h264_nvenc writer               | 75.75 s  | 3.16 | 2.07x |
| + TensorRT engines (.engine)      | 75.32 s  | 3.17 | 2.08x |
| + Threaded reader + writer        | **43.81 s**  | **5.46** | **3.57x** |

A 30-minute source video at 7 fps (12,600 frames) now projects to ~38 min
of processing instead of ~138 min at the original baseline.

## Performance optimization notes

1. **Build TensorRT engines once per GPU + per TensorRT version**. They are not portable:

   ```bash
   yolo export model=models/license_plate_detector.pt format=engine half=True imgsz=1280 device=0
   yolo export model=models/yolov8n-face.pt        format=engine half=True imgsz=1280 device=0
   ```

   These produce `models/license_plate_detector.engine` and `models/yolov8n-face.engine`.
   Each export takes ~5-10 min and ~500 MiB VRAM. Rebuild if you change GPU, change TensorRT major version, or move to a different machine.

2. **Task hint is required for engines**. The akanametov yolov8-face model is actually a `task=pose` model (face bbox + 5 keypoints). TensorRT `.engine` files lose Ultralytics task metadata, so the CLI passes `--face-task pose` by default. If you use a pure-detection face model, pass `--face-task detect` instead.

3. **FFmpeg with NVENC is required** for the fastest writer. On Windows:

   ```powershell
   winget install --id=Gyan.FFmpeg -e
   ```

   Verify with `ffmpeg -encoders | findstr nvenc`.

4. **--sensor rotation matters for detection quality** in this dataset (89% fewer detections without it). The threaded pipeline hides its cost behind GPU work, so it's effectively free at runtime.


### How to run code

The fastest production command (TensorRT + NVENC + threaded pipeline, default flags):

```powershell
privacy-blur --plate-weights models/license_plate_detector.engine --face-detector yolo --face-yolo-weights models/yolov8n-face.engine --input input/test_1.mkv --output output/test_1.mkv --conf 0.35 --imgsz 1280 --device auto --sensor
```

Useful flags:

- `--encoder {nvenc, hevc_nvenc, x264, mp4v}` - encoder choice (default `nvenc`).
- `--reader {opencv, ffmpeg}` - input decoder (default `opencv`, faster on most files).
- `--threads` / `--no-threads` - threaded pipeline on/off (default on).
- `--queue-size N` - frames buffered between threads (default 4, larger is rarely better).
- `--cq N` - NVENC constant quality 18-28 (default 23).
- `--preset {p1..p7}` - NVENC speed/quality preset (default `p4`).
- `--face-task {pose, detect}` - required to match the face model's actual task.
- `--plate-task {detect, pose}` - required to match the plate model's actual task.
- `--show` - preview window while processing.
- `--no-blur-plates` / `--no-blur-faces` - disable individual detectors.
