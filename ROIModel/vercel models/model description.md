SpatialNet-Detect — Real-Time Pointing Gesture ROI Detection
A Next.js web app that runs SpatialNet-Fastest and YOLO-Fastest V2 deep learning models entirely in the browser to detect what object a person is pointing at, using their hand and finger gesture.

Live Demo: Deployed on Vercel

How It Works
The user points at an object with their finger. The app captures video frames from the device camera, runs AI inference on-device (no server), and draws bounding boxes around the hand, fingertip, and the target object (ROI) being pointed at. If the user is not pointing, it shows a "Non-Pointing" label with the hand bounding box.

Architecture Overview
Tap → Capture 60 Frames → Select 5 Best Frames → Gesture Model (Point/Non-Point) → SpatialNet Inference → Overlay
Internal Pipeline
1. Camera & Capture (CameraView.js)
Opens the rear camera at 640×480 via getUserMedia
On "Detect" tap, enters burst capture mode — records ~60 frames (~2 seconds at 30fps)
Frames are captured at 352×352 (gesture model's native resolution)
After capture completes, transitions to analysis phase
2. Key Frame Selection (frameSelector.js)
Laplacian Variance (blur scoring): Computes the variance of the Laplacian kernel on each frame. Higher variance = sharper image. Rejects blurry frames caused by hand motion.
Temporal Segment Networks (TSN): Divides the 60-frame burst into K=5 equal segments and picks the sharpest frame from each. This ensures temporal diversity while filtering blur.
Output: 5 high-quality, temporally diverse frames at 352×352
3. Gesture Detection (modelService.js — YOLO-Fastest V2)
Loads the YOLO-Fastest V2 binary gesture classifier (~475 KB ONNX, float32)
Input: 352×352, BGR channel order, normalized by /255, NCHW layout
Canvas provides RGBA (RGB order) — R and B channels are swapped to match the model's BGR training format
YOLO decoding: 6 anchors across 2 scales, YOLOv2/v3 bbox formula (sigmoid + grid for center, anchor * exp for size), sigmoid objectness + class probabilities, NMS
GESTURE_CONF_THRESH = 0.15, GESTURE_IOU_THRESH = 0.5
Detects pointing vs non-pointing gestures across all 5 key frames
If at least 1 frame detects a pointing gesture, proceeds to ROI detection
If no pointing detected, still runs SpatialNet and shows "Non-Pointing" with hand bbox
4. ROI Detection (modelService.js — SpatialNet-Fastest)
Loads the SpatialNet-Fastest ONNX model (~870 KB, float32)
Preprocessing per frame:
Resize from 352×352 to 160×160 via canvas
Normalize with ImageNet mean/std: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
Convert from HWC RGBA → NCHW RGB float32
Runs inference on all 5 key frames
Confidence-weighted averaging aggregates the 5 predictions into one stable result
5. Model Output (13 values + gesture flag)
Index	Field	Format	Description
0–3	Hand bbox	cx, cy, w, h	Hand bounding box (normalized 0–1)
4–5	Fingertip	x, y	Fingertip keypoint (normalized 0–1)
6–7	Pointing direction	sin θ, cos θ	Unit vector of pointing angle
8–11	ROI bbox	cx, cy, w, h	Target object bounding box (normalized 0–1)
12	Objectness	confidence	Detection confidence score (0–1)
—	gesture	'point' / 'none'	Gesture classification result
6. Visualization (DetectionOverlay.js)
Pointing mode:
Green box — Target ROI (the object being pointed at)
Blue dashed box — Hand bounding box
Red dot — Predicted fingertip position
Yellow dashed ray — Pointing direction from fingertip to ROI center
Confidence label shown above the ROI box
Non-pointing mode:
Red box — Hand bounding box
"Non-Pointing" label above the hand box
7. Temporal Filtering (temporalFilter.js) (available for real-time mode, not used in burst mode)
EMA Smoothing (α=0.35): Exponential moving average on bbox coordinates to reduce jitter
Confidence Hysteresis: Dual-threshold (enter=0.45, exit=0.25) prevents flickering detections
Frame Differencing: Skips inference on near-identical frames to save compute
The AI Models
SpatialNet-Fastest (~280K parameters, <1MB)
A lightweight multi-task network designed for edge deployment:

Backbone: ShuffleNetV2 (pretrained on ImageNet)
Feature Fusion: Lightweight FPN with SimAM attention
Spatial Awareness: CoordConv (appends X/Y coordinate grids to features)
Fingertip Head: Spatial Soft-Argmax — generates a heatmap from spatial features and extracts sub-pixel coordinates via differentiable integral regression
Detection Heads: Parallel-Context Sharing Head — independent task embeddings fused through a shared context mixer
Trained with CIoU loss (bounding boxes), Wing Loss (fingertip), ray alignment loss, and geometric consistency regularization
YOLO-Fastest V2 Gesture Classifier (~237K parameters, ~475KB)
A binary gesture detection model for pointing vs non-pointing:

Architecture: YOLO-Fastest V2 with 2 detection scales (stride 16 and 32)
Input: 352×352, /255 normalized, NCHW
Output: 6 tensors (reg, obj, cls per scale)
Classes: Non-Point (0), Point (1)
Anchors: 6 anchors across 2 scales
Training: AP=0.936, Precision=0.852, Recall=0.954, F1=0.900
Project Structure
app/
├── public/models/          # ONNX models + metadata
│   ├── model.onnx          # SpatialNet-Fastest (float32)
│   ├── model_metadata.json
│   ├── yolo_fastest_gesture_binary_fp32.onnx  # Gesture classifier (float32)
│   └── yolo_metadata.json
├── src/
│   ├── app/
│   │   ├── page.js         # Landing page
│   │   ├── layout.js       # Root layout
│   │   └── globals.css     # Styles
│   ├── components/
│   │   ├── CameraView.js   # Camera + capture workflow
│   │   └── DetectionOverlay.js  # Canvas rendering
│   └── lib/
│       ├── modelService.js     # ONNX loading + inference
│       ├── frameSelector.js    # Burst capture + blur filtering
│       └── temporalFilter.js   # EMA + hysteresis smoothing
└── package.json
Getting Started
npm install
npm run dev
Open http://localhost:3000. Works best on mobile devices with a rear camera.

Tech Stack
Next.js 16 — React framework (deployed on Vercel)
onnxruntime-web 1.24 — WASM-based ONNX inference in browser (single-threaded)
SpatialNet-Fastest — Custom lightweight multi-task CNN for pointing ROI detection
YOLO-Fastest V2 — Binary gesture classifier (point / non-point)
Deployment Notes
No cross-origin isolation headers (COEP/COOP) — they break getUserMedia on iOS Safari and aren't needed since WASM runs single-threaded
Works on iOS Safari, Android Chrome, and desktop browsers
All inference runs on-device — no server-side processing