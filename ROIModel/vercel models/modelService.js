/**
 * SpatialNet-Fastest ONNX Model Service
 * ======================================
 * Loads the ONNX model via onnxruntime-web (WASM backend)
 * and runs real-time inference on camera frames.
 *
 * ONNX input:  [1, 3, 160, 160]  (NCHW, float32, ImageNet normalized)
 * ONNX output: [1, 13]           (hand[4], fingertip[2], angle[2], roi[4], objectness[1])
 */

const MODEL_PATH = '/models/model.onnx';
const GESTURE_MODEL_PATH = '/models/yolo_fastest_gesture_binary_fp32-1.onnx';

const INPUT_SIZE = 160;
const GESTURE_INPUT_SIZE = 352;

const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

// YOLO-Fastest V2 Constants
const GESTURE_ANCHORS = [
  [[23.06, 39.50], [43.81, 70.43], [107.09, 166.69]], // STRIDE 16
  [[146.80, 246.85], [187.71, 149.59], [224.67, 245.59]] // STRIDE 32
];
const GESTURE_STRIDES = [16, 32];
const GESTURE_NUM_CLASSES = 2; // 0: Non-Point, 1: Point
const GESTURE_CONF_THRESH = 0.15;
const GESTURE_IOU_THRESH = 0.5;

// Output tensor layout (13 values total)
const SLICES = {
  hand:       { start: 0,  end: 4  },  // cx, cy, w, h
  fingertip:  { start: 4,  end: 6  },  // x, y
  angle:      { start: 6,  end: 8  },  // sin, cos
  roi:        { start: 8,  end: 12 },  // cx, cy, w, h
  objectness: { start: 12, end: 13 },  // confidence
};

class ModelService {
  constructor() {
    this.session = null;
    this.gestureSession = null;
    this.isLoading = false;
    this.isReady = false;
    this.ort = null;
    // Offscreen canvas for preprocessing
    this._offscreen = null;
    this._offCtx = null;
    // Reusable input buffers
    this._inputBuffer = null;
    this._gestureInputBuffer = null;
    // Session options for low-memory devices
    this._sessionOpts = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
    };
  }

  async getModelSize() {
    try {
      const resp1 = await fetch(MODEL_PATH, { method: 'HEAD' });
      const bytes1 = parseInt(resp1.headers.get('content-length'), 10) || 0;
      
      const resp2 = await fetch(GESTURE_MODEL_PATH, { method: 'HEAD' });
      const bytes2 = parseInt(resp2.headers.get('content-length'), 10) || 0;

      const totalBytes = bytes1 + bytes2;
      if (totalBytes === 0) return null;
      if (totalBytes >= 1048576) return `${(totalBytes / 1048576).toFixed(2)} MB`;
      return `${(totalBytes / 1024).toFixed(0)} KB`;
    } catch {
      return null;
    }
  }

  async load() {
    if (this.isReady || this.isLoading) return;
    this.isLoading = true;

    try {
      // Dynamic import to avoid SSR issues
      const ort = await import('onnxruntime-web');
      this.ort = ort;

      // Configure WASM backend — version must match installed npm package
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/';
      // Use single thread to avoid cross-origin isolation requirements
      ort.env.wasm.numThreads = 1;

      // Warm up the WASM backend by loading just the smaller model (SpatialNet ~870KB).
      // On low-memory devices (iOS WebViews), loading both models simultaneously
      // causes OOM. Instead we load models on-demand in predictBatch().
      this.session = await ort.InferenceSession.create(MODEL_PATH, this._sessionOpts);

      // Create offscreen canvas for resizing (sized to the larger of the two inputs)
      this._offscreen = document.createElement('canvas');
      this._offscreen.width = Math.max(INPUT_SIZE, GESTURE_INPUT_SIZE);
      this._offscreen.height = Math.max(INPUT_SIZE, GESTURE_INPUT_SIZE);
      this._offCtx = this._offscreen.getContext('2d', { willReadFrequently: true });

      // Pre-allocate input buffers
      this._inputBuffer = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);
      this._gestureInputBuffer = new Float32Array(1 * 3 * GESTURE_INPUT_SIZE * GESTURE_INPUT_SIZE);

      this.isReady = true;
      console.log('[ModelService] ONNX runtime initialized, SpatialNet loaded');
    } catch (err) {
      console.error('[ModelService] Failed to load model:', err);
      throw err;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Ensure the gesture model session is loaded.
   * On low-memory devices we load it on-demand to avoid OOM.
   */
  async _ensureGestureSession() {
    if (this.gestureSession) return;
    this.gestureSession = await this.ort.InferenceSession.create(
      GESTURE_MODEL_PATH, this._sessionOpts
    );
  }

  /**
   * Release the gesture model session to free memory.
   */
  _releaseGestureSession() {
    if (this.gestureSession) {
      this.gestureSession.release?.();
      this.gestureSession = null;
    }
  }

  /**
   * Preprocesses an ImageData object: normalize with ImageNet stats,
   * and arrange as NCHW Float32Array.
   * Overloaded to handle either video element or direct ImageData.
   */
  _preprocess(source) {
    const ctx = this._offCtx;
    const buf = this._inputBuffer;

    let pixels;
    if (source instanceof ImageData) {
        // ImageData may be larger (e.g. 352×352) — resize to 160×160 via canvas
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = source.width;
        tmpCanvas.height = source.height;
        const tmpCtx = tmpCanvas.getContext('2d');
        tmpCtx.putImageData(source, 0, 0);
        ctx.drawImage(tmpCanvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
        const resized = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
        pixels = resized.data;
    } else {
        // Draw video frame resized to 160x160
        ctx.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
        const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
        pixels = imageData.data;
    }

    const numPixels = INPUT_SIZE * INPUT_SIZE;
    const rOffset = 0;
    const gOffset = numPixels;
    const bOffset = numPixels * 2;

    // Convert RGBA HWC → RGB NCHW, normalized with ImageNet mean/std
    for (let i = 0; i < numPixels; i++) {
      const r = pixels[i * 4]     / 255.0;
      const g = pixels[i * 4 + 1] / 255.0;
      const b = pixels[i * 4 + 2] / 255.0;

      buf[rOffset + i] = (r - MEAN[0]) / STD[0];
      buf[gOffset + i] = (g - MEAN[1]) / STD[1];
      buf[bOffset + i] = (b - MEAN[2]) / STD[2];
    }

    return buf;
  }

  /**
   * Preprocesses an ImageData object for the Gesture model: 
   * resize to 352x352, float32, / 255.0, NCHW BGR layout.
   * Model was trained on BGR input (cv2.imread), so we swap R↔B from canvas RGB.
   */
  _preprocessGesture(source) {
    const ctx = this._offCtx;
    const buf = this._gestureInputBuffer;

    let pixels;
    if (source instanceof ImageData) {
        // ImageData may be smaller (e.g. 160×160) — resize to 352×352 via canvas
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = source.width;
        tmpCanvas.height = source.height;
        const tmpCtx = tmpCanvas.getContext('2d');
        tmpCtx.putImageData(source, 0, 0);
        ctx.drawImage(tmpCanvas, 0, 0, GESTURE_INPUT_SIZE, GESTURE_INPUT_SIZE);
        const resized = ctx.getImageData(0, 0, GESTURE_INPUT_SIZE, GESTURE_INPUT_SIZE);
        pixels = resized.data;
    } else {
        ctx.drawImage(source, 0, 0, GESTURE_INPUT_SIZE, GESTURE_INPUT_SIZE);
        const imageData = ctx.getImageData(0, 0, GESTURE_INPUT_SIZE, GESTURE_INPUT_SIZE);
        pixels = imageData.data;
    }

    const numPixels = GESTURE_INPUT_SIZE * GESTURE_INPUT_SIZE;
    const ch0Offset = 0;             // B channel (was R in canvas)
    const ch1Offset = numPixels;     // G channel
    const ch2Offset = numPixels * 2; // R channel (was B in canvas)

    for (let i = 0; i < numPixels; i++) {
      // Canvas gives RGBA; model expects BGR order
      buf[ch0Offset + i] = pixels[i * 4 + 2] / 255.0; // B
      buf[ch1Offset + i] = pixels[i * 4 + 1] / 255.0; // G
      buf[ch2Offset + i] = pixels[i * 4]     / 255.0; // R
    }

    return buf;
  }

  /**
   * Sigmoid activation function
   */
  _sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Process YOLO-Fastest V2 output tensors to extract bounding boxes.
   * Outputs from the model are typical YOLOv5/Fastest format:
   * scale1: reg [1, 12, 22, 22], obj [1, 3, 22, 22], cls [1, 2, 22, 22]
   * scale2: reg [1, 12, 11, 11], obj [1, 3, 11, 11], cls [1, 2, 11, 11]
   */
  _extractDetections(results) {
    const boxes = [];
    const outputNames = this.gestureSession.outputNames;

    // We have 2 scales (stride 16 and stride 32).
    for (let scaleIdx = 0; scaleIdx < 2; scaleIdx++) {
      const stride = GESTURE_STRIDES[scaleIdx];
      const anchors = GESTURE_ANCHORS[scaleIdx];
      const gridSize = GESTURE_INPUT_SIZE / stride; // 22 (for 352/16) or 11 (for 352/32)
      const numAnchors = anchors.length; // 3
      const gridArea = gridSize * gridSize;

      // Match outputs by shape: reg has 12 channels, obj has 3, cls has 2 (shared across anchors)
      // Filter to outputs matching this scale's spatial size
      const scaleOutputs = outputNames
        .map(name => ({ name, tensor: results[name] }))
        .filter(o => {
          const dims = o.tensor.dims;
          return dims[dims.length - 1] === gridSize && dims[dims.length - 2] === gridSize;
        });

      let regOutput, objOutput, clsOutput;
      for (const o of scaleOutputs) {
        const channels = o.tensor.dims[1];
        if (channels === numAnchors * 4) regOutput = o.tensor.data;      // 12 channels
        else if (channels === numAnchors) objOutput = o.tensor.data;      // 3 channels
        else if (channels === GESTURE_NUM_CLASSES) clsOutput = o.tensor.data; // 2 channels (shared)
      }

      if (!regOutput || !objOutput || !clsOutput) {
        console.warn(`[ModelService] Could not match YOLO outputs for scale ${scaleIdx} (grid ${gridSize})`);
        continue;
      }

      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          const gridIdx = y * gridSize + x;

          for (let a = 0; a < numAnchors; a++) {
            // Memory layout is NCHW: Channel is the outer loop regarding spatial dimensions.
            // For Objectness (3 channels total, 1 per anchor):
            const objChannel = a;
            const objIdx = objChannel * gridArea + gridIdx;
            const objScore = this._sigmoid(objOutput[objIdx]);

            if (objScore < GESTURE_CONF_THRESH) continue;

            // For Classes (2 channels total, shared across all anchors):
            let maxClassProb = 0;
            let classId = -1;
            
            for (let c = 0; c < GESTURE_NUM_CLASSES; c++) {
              const clsChannel = c;
              const clsIdx = clsChannel * gridArea + gridIdx;
              
              const clsProb = this._sigmoid(clsOutput[clsIdx]);
              if (clsProb > maxClassProb) {
                maxClassProb = clsProb;
                classId = c;
              }
            }

            const confidence = objScore * maxClassProb;
            if (confidence < GESTURE_CONF_THRESH) continue;

            // For Regression (12 channels total, 4 per anchor):
            // Order is typically tx, ty, tw, th
            const regChannelBase = a * 4;
            const tx = regOutput[(regChannelBase + 0) * gridArea + gridIdx];
            const ty = regOutput[(regChannelBase + 1) * gridArea + gridIdx];
            const tw = regOutput[(regChannelBase + 2) * gridArea + gridIdx];
            const th = regOutput[(regChannelBase + 3) * gridArea + gridIdx];

            // Decode bounding box (YOLOv2/v3 formula)
            const cx = (this._sigmoid(tx) + x) * stride;
            const cy = (this._sigmoid(ty) + y) * stride;
            const w = anchors[a][0] * Math.exp(Math.min(Math.max(tw, -5), 5));
            const h = anchors[a][1] * Math.exp(Math.min(Math.max(th, -5), 5));

            // Normalize coordinates to 0-1
            boxes.push({
              x: (cx - w / 2) / GESTURE_INPUT_SIZE,
              y: (cy - h / 2) / GESTURE_INPUT_SIZE,
              w: w / GESTURE_INPUT_SIZE,
              h: h / GESTURE_INPUT_SIZE,
              confidence,
              classId
            });
          }
        }
      }
    }
    return boxes;
  }

  /**
   * Calculate Intersection over Union (IoU) between two boxes
   */
  _iou(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = boxA.w * boxA.h;
    const boxBArea = boxB.w * boxB.h;

    const iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
  }

  /**
   * Non-Maximum Suppression (NMS)
   */
  _nms(boxes, iouThreshold) {
    if (boxes.length === 0) return [];

    // Sort by confidence descending
    boxes.sort((a, b) => b.confidence - a.confidence);

    const selectedBoxes = [];
    const active = new Array(boxes.length).fill(true);

    for (let i = 0; i < boxes.length; i++) {
      if (!active[i]) continue;
      
      const boxA = boxes[i];
      selectedBoxes.push(boxA);

      for (let j = i + 1; j < boxes.length; j++) {
        if (!active[j]) continue;
        
        const boxB = boxes[j];
        if (boxA.classId === boxB.classId && this._iou(boxA, boxB) > iouThreshold) {
          active[j] = false;
        }
      }
    }

    return selectedBoxes;
  }

  /**
   * Run inference on the gesture model.
   * Returns true if a "Point" gesture is detected.
   */
  async predictGesture(videoElement) {
    if (!this.isReady) return false;
    await this._ensureGestureSession();

    const ort = this.ort;

    try {
      const inputData = this._preprocessGesture(videoElement);
      const inputTensor = new ort.Tensor('float32', inputData, [1, 3, GESTURE_INPUT_SIZE, GESTURE_INPUT_SIZE]);

      const gestureInputName = this.gestureSession.inputNames[0];
      const feeds = { [gestureInputName]: inputTensor };
      const results = await this.gestureSession.run(feeds);

      const boxes = this._extractDetections(results);
      const nmsBoxes = this._nms(boxes, GESTURE_IOU_THRESH);

      // Find the best pointing detection (classId 1)
      const pointBoxes = nmsBoxes.filter(b => b.classId === 1);
      if (pointBoxes.length > 0) {
        return { pointing: true, confidence: pointBoxes[0].confidence };
      }
      // Return best non-pointing confidence if no pointing found
      const bestConf = nmsBoxes.length > 0 ? nmsBoxes[0].confidence : 0;
      return { pointing: false, confidence: bestConf };

    } catch (err) {
      console.error('[ModelService] Gesture inference error:', err);
      return { pointing: false, confidence: 0 }; 
    }
  }

  /**
   * Run inference on a video element.
   * Returns parsed predictions or null if model isn't ready.
   */
  async predict(videoElement) {
    if (!this.isReady || !this.session) return null;

    const ort = this.ort;

    try {
      // Preprocess
      const inputData = this._preprocess(videoElement);

      // Create ONNX tensor [1, 3, 160, 160]
      const inputTensor = new ort.Tensor('float32', inputData, [1, 3, INPUT_SIZE, INPUT_SIZE]);

      // Run inference — use dynamic input name from the model
      const mainInputName = this.session.inputNames[0];
      const feeds = { [mainInputName]: inputTensor };
      const results = await this.session.run(feeds);

      // Get output tensor
      const outputName = this.session.outputNames[0];
      const output = results[outputName].data;

      // Parse output into structured predictions
      return {
        hand: {
          cx: output[SLICES.hand.start],
          cy: output[SLICES.hand.start + 1],
          w:  output[SLICES.hand.start + 2],
          h:  output[SLICES.hand.start + 3],
        },
        fingertip: {
          x: output[SLICES.fingertip.start],
          y: output[SLICES.fingertip.start + 1],
        },
        angle: {
          sin: output[SLICES.angle.start],
          cos: output[SLICES.angle.start + 1],
        },
        roi: {
          cx: output[SLICES.roi.start],
          cy: output[SLICES.roi.start + 1],
          w:  output[SLICES.roi.start + 2],
          h:  output[SLICES.roi.start + 3],
        },
        objectness: output[SLICES.objectness.start],
      };
    } catch (err) {
      console.error('[ModelService] Inference error:', err);
      return null;
    }
  }

  /**
   * Run inference on a batch of ImageData frames and aggregate results.
   * Uses confidence-weighted averaging for stable final output.
   * Now includes a pre-filtering step using the gesture detection model.
   * @param {ImageData[]} frames - Array of images to process
   */
  async predictBatch(frames) {
    if (!this.isReady || frames.length === 0) return null;

    // --- 1. Gesture Pre-Filtering ---
    // Release SpatialNet to free memory for the gesture model on low-memory devices
    const hadSession = !!this.session;
    if (this.session) { this.session.release?.(); this.session = null; }

    let frameGestures = 0;
    let maxGestureConf = 0;
    try {
      await this._ensureGestureSession();
      for (const frame of frames) {
        const result = await this.predictGesture(frame);
        if (result.pointing) frameGestures++;
        if (result.confidence > maxGestureConf) maxGestureConf = result.confidence;
      }
    } finally {
      // Release gesture model and reload SpatialNet
      this._releaseGestureSession();
      this.session = await this.ort.InferenceSession.create(MODEL_PATH, this._sessionOpts);
    }

    // Require at least 1 frame to detect a pointing gesture.
    const isPointing = frameGestures >= 1;

    // --- 2. Main Spatial Inference ---
    const predictions = [];
    
    // Process all frames
    for (const frame of frames) {
       const pred = await this.predict(frame);
       if (pred) predictions.push(pred);
    }

    if (predictions.length === 0) return null;

    // Aggregate results: Confidence-Weighted Averaging
    let totalWeight = 0;
    let maxConfidence = 0;

    const aggregated = {
        hand: { cx: 0, cy: 0, w: 0, h: 0 },
        fingertip: { x: 0, y: 0 },
        angle: { sin: 0, cos: 0 },
        roi: { cx: 0, cy: 0, w: 0, h: 0 },
        objectness: 0
    };

    for (const p of predictions) {
        // Use objectness score as weight
        const weight = p.objectness;
        totalWeight += weight;
        maxConfidence = Math.max(maxConfidence, p.objectness);

        // Weighted sum for hand
        aggregated.hand.cx += p.hand.cx * weight;
        aggregated.hand.cy += p.hand.cy * weight;
        aggregated.hand.w += p.hand.w * weight;
        aggregated.hand.h += p.hand.h * weight;

        // Weighted sum for fingertip
        aggregated.fingertip.x += p.fingertip.x * weight;
        aggregated.fingertip.y += p.fingertip.y * weight;

        // Weighted sum for angle
        aggregated.angle.sin += p.angle.sin * weight;
        aggregated.angle.cos += p.angle.cos * weight;

        // Weighted sum for ROI
        aggregated.roi.cx += p.roi.cx * weight;
        aggregated.roi.cy += p.roi.cy * weight;
        aggregated.roi.w += p.roi.w * weight;
        aggregated.roi.h += p.roi.h * weight;
    }

    if (totalWeight > 0) {
        // Normalize by total weight
        aggregated.hand.cx /= totalWeight;
        aggregated.hand.cy /= totalWeight;
        aggregated.hand.w /= totalWeight;
        aggregated.hand.h /= totalWeight;

        aggregated.fingertip.x /= totalWeight;
        aggregated.fingertip.y /= totalWeight;

        // Normalize angle vector
        const angleNorm = Math.sqrt(aggregated.angle.sin ** 2 + aggregated.angle.cos ** 2) || 1;
        aggregated.angle.sin /= angleNorm;
        aggregated.angle.cos /= angleNorm;

        aggregated.roi.cx /= totalWeight;
        aggregated.roi.cy /= totalWeight;
        aggregated.roi.w /= totalWeight;
        aggregated.roi.h /= totalWeight;
        
        // Final confidence is the max confidence found in the burst
        aggregated.objectness = maxConfidence;
    } else {
        // If all weights are 0, just use the first prediction
        return { ...predictions[0], gesture: isPointing ? 'point' : 'none', gestureConf: maxGestureConf };
    }

    aggregated.gesture = isPointing ? 'point' : 'none';
    aggregated.gestureConf = maxGestureConf;
    return aggregated;
  }

  dispose() {
    if (this.session) {
      this.session.release?.();
      this.session = null;
    }
    this._releaseGestureSession();
    this.isReady = false;
  }
}

// Singleton
let serviceInstance = null;

export function getModelService() {
  if (!serviceInstance) {
    serviceInstance = new ModelService();
  }
  return serviceInstance;
}

export { INPUT_SIZE, SLICES };
export default ModelService;
