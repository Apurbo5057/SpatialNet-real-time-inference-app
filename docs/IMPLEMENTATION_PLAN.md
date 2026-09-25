# Master Implementation Plan

Status: draft · Scope: feature 1, on-device object identification inside the ROI, plus groundwork for the features that follow.

## 1. Where the app stands today

The app is a single-activity Jetpack Compose app. It has one CameraX `ImageAnalysis` stream and one ONNX Runtime model.

```text
ImageAnalysis (back camera, KEEP_ONLY_LATEST, default ~640x480)
  -> InferenceViewModel.processFrame()          [Dispatchers.Default, AtomicBoolean gate]
       imageProxy.toBitmap() -> rotateBitmap()  => rotated full frame (kept for display + capture)
       SpatialNetModel.predict(rotated)         => ROIResult (hand, fingertip, angle, roi + 3 gates)
       applyTemporalSmoothing()                 => isPointing (5-frame window)
       _result.value = CombinedResult(frame, roiResult, isPointing, ms)
  -> InferenceScreen
       Image(frame, Fit) + Canvas overlay        (Compose drawing)
       Capture -> captureWithOverlays()          (android.graphics drawing, duplicated logic)
```

| File | Role | Relevance to object identification |
| --- | --- | --- |
| [SpatialNetModel.kt](../app/src/main/java/com/example/roidetection/SpatialNetModel.kt) | ORT session, 160x160 ImageNet-normalized NCHW preprocessing, output parsing | Preprocessing and session setup can be shared with a classifier |
| [InferenceViewModel.kt](../app/src/main/java/com/example/roidetection/InferenceViewModel.kt) | Frame gate, rotation, inference, smoothing, FPS | The classifier is scheduled here |
| [ModelData.kt](../app/src/main/java/com/example/roidetection/ModelData.kt) | `ROIResult`, `BBox`, `Corners`, `Point`, `Direction` | Gains an `ObjectResult` |
| [ModelInfo.kt](../app/src/main/java/com/example/roidetection/ModelInfo.kt) | Asset digest badge, assumes one model (`SPATIALNET_ASSET`) | Must handle two models |
| [ui/InferenceScreen.kt](../app/src/main/java/com/example/roidetection/ui/InferenceScreen.kt) | Live overlay, stats bar, Capture, gallery save (845 lines) | Label drawing is needed in two places today |
| [app/build.gradle.kts](../app/build.gradle.kts) | `fp32` / `fromFp16` flavors; `Sync` tasks copy one model per flavor into generated assets | Needs a classifier asset task for both flavors |

Facts that shape the design:

- ROI coordinates are normalized against the whole frame. The 160x160 model input is a direct stretch of that frame, so an ROI maps to rotated-bitmap pixels with `x * width`, `y * height`. No letterbox correction is needed.
- The rotated frame bitmap is kept alive by `CombinedResult` and is never recycled, so a crop can be taken from it safely after inference.
- The analysis resolution is whatever CameraX picks (usually 640x480). A typical ROI is 0.06–0.35 of the frame on each side, which can mean a crop only 30–200 px wide. Small crops are the main accuracy risk (see §7).
- SpatialNet takes roughly 70 ms per frame on the tested phone. Frames are dropped while a frame is processing.
- The overlay code is written twice, once in Compose (`Canvas`) and once in `android.graphics` for Capture. Every new overlay element, the object label included, would have to be added twice.
- There are no unit or instrumentation tests, and no test dependencies are declared.

## 2. Feature 1 goal

When SpatialNet reports a target region (`POINTING` with an ROI box), the app names the object inside that box, for example "television · 84%". It shows the name on the live view and in the saved capture. Everything runs on the phone with no network access.

Acceptance criteria:

1. Works in airplane mode. The app never requests the `INTERNET` permission for this feature.
2. The label appears within about 500 ms of the ROI settling on an object and does not flicker between classes from frame to frame.
3. SpatialNet processed-frame FPS drops by no more than about 10% while identification is on.
4. If the classifier fails to load, ROI detection keeps working and the UI shows that identification is unavailable.
5. APK size grows by no more than about 12 MB per flavor.
6. On a curated set of at least 50 real ROI crops from this app, top-1 accuracy at the friendly-label level (§4.3) is at least 70%, and top-3 accuracy is at least 85%. These targets can be revised after the baseline measurement in Phase 1.

## 3. Approach options and recommendation

| Option | How it works | Vocabulary | Size | Fit with this repo |
| --- | --- | --- | --- | --- |
| **A. ONNX image classifier (recommended)** | MobileNetV3-Large (ImageNet-1k) on the ROI crop, using the existing ONNX Runtime | 1000 fixed classes, mapped to friendlier names | ~11 MB with FP16-stored weights, ~5.5 MB INT8 | Same runtime, same asset/digest/flavor pattern, no new SDK |
| B. ML Kit Image Labeling (bundled) | Google's on-device base labeler | ~400 generic labels | ~6 MB | New SDK and a second inference stack; less control over preprocessing and thresholds |
| C. Open-vocabulary (MobileCLIP-S0 image encoder + precomputed text embeddings) | Cosine similarity against a label list you choose | Any list, editable without retraining | ~40 MB+ | Best long-term flexibility; heavier; good Phase 6 upgrade |
| D. Object detector (YOLO / EfficientDet) on the ROI | Detect objects inside the crop | COCO's 80 classes | 4–12 MB | Duplicates SpatialNet's localization; small vocabulary |

**Recommendation: Option A, behind an `ObjectClassifier` interface.** It reuses ONNX Runtime, preprocessing, and the model-badge mechanism the app already has. It covers household items that ImageNet includes (television, monitor, remote control, laptop, lamp shade, switch, bottle, cup, chair, and others). The interface keeps Options B and C as drop-in replacements if ImageNet's vocabulary turns out to be too narrow.

## 4. Design

### 4.1 Pipeline

```text
processFrame (unchanged hot path)
  SpatialNet -> ROIResult -> temporal smoothing -> CombinedResult
                                   |
                                   v  if smoothed pointing && roiBBox != null && scheduler says "go"
                        RoiCropper.crop(rotatedFrame, roiBBox)   (copy, padded, square, clamped)
                                   |
                                   v  classifierDispatcher (single thread, own AtomicBoolean gate)
                        ObjectClassifier.classify(crop) -> top-K (label, prob)
                                   |
                        LabelSmoother.update(topK, roiBBox) -> ObjectResult
                                   |
                        _objectResult (StateFlow)  ->  merged into UI state
```

Classification never blocks SpatialNet. It runs on its own single-thread dispatcher (`Dispatchers.Default.limitedParallelism(1)`) with its own busy flag. If the classifier is still working when a new ROI arrives, that ROI is skipped.

### 4.2 Components (new package `com.example.roidetection.objectid`)

| Component | Responsibility | Key details |
| --- | --- | --- |
| `ObjectClassifier` (interface) | `initialize()`, `classify(Bitmap): List<LabelScore>`, `close()`, `modelInfo` | Lets ML Kit or MobileCLIP replace the ONNX model later |
| `OnnxImageClassifier` | ORT session for the classifier | 224x224 input, ImageNet mean/std (same as SpatialNet), softmax, top-5. Uses 2 intra-op threads so it does not starve SpatialNet's 4. Loads `labels.txt` from assets |
| `RoiCropper` | Normalized `BBox` -> pixel crop from the rotated frame | Pads by 15–20% for context, expands the shorter side toward a square so resizing does not distort it, clamps to the frame, and rejects crops under a minimum size (e.g. 32 px) |
| `ClassificationScheduler` | Decides when to classify | Classifies when the ROI first appears, then at most every ~250 ms, or immediately when the box moves (IoU with the last classified box < 0.5). No classification when not pointing |
| `LabelSmoother` | Stabilizes labels | Exponential moving average of probabilities per friendly label over the current ROI "track". Resets when the ROI disappears for > ~600 ms or jumps (IoU < 0.3). A label is shown only if its smoothed score is ≥ 0.35; otherwise the UI shows "Unknown object" |
| `LabelMap` | ImageNet class -> friendly label | JSON asset that merges near-duplicates (e.g. `television`, `monitor`, `screen` -> "TV / screen"), renames awkward classes, and hides classes that make no sense indoors. Scores for merged classes are summed |

### 4.3 Data model changes ([ModelData.kt](../app/src/main/java/com/example/roidetection/ModelData.kt))

```kotlin
data class LabelScore(val label: String, val score: Float)

data class ObjectResult(
    val top: LabelScore?,            // null => "Unknown object"
    val candidates: List<LabelScore>,// smoothed top-3, for a detail view
    val box: BBox,                   // ROI the label was computed for
    val computedAtMs: Long,
    val inferenceTimeMs: Long
)
```

`CombinedResult` gains `objectResult: ObjectResult?`. The overlay attaches the label to the current ROI box only when the box it was computed for still overlaps it (IoU ≥ 0.3) and it is less than about 1 s old. Otherwise the overlay shows "Identifying…". This prevents a stale label from following the box onto a different object.

### 4.4 UI

- **Live view:** a filled label chip on the green ROI box, e.g. "TV / screen 84%". "Identifying…" appears while waiting and "Unknown object" appears below the threshold.
- **Stats bar:** adds the classifier time next to the SpatialNet time, e.g. `70ms + 18ms`.
- **Capture:** the saved JPEG includes the chip, and the stats bar text includes the label.
- **Toggle:** an "Identify objects" switch in the top bar that turns classification off to save battery or compare FPS. A per-session setting is enough at first; DataStore can be added later.
- **Badge:** `ModelBadge` lists both models with their digests.
- **Optional:** tapping the chip shows the top-3 candidates.

### 4.5 Build and assets

- New directory `ObjectModel/` containing `mobilenet_v3_large_fp16w.onnx`, `imagenet_labels.txt`, and `label_map.json`.
- A new `Sync` task copies these into a generated `main` asset directory, so **both** flavors bundle the same classifier. Classifier results then stay comparable across the FP32 and FP16 SpatialNet builds.
- `ModelInfo` changes from a single `SPATIALNET_ASSET` to a registry, for example `ModelInfo.read(context, BuildConfig.CLASSIFIER_ASSET)`.
- Add `testImplementation(junit)` and `androidTestImplementation(androidx.test.ext:junit, espresso/runner)` to `libs.versions.toml`.

### 4.6 Model preparation (offline, Python)

New `tools/classifier/` folder, reproducible and documented like `image_walkthrough/`:

1. `export_classifier.py` exports torchvision `mobilenet_v3_large(weights=IMAGENET1K_V2)` to ONNX (opset 17), with input `image [1,3,224,224]` and output `logits [1,1000]`, and writes `imagenet_labels.txt`.
2. `convert_fp16_weights.py` stores the weights as FLOAT16 with float32 I/O, following the same pattern as the existing `spatialnet_fastest_from_fp16.onnx`. Later, optionally, `quantize_int8.py` applies static QDQ quantization calibrated on real ROI crops.
3. `verify_parity.py` compares PyTorch and ORT outputs on sample crops (max abs logit diff, top-1 agreement) and prints the SHA-256 prefix the app badge will show.

## 5. Phased plan

### Phase 0: Foundations (refactor, no behavior change)

1. Extract `ImageTensor.toNchwFloatBuffer(bitmap, size, mean, std)` from `SpatialNetModel.predict`. Rewrite it as a single pass over the pixels (the current loop reads every pixel three times) and reuse the `FloatBuffer` and `IntArray`.
2. Extract one `OverlayRenderer` that draws on an `android.graphics.Canvas`, given a transform (scale and offset). Compose calls it through `drawContext.canvas.nativeCanvas`, and Capture calls it with a scale of 1. This removes the duplicated drawing code before a label is added to it. Cache `Paint` objects instead of allocating them each frame.
3. Change `ModelInfo` to support multiple assets.
4. Add test dependencies and a first unit test for `BBox.toCorners().clamp()`.

Exit: the app looks and behaves identically, and `./gradlew testFp32DebugUnitTest` passes.

### Phase 1: Model selection and baseline (Python plus device check)

1. Export MobileNetV3-Large and, as a fallback, MobileNetV3-Small. Run parity checks.
2. Build the evaluation set: collect 50–100 ROI crops from real app captures (a debug-only "save ROI crop" action is useful here) and label them by hand.
3. Measure top-1 and top-3 accuracy on the crops with 0%, 15%, and 30% padding, and at 640x480 versus 1280x720 analysis resolution.
4. Write `label_map.json` based on the classes that actually appear.

Exit: chosen model, padding, and resolution, with numbers recorded in `tools/classifier/RESULTS.md`.

### Phase 2: Classifier engine (Kotlin)

1. `ObjectClassifier`, `OnnxImageClassifier`, `LabelMap`, and asset packaging.
2. Lazy initialization in the ViewModel on `Dispatchers.IO`, separate from SpatialNet. Failure sets `classifierError` and does not affect `isModelReady`.
3. An instrumented test that classifies a fixture crop of the TV in `ROI_issues/` and expects "TV / screen" in the top 3.

Exit: `classify()` returns sensible labels on a device, and latency is logged.

### Phase 3: Pipeline integration

1. `RoiCropper` with unit tests for the normalized-to-pixel mapping, padding, squaring, clamping at frame edges, and minimum-size rejection.
2. `ClassificationScheduler` and `LabelSmoother` with unit tests: a stable sequence produces a stable label, a class switch requires a sustained change, the smoother resets on ROI loss or jump, and the threshold yields "Unknown".
3. Wire them into `InferenceViewModel` using the dedicated dispatcher and busy flag, and expose `objectResult`.
4. If Phase 1 showed that higher resolution helps, set the `ImageAnalysis` `ResolutionSelector` to about 1280x720 and confirm the added `toBitmap`/rotate cost fits the FPS budget.

Exit: labels flow to the UI state without reducing SpatialNet FPS by more than 10%.

### Phase 4: UI and capture

1. Label chip, "Identifying…" and "Unknown object" states, classifier timing in the stats bar, drawn through `OverlayRenderer` so that live view and Capture match.
2. Identify toggle, dual-model badge, and an optional top-3 detail view.
3. Error state when the classifier is unavailable.

Exit: live view and saved JPEG show the same label for the same frame.

### Phase 5: Hardening, performance, release

1. Profile on at least 2 phones (mid-range and low-end): FPS, classifier latency, memory, thermal behavior over 5 minutes.
2. Optionally try the XNNPACK or NNAPI execution provider for the classifier, and INT8 quantization if size or latency need it.
3. Update the README (new pipeline stage, model table, digests), the visual walkthrough (a crop and classification step), and the release notes. Build both APKs and publish v1.1.

Exit: all acceptance criteria in §2 are met and recorded.

### Phase 6: Later upgrades (optional)

- **Open vocabulary:** replace the classifier with a MobileCLIP image encoder and a user-editable label list with precomputed text embeddings.
- **"Teach an object":** store embeddings of the user's own objects on the phone and match against them (few-shot, no retraining).
- **Fine-tuning:** fine-tune MobileNetV3 on the collected ROI crops for this app's actual scenes.

## 6. Extension points for the next features

Phase 0 and Phase 3 establish a pattern that later features can reuse:

- **Per-ROI stages.** The cropper, scheduler, background dispatcher, and smoother form a reusable "secondary model on the ROI" stage. OCR on the pointed-at text, color naming, and similar features can plug into the same stage.
- **One overlay renderer.** New overlay elements are drawn once and appear on both the live view and Capture.
- **Model registry.** `ModelInfo` and the badge scale to any number of bundled models.
- **UI state.** `CombinedResult` gains nullable fields per feature, and each feature has its own `StateFlow` for status and errors.

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Tiny ROI crops at 640x480 | Low accuracy | Padding, higher analysis resolution (measured in Phase 1), minimum-size rejection with an "Move closer" hint |
| ImageNet vocabulary misses the user's objects | "Unknown object" or wrong labels | `LabelMap` merging, top-3 display, open-vocabulary upgrade path (Phase 6) |
| ROI box is loose or includes the fingertip or hand | Classifier names "hand" or background | Evaluate padding in Phase 1; optionally shrink the crop slightly on the side facing the fingertip |
| CPU contention between two ORT sessions | Lower FPS, heat | Thread split (4 + 2), classification rate limit, toggle, optional XNNPACK/NNAPI |
| Label flicker | Poor UX | EMA smoother, IoU-based tracking, display threshold |
| Stale label on a moved box | Wrong object named | Box-IoU and age check before drawing |
| APK growth | Larger download | FP16-stored weights (~11 MB), INT8 fallback (~5.5 MB) |

## 8. Decisions to confirm before Phase 1

1. **Engine:** ONNX MobileNetV3 (recommended), ML Kit, or MobileCLIP from the start.
2. **Vocabulary:** is ImageNet's 1000-class set acceptable to start with, or is there a fixed list of target objects (e.g. home appliances only)? A fixed list favors MobileCLIP or a fine-tuned model.
3. **Classifier per flavor:** the same classifier in both APKs (recommended), or an FP32 classifier in the FP32 APK and an FP16 classifier in the FP16 APK.
4. **Analysis resolution:** is a possible FPS cost acceptable in exchange for sharper crops? This is decided with Phase 1 data.
