# ROI Detection

This Android app watches the back camera for a hand, determines whether it is pointing, and marks a target region when one is found. Two APK flavors use the same camera and drawing code, each bundling one model from `ROIModel/`. The same camera session drives the live view and the Capture button. The sections below follow one frame from the camera through the model to those two outputs.

**Want to see every image stage?** Open [README_FIXED.md](image_walkthrough/README_FIXED.md) and its new visual flow. The [earlier walkthrough](image_walkthrough/README.md) remains as a record of the cropped-preview problem.

## Get the Android app

The app requires **Android 7.0 (API 24) or newer** and a back camera. Choose one APK:

| Release asset | Model bundled in that APK | App label | Model badge |
| --- | --- | --- | --- |
| `app-fp32-debug.apk` | `spatialnet_v4.onnx` (float32 weights) | ROI Detection FP32 | `spatialnet_v4` / `76710641` |
| `app-fromFp16-debug.apk` | `spatialnet_fastest_from_fp16.onnx` (FLOAT16 stored weights; float32 input/output) | ROI Detection FP16 Source | `spatialnet_fastest_from_fp16` / `ed53f221` |

The two apps have different Android application IDs, so they can be installed together for comparison. Both use the same camera, preprocessing, preview, overlay, and Capture code. Only the bundled model differs. The badge in the app is calculated from the installed model bytes; its short digest helps distinguish an old installation from the intended build.

### Download from a GitHub Release

Once this project has a published release with both APKs attached:

1. Open the **new repository's URL**, then open **Releases**. The latest release is also at `<repository URL>/releases/latest`.
2. Under that release's **Assets**, select `app-fp32-debug.apk` or `app-fromFp16-debug.apk`. Download the **APK asset**, not the automatically generated "Source code" ZIP or tarball. GitHub's [release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/linking-to-releases) describes the latest-release link format.
3. On your Android phone, open the downloaded APK and follow the install prompts. If Android asks you to allow installs from the browser or file manager you used, enable that source in the phone's settings and retry the APK. Available prompts can differ by Android version and device policy.
4. Open the app, grant **Camera** permission, and tap **Start Inference**. Check the model badge against the table above. The live image shows the full analyzed frame; black margins can appear so it is not cropped. Tap **Capture** to save an annotated JPEG to the gallery.

The repository URL and release do not exist in this README yet because the new GitHub destination has not been provided or published. APKs are attached to the release separately; they are not stored in Git history.

### Build and install from source

Install Android Studio with **Android SDK 35** and a **JDK 17 or newer**. Open the repository root as an Android project, let Gradle sync, and connect an Android device with USB debugging if you want to run from Android Studio. The Gradle wrapper downloads the configured Gradle version when needed.

From the repository root, build both debug variants:

```powershell
# Windows PowerShell
.\gradlew.bat assembleFp32Debug assembleFromFp16Debug
```

```sh
# macOS or Linux
./gradlew assembleFp32Debug assembleFromFp16Debug
```

The outputs are:

```text
app/build/outputs/apk/fp32/debug/app-fp32-debug.apk
app/build/outputs/apk/fromFp16/debug/app-fromFp16-debug.apk
```

To install from a computer with Android Debug Bridge, use `adb install -r <path-to-apk>` for each APK. In Android Studio, select the `fp32Debug` or `fromFp16Debug` build variant before pressing Run. These are **debug builds**; distributing a signed production release would require a release signing configuration.

### Visual walkthrough

The [fixed visual walkthrough](image_walkthrough/README_FIXED.md) follows a real sample photo through rotation, 160 x 160 model input, normalized tensor, model output, fitted live view, and saved capture. Its generated images are in `image_walkthrough/fixed_output/` and `image_walkthrough/output/`. To regenerate them, install [the Python requirements](image_walkthrough/requirements.txt) and run `python image_walkthrough/make_walkthrough.py` from the repository root. Python is only needed for this explanation; it is not needed to build or use the Android app.

## Repository map

| Path | Purpose |
| --- | --- |
| `ROIModel/` | The original `spatialnet_v4.onnx` and the alternative `spatialnet_fastest_from_fp16.onnx`. Each APK packages only its selected model. |
| `app/src/main/java/com/example/roidetection/` | Camera-frame processing, ONNX inference, result types, and app entry point. |
| `app/src/main/java/com/example/roidetection/ui/` | Home screen, live preview, overlays, and Capture. |
| `app/src/main/res/` | Android theme, strings, and launcher icons. |
| `app/src/main/AndroidManifest.xml` | Camera permission and portrait activity. |
| `app/build.gradle.kts` | Android build settings, dependencies, and the model asset source. |
| `ROI_issues/` | Supplied sample photo and actual phone preview/capture images used for the walkthrough. |
| `image_walkthrough/` | Python script and generated pictures showing each image transformation. |
| `gradle/`, `gradlew`, `gradlew.bat`, root Gradle files | Gradle wrapper and project configuration used to build the APK. |

The camera and screen sizes below vary by phone because the app does not request a fixed camera resolution.

## Sizes at each stage

| Stage | Pixel or tensor dimensions |
| --- | --- |
| On-screen camera image | The rotated `ImageAnalysis` frame, shown in full |
| Analysis frame | `imageProxy.width × imageProxy.height`; selected by CameraX |
| Rotated analysis bitmap | Same size, or width and height swapped for 90°/270° rotation |
| Image passed to model | **160 × 160 RGB** |
| Model tensor | **`[1, 3, 160, 160]` float32**, in batch/channel/height/width order |
| On-screen image and overlay | Same analysis frame and result pair, fitted without cropping into the available `Scaffold` area |
| Saved JPEG | **Rotated analysis bitmap width × height** |

For example, a 1280 × 720 analysis frame rotated 90° yields a 720 × 1280 bitmap and saved JPEG, while the model still receives 160 × 160. **1280 × 720 is an example, not a configured resolution.** The first actual analysis frame dimensions and rotation are logged under `InferenceViewModel`.

## From the camera to the analysis frame

The activity runs in portrait and uses the default **back camera**. It binds one CameraX use case:

- `ImageAnalysis` supplies frames for both inference and the visible image. `STRATEGY_KEEP_ONLY_LATEST` drops older queued frames if analysis falls behind. The ViewModel also skips a frame while another is processing.

No target resolution or aspect ratio is set. Each accepted frame is rotated and then displayed by Compose with `ContentScale.Fit`. The whole frame remains visible; black margins appear when its aspect ratio differs from the available screen area. Display updates at the processed inference rate, which can be slower than the camera's capture rate. There is no separate, differently cropped `Preview` stream.

## From the analysis frame to SpatialNet

Each accepted analysis frame is converted from `ImageProxy` to a bitmap and rotated by `imageProxy.imageInfo.rotationDegrees`. This full rotated bitmap is paired with its prediction for the screen and Capture. It is also processed for SpatialNet:

1. Resize the **whole bitmap directly to 160 × 160** with filtered bitmap scaling. There is no crop, letterbox padding, or aspect-ratio preservation. A rectangular frame is stretched into a square.
2. Read red, green, and blue pixel values and divide each 8-bit channel by 255.
3. Apply `(channel - mean) / std`. RGB means are `[0.485, 0.456, 0.406]`; standard deviations are `[0.229, 0.224, 0.225]`.
4. Put the float32 values in channel-first order into one **`[1, 3, 160, 160]`** tensor.

Gradle packages the selected file from `ROIModel/` as an APK asset. The app loads its bytes into ONNX Runtime. The app does not retrain or alter either model. Both currently expose the same float32 input and output shapes, so one SpatialNet processing path handles both. One model run processes each accepted frame.

## From SpatialNet outputs to a detection

| Named output | Shape | Meaning |
| --- | --- | --- |
| `hand` | `[1, 4]` | Hand box `(center_x, center_y, width, height)` |
| `fingertip` | `[1, 2]` | Fingertip position `(x, y)` |
| `angle` | `[1, 2]` | Pointing direction `(sin θ, cos θ)` |
| `roi` | `[1, 4]` | Target box `(center_x, center_y, width, height)` |
| `hand_conf` | `[1, 1]` | Raw hand-presence logit |
| `point_conf` | `[1, 1]` | Raw pointing logit |
| `roi_conf` | `[1, 1]` | Raw target-presence logit |

Positions and box sizes are normalized relative to the 160 × 160 model image, roughly in the 0–1 range. The app applies **sigmoid to all three logits** to obtain probabilities. It then checks each gate at **0.5**, in order:

| Result | Condition | Geometry retained |
| --- | --- | --- |
| `NO HAND` | Hand probability < 0.5 | None |
| `NON-POINTING` | Hand ≥ 0.5, pointing < 0.5 | Hand box |
| `POINTING-EMPTY` | Hand and pointing ≥ 0.5, ROI < 0.5 | Hand box, fingertip, direction |
| `POINTING` | All three ≥ 0.5 | Hand box, fingertip, direction, ROI box |

For overlay mode, the ViewModel keeps the most recent **five** pointing results. It enters pointing mode when any result in that window is pointing with probability at least **0.30**, and stays there while a pointing result remains in the window. The category text and confidence use the **current frame's** result, so they can briefly disagree with the smoothed overlay mode. A `0.3` default argument exists in `SpatialNetModel.predict`, but the classification itself uses the 0.5 thresholds above.

## From the detection to the live preview

The model does **not** return an image to resize back. Compose displays the complete rotated analysis frame with `ContentScale.Fit`, centered in a black area. A transparent canvas covers that same area. Let `W × H` be the canvas size and `F_w × F_h` the rotated frame size:

```text
scale = min(W / F_w, H / F_h)
image_width = F_w × scale
image_height = F_h × scale
offset_x = (W - image_width) / 2
offset_y = (H - image_height) / 2
point_x = offset_x + normalized_x × image_width
point_y = offset_y + normalized_y × image_height
left = offset_x + (center_x - width/2) × image_width
top = offset_y + (center_y - height/2) × image_height
right = offset_x + (center_x + width/2) × image_width
bottom = offset_y + (center_y + height/2) × image_height
```

Box corners are clamped to 0–1 before drawing. Pointing mode shows a blue hand box, green ROI box if present, red fingertip, and yellow direction arrow. Non-pointing mode shows a red hand box if present. The arrow begins at the fingertip, points along `atan2(sin θ, cos θ)`, and is **80 canvas pixels** long. A bottom panel shows processed-frame FPS, inference time, category, and confidence; FPS is not necessarily the preview frame rate.

The image and overlay now use the **same frame and the same fit geometry**, so the old center-crop displacement is removed. The model still sees a square-stretched 160 × 160 version of the full frame. The on-screen overlay is drawn independently from the picture, so the bottom status panel and model badge can cover parts of it.

## From the detection to a saved capture

Capture uses the **same rotated ImageAnalysis bitmap and prediction pair** shown on screen. It is not a screen screenshot, a CameraX still photo, or the 160 × 160 model input. It makes a mutable copy at the **same pixel dimensions**, draws the result onto it, and saves it to the gallery.

The saved overlay uses the same normalized-coordinate formulas, replacing `W` and `H` with the bitmap width and height. It draws the boxes, fingertip, and a **90 bitmap-pixel** direction arrow. A 50-pixel-tall dark bar at the bottom contains inference time, category, and confidence. The live screen's top bar, buttons, model badge, and FPS are not included. Because the saved and live arrows use different fixed pixel lengths, their relative appearance can differ.

The bitmap is encoded as **JPEG at quality 95** with a name like `ROI_Screenshot_<timestamp>.jpg`. On Android 10 and newer, the requested gallery folder is `Pictures/ROIDetection`. Saved width and height are exactly those of the rotated analysis bitmap; there is no additional resize or crop. The screen's black fit margins are not saved.

## Model variants

The FP32 app uses application ID `com.example.roidetection`; the alternative uses `com.example.roidetection.fromfp16`. The supplied `spatialnet_fastest_from_fp16.onnx` has **FLOAT16 stored weights** and **float32 input and outputs**. The Android preprocessing and output parsing therefore remain float32, while the FP16-source APK bundles the smaller model file. Actual execution speed and numerical results can vary by phone and ONNX Runtime execution provider.
