# ROI Detection v1.0

This release provides two Android debug APKs for on-device SpatialNet inference. Both use the same full-frame camera view and aligned detection overlay. They can be installed side by side.

| Asset | Model | Android app ID |
| --- | --- | --- |
| `app-fp32-debug.apk` | `spatialnet_v4.onnx`, FP32 weights | `com.example.roidetection` |
| `app-fromFp16-debug.apk` | `spatialnet_fastest_from_fp16.onnx`, FLOAT16 stored weights with float32 input/output | `com.example.roidetection.fromfp16` |

**Install:** Download the APK for the model you want, open it on an Android 7.0+ phone with a back camera, and grant camera permission. The model badge shows the bundled asset name and digest. Expected digest prefixes: FP32 `76710641`; FP16-source `ed53f221`.

The app shows the complete analyzed frame without center cropping. Boxes, fingertip, and pointing arrow are drawn over that same frame. Capture saves an annotated JPEG at the analysis frame's pixel dimensions. See the repository README and visual walkthrough for details.

These are debug builds. The APKs should be attached to this release as assets; the automatically generated source archives are not Android installers.
