# Shrimp Disease Detection — Flutter App (Android)

A mobile app that runs the YOLO11n model **entirely on the device** — no server,
no network connection. Pick a photo and the app boxes each shrimp and counts how
many are healthy versus diseased.

It shares its weights with the FastAPI web app at the repository root
(`models/best.pt`), converted to TFLite so it can run on a phone.

| | |
|---|---|
| Model | YOLO11n, 2.58M parameters, 4.1 GFLOPs |
| Input size | 512×512 (matches training) |
| Accuracy | mAP50-95 = 0.807 |
| Classes | `0` = Healthy (blue ✓), `1` = Diseased (red !) |
| Model file on device | ~10.5 MB |
| Release APK | ~78 MB (bundles every CPU architecture) |

## Quick start

If Flutter and the Android SDK are already installed:

```bash
flutter emulators --launch Pixel_9a    # or plug in a phone over USB
cd mobile_app
flutter pub get
flutter run
```

`flutter run` builds, installs, and launches in one step. First run takes 1–2
minutes; later runs are faster.

Starting from nothing? See section 1. Note that an emulator's photo gallery is
empty by default — push some images first, see section 3.

---

## 1. Environment setup

### What you need

| Component | Verified version | Required? |
|---|---|---|
| Flutter (includes Dart) | 3.44.8 / Dart 3.12.2 | Yes |
| Android SDK | platform 36, build-tools 36.1.0 | Yes |
| JDK | 21 (bundled with Android Studio) | Yes |
| Emulator or physical Android device | Android 13+ | Yes |
| Python | 3.11 | Only to re-export the model |
| Xcode | — | Only to build for iPhone |

### Steps on macOS

```bash
# 1. Flutter
brew install --cask flutter

# 2. Android Studio (brings the SDK, JDK 21 and the emulator manager)
brew install --cask android-studio
# open Android Studio once so it downloads the SDK, then close it

# 3. Accept all Android licences
flutter doctor --android-licenses     # press y for everything

# 4. Create an emulator if you do not have one
flutter emulators --create --name Pixel_9a

# 5. Verify
flutter doctor
```

### What a working install looks like

```
[✓] Flutter (Channel stable, 3.44.8, on macOS, locale vi-VN)
[✓] Android toolchain - develop for Android devices (Android SDK version 36.1.0)
[!] Xcode - develop for iOS and macOS          ← ignorable for Android-only work
[✓] Chrome - develop for the web
[✓] Connected device (3 available)
[✓] Network resources
```

**Only `[✓] Flutter` and `[✓] Android toolchain` matter** for this app. The
`[!] Xcode` line has no effect on Android — see section 7 if you want to build
for iPhone.

### Fetch the project's packages

```bash
cd mobile_app
flutter pub get
```

| Package | Purpose |
|---|---|
| `tflite_flutter` | runs the TFLite model on device |
| `image` | decodes, resizes, and fixes EXIF orientation |
| `image_picker` | picks a photo from the gallery or takes a new one |
| `integration_test` | runs tests on a real device |

<details>
<summary>If the Flutter download breaks partway through</summary>

```
Error: Download failed on Cask 'flutter'
curl: (56) Recv failure: Connection reset by peer
```

Re-run the same `brew install --cask flutter` command — Homebrew resumes from
where it stopped instead of restarting the download.
</details>

---

## 2. Prepare the model file

The app needs `assets/models/shrimp_yolo11n_512.tflite`. **If that file is
already there, skip this whole section** and go to section 3.

You only need this after retraining and producing a new `models/best.pt`:

```bash
# run from the repository root, not from mobile_app/
cd ..

# first time only: build a separate Python environment for exporting
python3.11 -m venv .venv-export
.venv-export/bin/pip install -r scripts/requirements-export.txt

# export
.venv-export/bin/python scripts/export_tflite.py
```

The script copies the result into `mobile_app/assets/models/` and prints a
verification block at the end. **You must see these values** (the script's own
output is in Vietnamese):

```
[export] input : shape=[1, 512, 512, 3] dtype=float32
[export] output: shape=[1, 6, 5376] dtype=float32
[export] layout: [1, kênh, anchor]              ← [1, channel, anchor]
[export] số anchor: 5376 (mong đợi 5376)        ← anchor count, expected 5376
[export] biên độ toạ độ lớn nhất: 531.475 -> PIXEL, dùng thẳng
                                                ← max coordinate → PIXEL space
```

If the layout line or the coordinate-range line differs, the Dart code will
decode the tensor wrongly and boxes will be drawn in the wrong place **with no
error message at all**. See section 7.

To confirm the TFLite build still matches the original PyTorch model:

```bash
.venv-export/bin/python scripts/verify_tflite.py
```

---

## 3. Running the app

### On an emulator

```bash
# list available emulators
flutter emulators

# start one
flutter emulators --launch Pixel_9a

# wait until it reaches the home screen, then
cd mobile_app
flutter run
```

**An emulator's photo gallery starts empty**, so push some images before you try
the app:

```bash
# push a dozen images from the test split
for f in $(ls ../data/tom_benh.v1i.yolov11/test/images/*.jpg | head -12); do
    adb push "$f" "/sdcard/Pictures/$(basename $f)"
done

# required — without a rescan the picker will not see the new files
for f in $(adb shell ls /sdcard/Pictures/ | tr -d '\r'); do
    adb shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE \
        -d "file:///sdcard/Pictures/$f"
done
```

The **Chụp ảnh** (take photo) button does work on an emulator, but the virtual
back camera renders a 3D room (`hw.camera.back=virtualscene`), so the result is
always zero shrimp. To test the camera path with real content, point it at your
Mac's webcam:

```bash
sed -i '' 's/hw.camera.back=virtualscene/hw.camera.back=webcam0/' \
    ~/.android/avd/Pixel_9a.avd/config.ini
```

then restart the emulator and hold a shrimp photo in front of the webcam.

### On a physical Android phone

Much faster than an emulator, and the camera is real:

1. On the phone: **Settings → About phone → tap "Build number" seven times** to
   unlock developer mode
2. **Settings → Developer options → enable "USB debugging"**
3. Connect the USB cable and tap **Allow** on the phone when prompted
4. Confirm it is detected: `flutter devices`
5. Run: `flutter run`

### Keys while the app is running

| Key | Action |
|---|---|
| `r` | Hot reload, keeping current state |
| `R` | Restart the app from scratch |
| `q` | Quit |

### Standalone install (APK)

```bash
flutter build apk --release
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

The file lands at `build/app/outputs/flutter-apk/app-release.apk`; you can also
copy it to a phone and install it directly. A release build runs AOT-compiled,
so it starts much faster than a debug build.

---

## 4. Using the app

1. Tap **Thư viện** (Gallery) to pick an existing photo, or **Chụp ảnh** (Take
   photo) for a new one
2. Detection runs automatically as soon as an image is chosen
3. The bottom panel shows the **healthy** count, the **diseased** count, the
   **disease rate**, and the **processing time**
4. Drag the **Ngưỡng tin cậy** (confidence threshold) slider to filter more or
   less strictly:
   - **Lower** it (say 0.10) to catch more shrimp, at the cost of false positives
   - **Raise** it (say 0.50) to keep only confident detections, at the cost of
     missing some

Changing the threshold re-runs detection on the image already loaded — you do
not need to pick it again.

---

## 5. Running the tests

There are two tiers, and **the second one is what proves the app is correct**.

### Unit tests — no device needed

```bash
flutter test
```

Covers the maths most likely to be wrong, in a couple of seconds:

- `test/letterbox_test.dart` — scale factor, padding, and the inverse mapping of
  coordinates back to the original image
- `test/nms_test.dart` — IoU, tensor decoding, duplicate-box suppression

### Integration tests — on a real device

```bash
flutter test integration_test/detector_test.dart -d emulator-5554
```

This loads the real `.tflite` file, pushes `assets/test/sample.jpg` through the
app's actual pipeline, and compares against the original PyTorch numbers:

```
class 0 (Healthy)  conf 0.9483  box (174.3, 219.3, 512.0, 392.9)
class 1 (Diseased) conf 0.9229  box (145.7, 231.2, 269.3, 350.5)
class 1 (Diseased) conf 0.9047  box (  0.2, 245.2,  63.6, 327.3)
```

If the unit tests pass but the integration tests fail, the fault is in model
loading or in how `yolo_detector.dart` wires the stages together — not in the
maths.

### Per-stage timing

```bash
flutter test integration_test/timing_test.dart -d emulator-5554
```

Prints where the time actually goes, so optimisation targets are measured rather
than guessed. Measured on a Pixel_9a emulator, averaged over 10 runs after
discarding 3 warm-up runs:

```
  letterbox           31.7 ms   11.7%
  build tensor        27.1 ms   10.1%
  run model          210.9 ms   78.1%
  decode + NMS         0.2 ms    0.1%
```

The conclusion that matters for real-time camera work: **most of the time is the
model itself, not the Dart-side data preparation.** Optimising the Dart code can
buy back at most ~60 ms. Getting substantially faster means running on real
hardware (an emulator has no NNAPI or GPU delegate), enabling a hardware
delegate, or exporting an int8-quantised model.

---

## 6. Source layout

```
mobile_app/
├── assets/models/
│   └── shrimp_yolo11n_512.tflite    # produced by scripts/export_tflite.py
├── lib/
│   ├── main.dart                    # entry point, theme
│   ├── core/constants.dart          # class names, colours, default thresholds
│   ├── models/detection.dart        # Detection, DetectionResult, timings
│   ├── services/
│   │   ├── letterbox.dart           # aspect-preserving resize + inverse mapping
│   │   ├── nms.dart                 # tensor decode + duplicate suppression
│   │   └── yolo_detector.dart       # loads TFLite, wires the whole pipeline
│   ├── screens/detect_screen.dart   # main screen
│   └── widgets/detection_painter.dart  # draws boxes over the image
├── test/                            # letterbox and NMS unit tests
└── integration_test/                # on-device correctness and timing tests
```

How one image flows through:

```
original image
  → bakeOrientation      apply the EXIF rotation flag (portrait shots otherwise
                         arrive sideways)
  → letterbox            aspect-preserving resize, grey-padded to 512×512
  → normalise /255       into a [1, 512, 512, 3] tensor
  → TFLite               out comes a [1, 6, 5376] tensor
  → decodeYolo           take the top-scoring class, threshold, xywh → xyxy
  → nonMaxSuppression    drop overlapping boxes, per class
  → toOriginal           map coordinates back onto the original image
```

---

## 7. Troubleshooting

### "Không nạp được model" (model failed to load)

The `.tflite` file is missing from `assets/models/`. Follow section 2. Also check
that `pubspec.yaml` declares it:

```yaml
flutter:
  assets:
    - assets/models/
```

### Boxes are drawn in the wrong place, or scaled wrongly

Almost always letterboxing rather than the model. Run `flutter test` first — if
the letterbox tests pass and boxes are still off, check the tensor layout with
`scripts/export_tflite.py`, which prints the layout and coordinate range at the
end.

`decodeYolo` auto-detects both tensor layouts (`[1,6,N]` and `[1,N,6]`) and both
coordinate conventions (pixel and normalised `[0,1]`), so a different export
still works — but confirm it detected the right one.

### No shrimp detected at all

Drag the threshold down to 0.10. If there is still nothing, the photo is probably
far from the training distribution (707 images, close-ups of individual shrimp).

### Android build fails with `Inconsistent JVM-target compatibility`

```
Execution failed for task ':tflite_flutter:compileDebugKotlin'.
> Inconsistent JVM-target compatibility detected for tasks
  'compileDebugJavaWithJavac' (11) and 'compileDebugKotlin' (21)
```

The `tflite_flutter` plugin declares Java 11, while Kotlin follows the JDK that
runs Gradle (JDK 21, bundled with Android Studio). `android/build.gradle.kts`
already contains a `subprojects` block forcing both to 17 — delete that block and
this error comes back.

That block must sit **before** `subprojects { evaluationDependsOn(":app") }` and
must exclude `:app`, because `evaluationDependsOn` forces the app module to
finish evaluating, after which you can no longer attach an `afterEvaluate` to it
(`Cannot run Project.afterEvaluate(Action) when the project is already
evaluated`).

### App stuck on the Flutter splash screen

If you install a debug APK by hand and open it, startup is very slow because
debug builds run under the JIT. Use `flutter run` instead of installing the APK,
or build a release:

```bash
flutter build apk --release
```

### `flutter: command not found`

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

Add that line to `~/.zshrc` so you do not have to retype it.

### Building for iPhone

Nothing in `lib/` is Android-specific and the `ios/` directory already exists,
but this machine only has the Command Line Tools, not a full Xcode, so iOS
builds are not possible yet. What is needed:

```bash
# 1. Install Xcode from the App Store (~20 GB — make room first)
# 2. Point the toolchain at Xcode instead of the Command Line Tools
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch

# 3. CocoaPods is already installed; just install the project's pods
cd ios && pod install && cd ..

# 4. Verify, then run
flutter doctor          # the Xcode line must turn [✓]
flutter run -d <iphone>
```

Running on a physical iPhone also needs an Apple ID for code signing — open
`ios/Runner.xcworkspace` in Xcode, go to **Signing & Capabilities**, and pick a
Team. Builds signed with a free account expire after 7 days and must be
reinstalled.

### Export fails with `ImportError: cannot import name 'get_cuda_generator_meta_val'`

This is exactly why the project ships `scripts/export_tflite.py` instead of
calling `yolo export format=tflite` directly. Since ultralytics 8.4.83,
`format='tflite'` redirects to `litert-torch`, which is not compatible with
torch 2.13. The script takes the older `.pt → ONNX → onnx2tf → TFLite` route and
never touches that package.

### Export fails with `_pickle.UnpicklingError: could not find MARK`

onnx2tf downloads a sample-image file from GitHub with a 5-second read timeout; a
slow connection returns corrupted content. The script now generates that file
locally instead of downloading it. If you still hit this, delete
`calibration_image_sample_data_20x128x128x3_float32.npy` from the repository root
and run again.

---

## 8. Technical notes

**Why the class names are hardcoded.** `models/best.pt` stores its two class
names as `'0'` and `'1'` — Roboflow defaults that carry no meaning. The real
mapping lives in `training_notebooks/shrimp.ipynb`, cell 11:

```python
model.model.names = {0: "Healthy", 1: "Diseased"}
```

The notebook assigns that at inference time, so it never gets written into the
model file. The app therefore defines it in `lib/core/constants.dart`. **If you
retrain and the class order changes, that file must be updated** — otherwise the
app will label diseased shrimp as healthy with no sign anything is wrong.

**Why 512 and not 640.** The model was trained at `imgsz=512`. Running inference
at a different size than training costs accuracy.

**Why 5376 anchors.** A 512-pixel image through three feature levels at strides
8/16/32: 64² + 32² + 16² = 4096 + 1024 + 256 = 5376. Each anchor carries 6
numbers — 4 box coordinates plus 2 class scores.

**The web app's distance filter was deliberately left out.** `app/main.py`
contains `distance_from_bbox_h()`, which estimates distance from box height,
calibrated against a constant `REF_SIGN_HEIGHT_M = 0.60` metres. That is a
leftover from a traffic-sign detection problem and is meaningless for shrimp.

**Default thresholds** are conf 0.25 and IoU 0.60 — the IoU value matches
`INFER_IOU` in `app/main.py` so both front ends agree.

**Why healthy shrimp are blue rather than green.** Green/red is the classic
colour-blindness trap. Measured as OKLab ΔE with Machado 2009 simulation
matrices:

| Colour pair | ΔE under deuteranopia | Safe threshold |
|---|---|---|
| Green `#22C55E` / red `#EF4444` | 7.4 | 8 |
| Green `#0CA30C` / red `#D03B3B` | 4.1 | 8 |
| **Blue `#2A78D6` / red `#D03B3B`** | **23.8** | 8 |

Roughly 8% of men have deuteranopia. This app exists to tell diseased shrimp from
healthy ones, so a confusion here has real consequences — the colours were picked
from measurements rather than habit. Two further channels mean colour is never
the only cue: the `✓` / `!` **glyphs** on box labels, and the check/alert
**icons** in the stat row.

The numbers in the stat row are deliberately left in ordinary text colour rather
than tinted per class — the coloured mark beside them carries the identity.
Colouring the digits themselves would cost a colour-blind reader both the number
and its label.
