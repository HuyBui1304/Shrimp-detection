# Nhận diện tôm bệnh — App Flutter (Android)

App di động chạy model YOLO11n **hoàn toàn trên máy**, không cần server và
không cần mạng. Chọn một tấm ảnh, app khoanh vùng từng con tôm và đếm số con
khoẻ / con bệnh.

Model dùng chung với web app FastAPI ở thư mục gốc (`models/best.pt`), nhưng
được chuyển sang định dạng TFLite để chạy được trên điện thoại.

| | |
|---|---|
| Model | YOLO11n, 2.58 triệu tham số, 4.1 GFLOPs |
| Kích thước ảnh vào | 512×512 (đúng bằng lúc train) |
| Chất lượng | mAP50-95 = 0.807 |
| Lớp | `0` = Tôm khỏe (xanh lá), `1` = Tôm bệnh (đỏ) |
| File model trên máy | ~10.5 MB |

---

## 1. Yêu cầu

- **Flutter** 3.44+ (kèm Dart 3.12+)
- **Android SDK** + một emulator hoặc điện thoại Android thật
- **Python 3.11** — chỉ cần khi bạn muốn xuất lại model từ `best.pt`

Kiểm tra nhanh:

```bash
flutter doctor
```

Chỉ cần hai dòng `[✓] Flutter` và `[✓] Android toolchain` là chạy được.
Dòng `[!] Xcode` báo đỏ thì kệ nó — đó là phần dành cho iOS.

<details>
<summary>Nếu máy chưa có Flutter</summary>

```bash
brew install --cask flutter
flutter doctor --android-licenses   # bấm y cho tất cả
```

Nếu tải giữa chừng đứt (`curl: (56) Recv failure`), chạy lại đúng lệnh đó —
Homebrew tải tiếp chỗ dở chứ không tải lại từ đầu.
</details>

---

## 2. Chuẩn bị file model

App cần file `assets/models/shrimp_yolo11n_512.tflite`. **Nếu file này đã có
sẵn thì bỏ qua cả mục này**, nhảy thẳng xuống mục 3.

Chỉ phải làm lại khi bạn train lại model và có `models/best.pt` mới:

```bash
# chạy từ thư mục gốc dự án, không phải từ mobile_app/
cd ..

# lần đầu: dựng môi trường Python riêng cho việc xuất model
python3.11 -m venv .venv-export
.venv-export/bin/pip install -r scripts/requirements-export.txt

# xuất model
.venv-export/bin/python scripts/export_tflite.py
```

Script tự chép file kết quả vào `mobile_app/assets/models/`. Khi xong nó in ra
phần kiểm chứng, và **bạn phải thấy đúng những dòng này**:

```
[export] input : shape=[1, 512, 512, 3] dtype=float32
[export] output: shape=[1, 6, 5376] dtype=float32
[export] layout: [1, kênh, anchor]
[export] số anchor: 5376 (mong đợi 5376)
[export] biên độ toạ độ lớn nhất: 531.475 -> PIXEL, dùng thẳng
```

Nếu `layout` hoặc dòng `biên độ toạ độ` khác đi, code Dart sẽ giải mã sai và
box vẽ ra sẽ lệch **mà không có thông báo lỗi nào**. Xem mục 7 để biết cách xử lý.

Muốn chắc chắn bản TFLite không lệch so với bản PyTorch gốc:

```bash
.venv-export/bin/python scripts/verify_tflite.py
```

---

## 3. Chạy app

### Trên emulator

```bash
# xem danh sách emulator có sẵn
flutter emulators

# bật lên (máy này đang có Pixel_9a)
flutter emulators --launch Pixel_9a

# đợi emulator vào tới màn hình chính rồi chạy
cd mobile_app
flutter run
```

### Trên điện thoại Android thật

Cách này chạy nhanh hơn emulator nhiều và dùng được camera thật:

1. Trên điện thoại: **Cài đặt → Giới thiệu → bấm 7 lần vào "Số hiệu bản dựng"**
   để mở khoá chế độ nhà phát triển
2. **Cài đặt → Tuỳ chọn nhà phát triển → bật "Gỡ lỗi qua USB"**
3. Cắm cáp USB, trên điện thoại bấm **Cho phép** khi hiện hộp thoại
4. Kiểm tra máy đã nhận: `flutter devices`
5. Chạy: `flutter run`

### Các phím tắt khi đang chạy

| Phím | Tác dụng |
|---|---|
| `r` | Nạp lại nóng, giữ nguyên trạng thái đang có |
| `R` | Khởi động lại app từ đầu |
| `q` | Thoát |

### Bản cài đặt thật (APK)

```bash
flutter build apk --release
```

File nằm ở `build/app/outputs/flutter-apk/app-release.apk`, copy sang điện
thoại rồi cài trực tiếp.

---

## 4. Cách dùng

1. Bấm **Thư viện** để chọn ảnh có sẵn, hoặc **Chụp** để chụp mới
2. App tự nhận diện ngay sau khi chọn ảnh
3. Bảng dưới cùng hiển thị: số **Tôm khỏe**, số **Tôm bệnh**, **tỉ lệ bệnh**
   và **thời gian xử lý**
4. Kéo thanh **Ngưỡng** để lọc chặt hay lỏng hơn:
   - Kéo **xuống** (ví dụ 0.10): bắt được nhiều con hơn, nhưng dễ báo nhầm
   - Kéo **lên** (ví dụ 0.50): chỉ giữ con chắc chắn, nhưng dễ bỏ sót

Đổi ngưỡng thì app chạy lại nhận diện trên ảnh cũ, không cần chọn lại ảnh.

---

## 5. Chạy test

Có hai tầng test, và **tầng thứ hai mới là tầng chứng minh app chạy đúng**.

### Test đơn vị — không cần thiết bị

```bash
flutter test
```

Kiểm phần toán học dễ sai nhất, chạy trong vài giây:

- `test/letterbox_test.dart` — tỉ lệ thu nhỏ, lượng pad, phép ánh xạ ngược toạ
  độ về ảnh gốc
- `test/nms_test.dart` — tính IoU, giải mã tensor, khử box trùng

### Test tích hợp — chạy trên thiết bị thật

```bash
flutter test integration_test/detector_test.dart -d emulator-5554
```

Bài này nạp file `.tflite` thật, chạy `assets/test/sample.jpg` qua đúng chuỗi xử
lý của app, rồi đối chiếu với số liệu của bản PyTorch gốc:

```
lớp 0 (Tôm khỏe) conf 0.9483  box (174.3, 219.3, 512.0, 392.9)
lớp 1 (Tôm bệnh) conf 0.9229  box (145.7, 231.2, 269.3, 350.5)
lớp 1 (Tôm bệnh) conf 0.9047  box (  0.2, 245.2,  63.6, 327.3)
```

Test đơn vị đạt mà test tích hợp hỏng thì lỗi nằm ở khâu nạp model hoặc ở phần
ghép chuỗi trong `yolo_detector.dart`, không phải ở phần toán học.

---

## 6. Cấu trúc mã nguồn

```
mobile_app/
├── assets/models/
│   └── shrimp_yolo11n_512.tflite    # model, sinh bởi scripts/export_tflite.py
├── lib/
│   ├── main.dart                    # điểm vào, theme
│   ├── core/constants.dart          # tên lớp, màu, ngưỡng mặc định
│   ├── models/detection.dart        # Detection + DetectionResult
│   ├── services/
│   │   ├── letterbox.dart           # resize giữ tỉ lệ + ánh xạ ngược toạ độ
│   │   ├── nms.dart                 # giải mã tensor + khử box trùng
│   │   └── yolo_detector.dart       # nạp TFLite, ghép cả chuỗi xử lý
│   ├── screens/detect_screen.dart   # màn hình chính
│   └── widgets/detection_painter.dart  # vẽ box lên ảnh
└── test/                            # test cho letterbox và nms
```

Luồng xử lý một tấm ảnh:

```
ảnh gốc
  → bakeOrientation      nắn theo cờ xoay EXIF (ảnh chụp dọc hay bị nằm ngang)
  → letterbox            thu nhỏ giữ tỉ lệ, pad xám cho vuông 512×512
  → chuẩn hoá /255       thành tensor [1, 512, 512, 3]
  → TFLite               ra tensor [1, 6, 5376]
  → decodeYolo           lấy lớp điểm cao nhất, lọc ngưỡng, xywh → xyxy
  → nonMaxSuppression    khử box chồng nhau, theo từng lớp riêng
  → toOriginal           đưa toạ độ về đúng ảnh gốc
```

---

## 7. Xử lý sự cố

### `Không nạp được model`

File `.tflite` chưa có trong `assets/models/`. Làm theo mục 2. Kiểm tra thêm
là `pubspec.yaml` có khai báo:

```yaml
flutter:
  assets:
    - assets/models/
```

### Box vẽ lệch chỗ, hoặc bị co giãn sai

Gần như luôn là do letterbox chứ không phải do model. Chạy `flutter test` trước
— nếu test letterbox đạt mà box vẫn lệch thì kiểm tra layout tensor bằng
`scripts/export_tflite.py` (nó in ra layout và biên độ toạ độ ở cuối).

Hàm `decodeYolo` tự nhận ra cả hai kiểu layout (`[1,6,N]` và `[1,N,6]`) lẫn hai
kiểu toạ độ (pixel và chuẩn hoá `[0,1]`), nên đổi cách export vẫn chạy được —
nhưng phải xác nhận nó nhận đúng.

### Không thấy con tôm nào

Kéo thanh ngưỡng xuống 0.10. Nếu vẫn không có gì, nhiều khả năng ảnh khác xa
tập dữ liệu train (707 ảnh, chụp cận cảnh con tôm).

### Build Android báo `Inconsistent JVM-target compatibility`

```
Execution failed for task ':tflite_flutter:compileDebugKotlin'.
> Inconsistent JVM-target compatibility detected for tasks
  'compileDebugJavaWithJavac' (11) and 'compileDebugKotlin' (21)
```

Plugin `tflite_flutter` khai báo Java 11, còn Kotlin lấy mặc định theo JDK đang
chạy Gradle (JDK 21 đi kèm Android Studio). `android/build.gradle.kts` đã có sẵn
khối `subprojects` ép cả hai về 17 để xử lý việc này — nếu bạn lỡ xoá khối đó
thì lỗi này quay lại.

Lưu ý khối đó phải đặt **trước** `subprojects { evaluationDependsOn(":app") }`
và phải loại trừ `:app`, vì `evaluationDependsOn` ép module app đánh giá xong
sớm nên không gắn thêm `afterEvaluate` vào nó được nữa
(`Cannot run Project.afterEvaluate(Action) when the project is already evaluated`).

### App kẹt ở màn hình splash có logo Flutter

Nếu bạn cài thẳng file APK debug rồi mở bằng tay, app khởi động rất chậm vì bản
debug chạy JIT. Dùng `flutter run` thay vì cài APK, hoặc build bản release:

```bash
flutter build apk --release
```

### `flutter: command not found`

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

Thêm dòng đó vào `~/.zshrc` để khỏi phải gõ lại mỗi lần.

### Muốn build cho iPhone

Toàn bộ mã Dart trong `lib/` không có gì riêng cho Android, và thư mục `ios/`
đã được sinh sẵn — nhưng máy hiện chỉ có Command Line Tools chứ chưa có Xcode
đầy đủ, nên chưa build được. Cần làm:

```bash
# 1. Cài Xcode từ App Store (~20 GB, để dành sẵn chỗ trống)
# 2. Trỏ toolchain sang Xcode thay vì Command Line Tools
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch

# 3. CocoaPods đã có sẵn, chỉ cần nạp pod cho project
cd ios && pod install && cd ..

# 4. Kiểm tra rồi chạy
flutter doctor          # dòng Xcode phải chuyển sang [✓]
flutter run -d <iphone>
```

Chạy trên iPhone thật còn cần một Apple ID để ký ứng dụng — mở
`ios/Runner.xcworkspace` bằng Xcode, vào tab **Signing & Capabilities** rồi chọn
Team. Bản ký bằng tài khoản miễn phí hết hạn sau 7 ngày và phải cài lại.

### Xuất model báo `ImportError: cannot import name 'get_cuda_generator_meta_val'`

Đây là lý do dự án dùng `scripts/export_tflite.py` thay vì gọi thẳng
`yolo export format=tflite`. Từ ultralytics 8.4.83, `format='tflite'` bị chuyển
hướng sang `litert-torch`, và gói đó chưa tương thích torch 2.13. Script đi
đường cũ `.pt → ONNX → onnx2tf → TFLite` nên không dính lỗi này.

### Xuất model báo `_pickle.UnpicklingError: could not find MARK`

onnx2tf tải một file ảnh mẫu từ GitHub với hạn chờ 5 giây; mạng chậm là nhận về
nội dung hỏng. Script đã tự sinh file đó tại chỗ để không phải tải. Nếu vẫn gặp,
xoá `calibration_image_sample_data_20x128x128x3_float32.npy` ở thư mục gốc rồi
chạy lại.

---

## 8. Ghi chú kỹ thuật

**Vì sao tên lớp bị hardcode.** File `models/best.pt` lưu tên hai lớp là `'0'`
và `'1'` — tên mặc định Roboflow sinh ra, không mang nghĩa. Ánh xạ thật nằm
trong `training_notebooks/shrimp.ipynb` cell 11:

```python
model.model.names = {0: "Healthy", 1: "Diseased"}
```

Notebook gán đè lúc chạy suy luận nên tên này không được ghi vào file model.
Vì vậy app phải tự định nghĩa trong `lib/core/constants.dart`. **Nếu bạn train
lại và thứ tự lớp đổi, phải sửa file đó**, nếu không app sẽ gọi tôm bệnh thành
tôm khoẻ mà không có dấu hiệu gì.

**Vì sao là 512 chứ không phải 640.** Model train ở `imgsz=512`. Chạy suy luận
ở kích thước khác lúc train sẽ làm giảm độ chính xác.

**Vì sao 5376 anchor.** Ảnh 512 qua ba tầng đặc trưng ở stride 8/16/32:
64² + 32² + 16² = 4096 + 1024 + 256 = 5376. Mỗi anchor có 6 số: 4 toạ độ hộp
+ 2 điểm số lớp.

**Phần lọc theo khoảng cách của web app không được đưa sang đây.** Trong
`app/main.py` có `distance_from_bbox_h()` ước lượng khoảng cách từ chiều cao
hộp, hiệu chuẩn bằng hằng số `REF_SIGN_HEIGHT_M = 0.60` mét. Đó là tàn dư từ
một bài toán nhận diện biển báo giao thông, không có ý nghĩa gì với con tôm.

**Ngưỡng mặc định** conf 0.25 và IoU 0.60 — lấy trùng `INFER_IOU` trong
`app/main.py` để hai đường cho kết quả nhất quán.
