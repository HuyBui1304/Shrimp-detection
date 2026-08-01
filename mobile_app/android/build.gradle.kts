allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
// Đồng bộ phiên bản JVM cho mọi module.
//
// Plugin tflite_flutter khai báo Java 11 trong build.gradle của nó, còn Kotlin
// lấy mặc định theo JDK đang chạy Gradle — ở máy này là JDK 21 đi kèm Android
// Studio. Gradle coi việc Java và Kotlin khác target là lỗi:
//
//   Inconsistent JVM-target compatibility detected for tasks
//   'compileDebugJavaWithJavac' (11) and 'compileDebugKotlin' (21)
//
// Ép cả hai về 17 cho khớp với module app. Dùng configureEach (vốn đã lười)
// chứ không bọc trong afterEvaluate, vì khối evaluationDependsOn bên dưới đã
// ép các project đánh giá xong trước đó rồi.
subprojects {
    // Bỏ qua :app — module đó đã tự đặt 17 trong app/build.gradle.kts, và khối
    // evaluationDependsOn bên dưới ép nó đánh giá sớm nên không gắn thêm
    // afterEvaluate vào được nữa.
    if (name != "app") {
        afterEvaluate {
            // Phải sửa ở extension 'android' chứ không phải ở task JavaCompile:
            // AGP đọc compileOptions từ build.gradle của chính plugin rồi mới
            // cấu hình task, nên đặt thẳng lên task sẽ bị ghi đè. Và phải đợi
            // afterEvaluate thì mới chạy sau script của plugin.
            (extensions.findByName("android") as? com.android.build.api.dsl.LibraryExtension)
                ?.compileOptions {
                    sourceCompatibility = JavaVersion.VERSION_17
                    targetCompatibility = JavaVersion.VERSION_17
                }
            tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
                compilerOptions {
                    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
                }
            }
        }
    }
}

subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
