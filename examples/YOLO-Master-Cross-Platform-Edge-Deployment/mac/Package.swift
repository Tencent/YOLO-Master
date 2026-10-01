// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YOLOMaster",
    platforms: [.macOS("14.0"), .iOS("17.0")],   // iOS 17 floor matches the p03 mlpackage deployment target   // Sonoma+. Floor is onKeyPress + zero-param onChange (SwiftUI 14); every other API used is 12–13 (Canvas/AV-async 13, Float16 MLMultiArray 12).
    products: [
        .library(name: "YOLOMasterKit", targets: ["YOLOMasterKit"]),
        .executable(name: "yolomaster-coreml", targets: ["YOLOMasterCoreML"]),   // CLI runner
        .executable(name: "YOLOMasterApp", targets: ["YOLOMasterApp"]),          // SwiftUI GUI
    ],
    targets: [
        // Portable C++17 core shared verbatim with the Linux / Windows / Jetson CMake build
        // (cpp/CMakeLists.txt target yolomaster_ccore): in-process mAP, BoT-SORT / ByteTrack,
        // bench statistics. Swift sees only the C header include/ymcore.h (no C++ interop).
        .target(name: "YOLOMasterCore", path: "Sources/YOLOMasterCore", publicHeadersPath: "include"),
        // Shared Core ML inference backend (letterbox -> predict -> decode -> NMS -> annotate).
        .target(name: "YOLOMasterKit", dependencies: ["YOLOMasterCore"], path: "Sources/YOLOMasterKit"),
        // Command-line frontend.
        .executableTarget(
            name: "YOLOMasterCoreML",
            dependencies: ["YOLOMasterKit"],
            path: "Sources/YOLOMasterCoreML"
        ),
        // SwiftUI app frontend (same backend).
        .executableTarget(
            name: "YOLOMasterApp",
            dependencies: ["YOLOMasterKit"],
            path: "Sources/YOLOMasterApp"
        ),
    ],
    cLanguageStandard: .c11,
    cxxLanguageStandard: .cxx17
)
