// YOLO-Master iOS - live detection + benchmark, one Kit shared with macOS.
// The inference backend is YOLOMasterKit (mac/Sources/YOLOMasterKit) verbatim:
// same letterbox -> Core ML -> decode -> NMS path as the Mac CLI and GUI.
import SwiftUI

@main
struct YOLOMasterIOSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
