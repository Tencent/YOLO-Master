// About & Licenses page - acknowledgements (upstream projects) + the full project license.
import SwiftUI
import AppKit

/// An acknowledged upstream project.
struct Ack: Identifiable {
    var id: String { repo }
    let name: String       // display name
    let logo: String       // resource base name in Resources/ack/
    let ext: String        // resource extension
    let blurb: String      // one-line description + copyright/license
    let repo: String       // repository URL
}

let acknowledgements: [Ack] = [
    Ack(name: "YOLO-Master @ Tencent", logo: "tencent", ext: "png",
        blurb: "The YOLO-Master detector family this runner packages. © 2026 Tencent - AGPL-3.0.",
        repo: "https://github.com/Tencent/YOLO-Master"),
    Ack(name: "Ultralytics", logo: "ultralytics", ext: "png",
        blurb: "The YOLO training & inference framework YOLO-Master builds on. © 2025 Ultralytics - AGPL-3.0.",
        repo: "https://github.com/ultralytics/ultralytics"),
    Ack(name: "Core ML @ Apple", logo: "apple", ext: "jpeg",
        blurb: "Model conversion and the on-device inference runtime. © 2020-2023 Apple Inc. - BSD-3-Clause.",
        repo: "https://github.com/apple/coremltools"),
]

/// Dev fallback root: when running WITHOUT an app bundle (`swift run` / the bare .build
/// binary) there is no Contents/Resources, so resolve mac/Resources/ from this source file's
/// compile-time path instead. Nil when the checkout isn't present (i.e. a real install, where
/// the bundle path below always wins anyway).
private let devResourcesRoot: URL? = {
    let root = URL(fileURLWithPath: #filePath)          // .../mac/Sources/YOLOMasterApp/Info.swift
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Resources")
    return FileManager.default.fileExists(atPath: root.path) ? root : nil
}()

/// Load an acknowledgement logo: bundle first (Resources/ack/<name>.<ext>, the v1.0.0 path),
/// then the dev checkout fallback so `swift run` shows the real logos too.
func ackLogo(_ name: String, _ ext: String) -> NSImage? {
    let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "ack")
        ?? Bundle.main.url(forResource: name, withExtension: ext)
        ?? devResourcesRoot?.appendingPathComponent("ack/\(name).\(ext)")
    return url.flatMap { NSImage(contentsOf: $0) }
}

/// The app mark for the sidebar header: the bundle's icon when packaged; the checkout's
/// AppIcon.png under `swift run`; the generic app icon as the last resort.
func appMark() -> NSImage {
    if Bundle.main.url(forResource: "AppIcon", withExtension: "icns") != nil,
       let i = NSImage(named: "NSApplicationIcon") { return i }
    if let root = devResourcesRoot, let i = NSImage(contentsOf: root.appendingPathComponent("AppIcon.png")) { return i }
    return NSImage(named: "NSApplicationIcon") ?? NSImage()
}

struct InfoView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var licenseExpanded = false   // license is collapsed by default
    private var version: String { (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "1.2.0" }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("About & Licenses").font(.headline)
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 18).padding(.vertical, 12)
            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("YOLO-Master").font(.title2.bold())
                        Text("Version \(version) · on-device YOLO-Master detection & segmentation via Core ML.")
                            .font(.callout).foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Author").font(.headline)
                        HStack(alignment: .center, spacing: 12) {
                            authorAvatar()   // round, to set it apart from the square org logos
                            VStack(alignment: .leading, spacing: 3) {
                                Text("Thomas Lee").font(.callout.weight(.semibold))
                                Text("The Hong Kong University of Science and Technology")
                                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                                if let url = URL(string: "https://github.com/skywalker-lt") {
                                    Link(destination: url) {
                                        Label("github.com/skywalker-lt", systemImage: "link").font(.caption)
                                    }
                                }
                            }
                            Spacer(minLength: 0)
                        }
                        if let src = URL(string: "https://github.com/skywalker-lt/yolo-master-edge") {
                            Link(destination: src) {
                                Label("Source code · github.com/skywalker-lt/yolo-master-edge",
                                      systemImage: "chevron.left.forwardslash.chevron.right").font(.caption)
                            }.padding(.top, 2)
                        }
                    }

                    VStack(alignment: .leading, spacing: 14) {
                        Text("Acknowledgements").font(.headline)
                        Text("This app stands on the following open-source projects, with gratitude.")
                            .font(.caption).foregroundStyle(.secondary)
                        ForEach(acknowledgements) { a in
                            HStack(alignment: .top, spacing: 12) {
                                logo(a)
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(a.name).font(.callout.weight(.semibold))
                                    Text(a.blurb).font(.caption).foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                    if let url = URL(string: a.repo) {
                                        Link(destination: url) {
                                            Label(a.repo.replacingOccurrences(of: "https://", with: ""), systemImage: "link")
                                                .font(.caption)
                                        }
                                    }
                                }
                                Spacer(minLength: 0)
                            }
                        }
                    }

                    Divider()

                    DisclosureGroup(isExpanded: $licenseExpanded) {
                        Text(acknowledgementLicense)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(14)
                            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(Color(nsColor: .textBackgroundColor)))
                            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
                            .padding(.top, 8)
                    } label: {
                        Text("License").font(.headline)
                    }
                    .tint(.secondary)
                }
                .padding(20)
            }
        }
        .frame(width: 580, height: 660)
        .tint(brandColor)
    }

    @ViewBuilder private func authorAvatar() -> some View {
        Group {
            if let img = ackLogo("skywalker-lt", "png") {
                Image(nsImage: img).resizable().scaledToFill()
            } else {
                Circle().fill(.quaternary).overlay(Image(systemName: "person.fill").foregroundStyle(.secondary))
            }
        }
        .frame(width: 42, height: 42)          // same side length as the organization logos
        .clipShape(Circle())                    // full circle (corner radius = side/2) so it reads as a person
        .overlay(Circle().strokeBorder(Color.primary.opacity(0.12), lineWidth: 1))
    }

    @ViewBuilder private func logo(_ a: Ack) -> some View {
        if let img = ackLogo(a.logo, a.ext) {
            Image(nsImage: img).resizable().scaledToFit().frame(width: 42, height: 42)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        } else {   // fallback until the logo image is bundled
            RoundedRectangle(cornerRadius: 8, style: .continuous).fill(.quaternary)
                .frame(width: 42, height: 42)
                .overlay(Text(String(a.name.prefix(1))).font(.title3.bold()).foregroundStyle(.secondary))
        }
    }
}
