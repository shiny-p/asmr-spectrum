// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "audioscope",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "audioscope",
            path: "Sources/audioscope"
        )
    ]
)
