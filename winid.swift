import CoreGraphics

// Prints the id of the largest on-screen window owned by the Unsloth app, for `screencapture -l`.
let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] ?? []
var best: (id: Int, area: Double)? = nil
for window in windows {
    let owner = (window[kCGWindowOwnerName as String] as? String ?? "").lowercased()
    guard owner.contains("unsloth"), (window[kCGWindowLayer as String] as? Int) == 0 else { continue }
    let bounds = window[kCGWindowBounds as String] as? [String: Double] ?? [:]
    let area = (bounds["Width"] ?? 0) * (bounds["Height"] ?? 0)
    if let id = window[kCGWindowNumber as String] as? Int, area > (best?.area ?? 0) { best = (id, area) }
}
if let best { print(best.id) }
