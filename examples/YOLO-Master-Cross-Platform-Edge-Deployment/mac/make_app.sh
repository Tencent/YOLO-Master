#!/usr/bin/env bash
# Build + package YOLO-Master.app as a redistributable, universal (Apple Silicon + Intel) macOS app.
# Run on macOS with the Swift toolchain + Command Line Tools. Usage: mac/make_app.sh [version]
#
#   Basic (ad-hoc signed; recipients right-click > Open on first launch):
#       mac/make_app.sh 1.0.0
#   Apple-Silicon only (smaller/faster build):
#       ARCHS=arm64 mac/make_app.sh
#   Developer ID signed (Gatekeeper-clean, no right-click needed once notarized):
#       CODESIGN_ID="Developer ID Application: Your Name (TEAMID)" mac/make_app.sh 1.0.0
#   Signed + notarized (fully shippable to anyone):
#       CODESIGN_ID="Developer ID Application: ..." NOTARY_PROFILE=ac-notary mac/make_app.sh 1.0.0
#       (create the profile once: xcrun notarytool store-credentials ac-notary
#          --apple-id you@example.com --team-id TEAMID --password APP_SPECIFIC_PW)
#
# Note: intentionally NOT using `set -u` -- macOS's stock bash 3.2 aborts on benign expansions.
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../mac
APP_NAME="YOLO-Master"                         # display + .app bundle name (Finder / Dock / menu bar)
EXEC_NAME="YOLOMaster"                          # Mach-O filename inside the bundle (no spaces -> simple paths)
ZIP_SLUG="YOLO-Master-macOS"                    # zip filename base (no spaces)
BUNDLE_ID="${BUNDLE_ID:-com.yolomaster.coreml}"   # override with BUNDLE_ID=com.you.app to use your own reverse-domain id
VERSION="${1:-$(tr -d '[:space:]' < "$HERE/../VERSION" 2>/dev/null || echo 1.2.0)}"   # repo VERSION file, like package_linux.sh
ARCHS="${ARCHS:-arm64 x86_64}"                 # universal by default; override e.g. ARCHS=arm64
DIST="$HERE/dist"
APP="$DIST/$APP_NAME.app"
ZIP="$DIST/${ZIP_SLUG}-${VERSION}.zip"

archargs=""
for a in $ARCHS; do archargs="$archargs --arch $a"; done

echo "[1/4] swift build -c release (arch:$ARCHS)..."
swift build -c release --package-path "$HERE" $archargs --product YOLOMasterApp
BIN="$(swift build -c release --package-path "$HERE" $archargs --show-bin-path)/YOLOMasterApp"
[ -x "$BIN" ] || { echo "build product not found: $BIN" >&2; exit 1; }
echo "  binary: $(lipo -archs "$BIN" 2>/dev/null || echo "$ARCHS")"

echo "[2/4] assembling $APP ..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/$EXEC_NAME"

# app icon (rendered from AppIcon.icon via scripts/make_icon.py; prebuilt .icns lives in Resources/)
ICON_KEY=""
if [ -f "$HERE/Resources/AppIcon.icns" ]; then
  cp "$HERE/Resources/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
  ICON_KEY="  <key>CFBundleIconFile</key>          <string>AppIcon</string>"
  echo "  icon: AppIcon.icns"
fi

# bundled default model (auto-loaded on launch; user can pick another). The .mlpackage is
# gitignored, so a fresh clone has none - fetch it from the models release, and FAIL rather
# than ship a model-less app (the 1.1.0 zip shipped without one exactly this way).
MODEL_URL="https://github.com/skywalker-lt/yolo-master-edge/releases/download/v0.1-seg-n/v0.1-seg-N.mlpackage.zip"
if ! ls -d "$HERE"/Resources/*.mlpackage >/dev/null 2>&1; then
  echo "  default model missing - fetching from the v0.1-seg-n release..."
  curl -fsSL "$MODEL_URL" -o "$HERE/Resources/model.zip" \
    && unzip -q "$HERE/Resources/model.zip" -d "$HERE/Resources/" \
    && rm -f "$HERE/Resources/model.zip"
fi
FOUND_MODEL=0
for pkg in "$HERE"/Resources/*.mlpackage; do
  [ -d "$pkg" ] || continue
  cp -R "$pkg" "$APP/Contents/Resources/"
  echo "  default model: $(basename "$pkg")"
  FOUND_MODEL=1
done
if [ "$FOUND_MODEL" = 0 ]; then
  echo "ERROR: no .mlpackage in mac/Resources/ and the download failed."
  echo "       Fetch it manually: curl -L $MODEL_URL -o m.zip && unzip m.zip -d mac/Resources/"
  exit 1
fi

# acknowledgement logos for the About page (optional; graceful fallback if absent)
if [ -d "$HERE/Resources/ack" ]; then
  cp -R "$HERE/Resources/ack" "$APP/Contents/Resources/ack"
  echo "  ack logos: $(ls "$HERE/Resources/ack" 2>/dev/null | tr '\n' ' ')"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key>       <string>${APP_NAME}</string>
  <key>CFBundleExecutable</key>        <string>${EXEC_NAME}</string>
  <key>CFBundleIdentifier</key>        <string>${BUNDLE_ID}</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
${ICON_KEY}
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key>           <string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key>    <string>14.0</string>
  <key>NSCameraUsageDescription</key>  <string>Live real-time object detection runs the selected Core ML model on the camera feed. Frames are processed on-device and never leave your Mac.</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>LSApplicationCategoryType</key> <string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

echo "[3/4] codesign..."
ENTITLEMENTS="$HERE/YOLOMaster.entitlements"   # camera entitlement (required under hardened runtime)
ENT_ARG=(); [ -f "$ENTITLEMENTS" ] && ENT_ARG=(--entitlements "$ENTITLEMENTS")
if [ -n "${CODESIGN_ID:-}" ]; then
  codesign --force --options runtime --timestamp "${ENT_ARG[@]}" --sign "$CODESIGN_ID" "$APP"
  echo "  signed: $CODESIGN_ID (hardened runtime${ENT_ARG:+, camera entitlement})"
  codesign --verify --strict "$APP" && echo "  verify: OK"
else
  codesign --force "${ENT_ARG[@]}" --sign - "$APP" 2>/dev/null \
    && echo "  ad-hoc signed (recipients: right-click > Open on first launch)" \
    || echo "  (codesign unavailable -- app still runs locally)"
fi

echo "[4/4] packaging -> $ZIP"
mkdir -p "$DIST"; rm -f "$ZIP"
/usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"

if [ -n "${NOTARY_PROFILE:-}" ]; then
  echo "[notarize] submitting to Apple (this can take a few minutes)..."
  xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
  rm -f "$ZIP"; /usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"   # re-zip the stapled app
  echo "  notarized + stapled."
fi

echo
echo "done."
echo "  app: $APP"
echo "  zip: $ZIP   ($(du -h "$ZIP" 2>/dev/null | cut -f1))"
echo "  run locally:  open \"$APP\""
echo "  ship:         send the .zip. If ad-hoc/unsigned, tell the recipient to right-click the app"
echo "                and choose Open on first launch (Gatekeeper). Developer ID + notarize = no prompt."
