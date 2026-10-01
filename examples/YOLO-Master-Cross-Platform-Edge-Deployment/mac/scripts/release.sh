#!/usr/bin/env bash
# One-command signed + notarized release of YOLO-Master.app.
# Usage:  mac/scripts/release.sh [version]      (default: the repo VERSION file)
#
# Prereqs (one-time):
#   1. A "Developer ID Application" certificate in your login keychain
#      (Xcode > Settings > Accounts > Manage Certificates > + > Developer ID Application).
#   2. Notary credentials stored as a keychain profile:
#      xcrun notarytool store-credentials ac-notary --apple-id you@x.com --team-id TEAMID --password APP_SPECIFIC_PW
#
# Overrides:  NOTARY_PROFILE=... (default ac-notary)   CODESIGN_ID="Developer ID Application: ... (TEAMID)"
#             BUNDLE_ID=com.you.app                     ARCHS="arm64 x86_64"
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"        # .../mac
VERSION="${1:-$(tr -d '[:space:]' < "$HERE/../VERSION" 2>/dev/null || echo 1.2.0)}"
PROFILE="${NOTARY_PROFILE:-ac-notary}"

echo "== release: YOLO-Master for macOS $VERSION =="

# auto-detect the Developer ID Application identity unless one was passed in.
# `|| true` + awk-with-exit avoids a SIGPIPE/pipefail interaction that used to kill the script silently.
if [ -z "${CODESIGN_ID:-}" ]; then
  CODESIGN_ID="$(security find-identity -v -p codesigning 2>/dev/null \
                 | awk -F'"' '/Developer ID Application/{print $2; exit}' || true)"
fi
if [ -z "${CODESIGN_ID:-}" ]; then
  echo "!! No 'Developer ID Application' certificate found in the keychain." >&2
  echo "   Create one: Xcode > Settings > Accounts > Manage Certificates > + > Developer ID Application" >&2
  echo "   (or developer.apple.com/account > Certificates). Then re-run." >&2
  exit 1
fi
echo "signing identity : $CODESIGN_ID"
echo "notary profile   : $PROFILE"
echo "version          : $VERSION"
echo

CODESIGN_ID="$CODESIGN_ID" NOTARY_PROFILE="$PROFILE" "$HERE/make_app.sh" "$VERSION"

APP="$HERE/dist/YOLO-Master.app"
echo
echo "== verifying =="
codesign --verify --deep --strict --verbose=2 "$APP" && echo "codesign: OK"
xcrun stapler validate "$APP" && echo "staple: OK"
spctl -a -vvv --type exec "$APP"    # expect: source=Notarized Developer ID
echo
echo "shippable: $HERE/dist/YOLO-Master-macOS-$VERSION.zip"
