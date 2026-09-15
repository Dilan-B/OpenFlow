#!/usr/bin/env bash
# Package a built OpenFlow.app into a drag-to-Applications disk image.
#
#     bash scripts/build_dmg.sh dist/OpenFlow.app dist_installer/OpenFlow-mac.dmg
#
# Runs after the smoke test on purpose: the signature check below then also
# proves that launching the app did not write anything inside its own bundle,
# which would invalidate the signature and make macOS report it as damaged.
set -euo pipefail

APP="${1:-dist/OpenFlow.app}"
DMG="${2:-dist_installer/OpenFlow-mac.dmg}"

if [[ ! -d "$APP" ]]; then
  echo "no app bundle at $APP" >&2
  exit 1
fi

echo "verifying signature of $APP"
codesign --verify --deep --strict --verbose=2 "$APP"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ditto "$APP" "$STAGE/$(basename "$APP")"
# The Applications shortcut is what makes "drag the app across" obvious.
ln -s /Applications "$STAGE/Applications"

mkdir -p "$(dirname "$DMG")"
rm -f "$DMG"

# hdiutil intermittently fails with "Resource busy" on CI runners; a retry is
# the standard remedy.
for attempt in 1 2 3 4 5; do
  if hdiutil create -volname OpenFlow -srcfolder "$STAGE" -ov -format UDZO "$DMG"; then
    echo "built $DMG"
    exit 0
  fi
  echo "hdiutil failed (attempt $attempt), retrying" >&2
  sleep $((attempt * 5))
done

echo "could not create $DMG" >&2
exit 1
