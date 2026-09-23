#!/usr/bin/env bash
# Re-sign OpenFlow.app with the project's own self-signed certificate.
#
#     MACOS_SIGNING_CERT=<base64 .p12> MACOS_SIGNING_CERT_PASSWORD=... \
#         bash scripts/sign_macos.sh dist/OpenFlow.app
#
# macOS files Accessibility, Input Monitoring and Microphone grants under the
# app's designated requirement. PyInstaller's ad-hoc signature makes that a
# hash of this exact build, so every update looked like a new app and lost
# its grants. Signed with one long-lived certificate, the requirement becomes
# "this bundle id, signed by this certificate" -- the same for every release,
# so grants carry over. It is not a Developer ID: Gatekeeper still asks once.
#
# Without the secret (forks, local builds) the ad-hoc signature is kept.
set -euo pipefail

APP="${1:?usage: sign_macos.sh path/to/OpenFlow.app}"

if [[ -z "${MACOS_SIGNING_CERT:-}" ]]; then
    echo "::warning::MACOS_SIGNING_CERT not set; keeping the ad-hoc signature"
    exit 0
fi

WORK="$(mktemp -d)"
KEYCHAIN="$WORK/signing.keychain-db"
KEYCHAIN_PASSWORD="$(uuidgen)"
trap 'security delete-keychain "$KEYCHAIN" 2>/dev/null || true; rm -rf "$WORK"' EXIT

echo "$MACOS_SIGNING_CERT" | base64 --decode > "$WORK/cert.p12"
security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
security set-keychain-settings -lut 3600 "$KEYCHAIN"
security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
security import "$WORK/cert.p12" -k "$KEYCHAIN" \
    -P "${MACOS_SIGNING_CERT_PASSWORD:-}" -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple: -s \
    -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN" >/dev/null

# Self-signed, so it is never "valid" to find-identity -v; pick it by hash.
IDENTITY="$(security find-identity -p codesigning "$KEYCHAIN" \
    | awk '/OpenFlow Code Signing/ {print $2; exit}')"
if [[ -z "$IDENTITY" ]]; then
    echo "::error::no 'OpenFlow Code Signing' identity in the certificate"
    exit 1
fi

# --deep reaches every bundled dylib and framework, so nothing is left with a
# different signer. No hardened runtime: library validation would refuse the
# Python extensions, which is what a self-signed identity is signing.
codesign --force --deep --timestamp=none \
    --keychain "$KEYCHAIN" --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict "$APP"
echo "designated requirement:"
codesign --display --requirements - "$APP" 2>&1 | sed -n 's/^designated => /  /p'
