# -*- mode: python ; coding: utf-8 -*-
import re
import sys

from PyInstaller.utils.hooks import collect_all

IS_MAC = sys.platform == 'darwin'

# openflow/__init__.py is the single source of truth for the version. Read it
# as text: importing the package here would drag its imports into the build.
with open('openflow/__init__.py', encoding='utf-8') as f:
    VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)

# Ship the window icon so ui/icon.py finds it instead of generating one next to
# the install. On macOS that write would land inside the .app and break its
# code signature.
datas = [('assets/openflow.ico', 'assets')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('onnx_asr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('moonshine_voice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# Claude (Pro cleanup) is imported lazily, inside the provider.
tmp_ret = collect_all('anthropic')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'customtkinter', 'pystray'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OpenFlow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=not IS_MAC,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/openflow.icns' if IS_MAC else 'assets/openflow.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    # UPX-compressed dylibs fail macOS code signing checks.
    upx=not IS_MAC,
    upx_exclude=[],
    name='OpenFlow',
)

if IS_MAC:
    # PyInstaller ad-hoc signs the bundle, which Apple Silicon requires before
    # it will run anything. Ad-hoc is not a Developer ID: Gatekeeper still
    # blocks the downloaded app until the user approves it once.
    app = BUNDLE(
        coll,
        name='OpenFlow.app',
        icon='assets/openflow.icns',
        bundle_identifier='io.github.dilan-b.openflow',
        version=VERSION,
        info_plist={
            'CFBundleName': 'OpenFlow',
            'CFBundleDisplayName': 'OpenFlow',
            'CFBundleShortVersionString': VERSION,
            'CFBundleVersion': VERSION,
            # onnxruntime's arm64 wheel is built for macOS 14.
            'LSMinimumSystemVersion': '14.0',
            'NSHighResolutionCapable': True,
            # Without this key macOS denies the microphone without ever
            # asking, and every recording comes back silent.
            'NSMicrophoneUsageDescription':
                'OpenFlow records your voice while you hold the dictation '
                'shortcut and turns it into text.',
        },
    )
