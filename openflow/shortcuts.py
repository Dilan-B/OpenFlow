"""Windows shortcut and startup-entry management.

Creates .lnk files by driving WScript.Shell through a throwaway VBScript, so
this needs no pywin32 -- one less dependency for a feature that runs twice in
the app's lifetime.

Shortcuts point at ``OpenFlow.exe`` whenever a packaged build is reachable, and
fall back to ``pythonw.exe -m openflow`` only in a source checkout. Both are
console-free: the point is that OpenFlow is an app, not a terminal session --
and pointing at the exe is also what makes Windows show "OpenFlow" rather than
a nameless Python process in the task list.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = "OpenFlow"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pythonw() -> Path:
    """The windowed interpreter next to the current one."""
    exe = Path(sys.executable)
    candidate = exe.with_name("pythonw.exe")
    return candidate if candidate.exists() else exe


def frozen() -> bool:
    """True when running from a PyInstaller build rather than a checkout."""
    return bool(getattr(sys, "frozen", False))


def installed_exe() -> Path | None:
    """The exe an installer dropped, if this machine has one."""
    for var in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
        root = os.environ.get(var)
        if not root:
            continue
        base = Path(root) / "Programs" if var == "LOCALAPPDATA" else Path(root)
        candidate = base / APP_NAME / f"{APP_NAME}.exe"
        if candidate.exists():
            return candidate
    return None


def app_target() -> tuple[Path, str]:
    """What a launcher should run, best first: this exe when we are already
    packaged, an installed build, a local PyInstaller build, and only then
    pythonw + module out of the checkout. Returns (target, arguments)."""
    if frozen():
        return Path(sys.executable).resolve(), ""

    installed = installed_exe()
    if installed:
        return installed, ""

    packaged = PROJECT_ROOT / "dist" / APP_NAME / f"{APP_NAME}.exe"
    if packaged.exists():
        return packaged, ""

    return pythonw(), "-m openflow"


def app_workdir(target: Path) -> Path:
    """Where a launcher should start. A packaged exe runs from its own
    directory; the module form needs the checkout so ``-m openflow`` resolves."""
    if target.name.lower() == f"{APP_NAME.lower()}.exe":
        return target.parent
    return PROJECT_ROOT


def desktop_dir() -> Path:
    return Path(os.path.expanduser("~")) / "Desktop"


def start_menu_dir() -> Path:
    return (
        Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
        / "Microsoft/Windows/Start Menu/Programs"
    )


def startup_dir() -> Path:
    return start_menu_dir() / "Startup"


def _create_shortcut(path: Path, target: Path, arguments: str, workdir: Path,
                     description: str, icon: Path | None = None) -> Path:
    """Write a .lnk via WScript.Shell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    icon_line = f'link.IconLocation = "{icon}"' if icon and icon.exists() else ""
    script = f'''
Set shell = CreateObject("WScript.Shell")
Set link = shell.CreateShortcut("{path}")
link.TargetPath = "{target}"
link.Arguments = "{arguments}"
link.WorkingDirectory = "{workdir}"
link.Description = "{description}"
link.WindowStyle = 7
{icon_line}
link.Save
'''
    with tempfile.NamedTemporaryFile("w", suffix=".vbs", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(script)
        vbs = Path(handle.name)
    try:
        subprocess.run(
            ["cscript", "//Nologo", str(vbs)],
            check=True, capture_output=True, timeout=20,
        )
    finally:
        vbs.unlink(missing_ok=True)
    return path


def install_shortcuts(desktop: bool = True, start_menu: bool = True) -> list[Path]:
    """Create Desktop and Start Menu launchers. Returns what was written."""
    if sys.platform != "win32":
        raise RuntimeError("shortcut installation is Windows-only")

    created: list[Path] = []
    from .ui.icon import ensure_ico

    icon = ensure_ico()
    run_target, arguments = app_target()
    targets = []
    if desktop:
        targets.append(desktop_dir() / f"{APP_NAME}.lnk")
    if start_menu:
        targets.append(start_menu_dir() / f"{APP_NAME}.lnk")

    for target in targets:
        created.append(
            _create_shortcut(
                target,
                run_target,
                arguments,
                app_workdir(run_target),
                "OpenFlow — system-wide voice to text",
                icon,
            )
        )
        log.info("created %s", target)
    return created


def set_launch_at_login(enabled: bool) -> Path | None:
    """Add or remove the Startup-folder shortcut."""
    if sys.platform != "win32":
        return None
    link = startup_dir() / f"{APP_NAME}.lnk"
    if not enabled:
        link.unlink(missing_ok=True)
        return None

    from .ui.icon import ensure_ico

    run_target, arguments = app_target()
    return _create_shortcut(
        link,
        run_target,
        (arguments + " --minimized").strip(),
        app_workdir(run_target),
        "Start OpenFlow at sign-in",
        ensure_ico(),
    )


def launch_at_login_enabled() -> bool:
    return (startup_dir() / f"{APP_NAME}.lnk").exists() if sys.platform == "win32" else False
