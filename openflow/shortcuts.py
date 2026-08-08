"""Windows shortcut and startup-entry management.

Creates .lnk files by driving WScript.Shell through a throwaway VBScript, so
this needs no pywin32 -- one less dependency for a feature that runs twice in
the app's lifetime.

Everything launches ``pythonw.exe``, which has no console window: the point is
that OpenFlow is an app, not a terminal session.
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


def app_target() -> tuple[Path, str]:
    """What a launcher should run: the packaged exe when it exists, else
    pythonw + module. Returns (target, arguments)."""
    packaged = PROJECT_ROOT / "dist" / "OpenFlow" / "OpenFlow.exe"
    if packaged.exists():
        return packaged, ""
    return pythonw(), "-m openflow"


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
                PROJECT_ROOT,
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
        PROJECT_ROOT,
        "Start OpenFlow at sign-in",
        ensure_ico(),
    )


def launch_at_login_enabled() -> bool:
    return (startup_dir() / f"{APP_NAME}.lnk").exists() if sys.platform == "win32" else False
