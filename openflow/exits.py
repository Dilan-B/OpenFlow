"""Process exit codes.

argparse owns 2 for a usage error and that cannot be changed, so nothing else
may use it. When "the app rejected its own arguments" and "this machine has no
microphone" both exited 2, a startup shortcut that the app could not parse was
indistinguishable from a laptop with its mic muted -- which is how the broken
"Start with Windows" entry went unnoticed. Keep these distinct.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_USAGE = 2               # argparse, fixed
EXIT_NO_MICROPHONE = 3
EXIT_NO_HOTKEY = 4
EXIT_SELF_TEST_FAILED = 5
