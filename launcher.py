"""PyInstaller entry point for OpenFlow.exe."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from openflow.__main__ import main

    sys.exit(main())
