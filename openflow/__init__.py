"""OpenFlow -- system-wide voice-to-text."""

# Single source of truth for the version. pyproject reads it from here, the
# diagnostics report prints it, and the release workflow tags the installer
# with it -- so a release cannot end up labelled three different ways.
__version__ = "1.1.0"
