"""Tests for input-device enumeration.

PortAudio lists the same microphone under every host API, and MME truncates
names at 31 characters. Left alone that produces a dropdown with eighteen
entries, four of which are the same webcam under different spellings. These
tests pin the collapse down to one entry per physical device.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.audio import devices


def _fake_sounddevice(entries, hostapis, default=(1, 5)):
    """A stand-in for the sounddevice module. entries are (name, hostapi_index,
    max_input_channels, samplerate)."""
    module = types.ModuleType("sounddevice")

    def query_devices(index=None):
        rows = [
            {"name": name, "hostapi": api, "max_input_channels": ch,
             "default_samplerate": sr}
            for name, api, ch, sr in entries
        ]
        return rows if index is None else rows[index]

    module.query_devices = query_devices
    module.query_hostapis = lambda i: {"name": hostapis[i]}
    module.default = types.SimpleNamespace(device=default)
    return module


class Enumeration(unittest.TestCase):
    HOSTAPIS = {0: "MME", 1: "Windows DirectSound", 2: "Windows WASAPI",
                3: "Windows WDM-KS"}

    def install(self, entries, default=(1, 5)):
        module = _fake_sounddevice(entries, self.HOSTAPIS, default)
        original = sys.modules.get("sounddevice")
        sys.modules["sounddevice"] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__("sounddevice", original)
            if original is not None else sys.modules.pop("sounddevice", None)
        )

    def test_one_entry_per_device_across_host_apis(self):
        """The same webcam under MME, DirectSound and WASAPI is one microphone."""
        self.install([
            ("Microphone (2- Logitech Webcam ", 0, 2, 44100),     # MME, truncated
            ("Microphone (2- Logitech Webcam C930e)", 1, 2, 44100),
            ("Microphone (2- Logitech Webcam C930e)", 2, 2, 48000),
        ])
        found = devices.list_input_devices()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].hostapi, "Windows WASAPI")

    def test_truncated_mme_name_loses_to_the_full_spelling(self):
        self.install([
            ("Microphone (2- Logitech Webcam ", 0, 2, 44100),
            ("Microphone (2- Logitech Webcam C930e)", 2, 2, 48000),
        ])
        self.assertEqual(devices.list_input_devices()[0].name,
                         "Microphone (2- Logitech Webcam C930e)")

    def test_output_only_devices_are_skipped(self):
        self.install([("Speakers (Realtek)", 2, 0, 48000),
                      ("Microphone (Realtek)", 2, 2, 48000)])
        self.assertEqual([d.name for d in devices.list_input_devices()],
                         ["Microphone (Realtek)"])

    def test_pseudo_devices_are_skipped(self):
        """These are aliases for 'whatever Windows is set to', which the
        System default entry already covers."""
        self.install([("Microsoft Sound Mapper - Input", 0, 2, 44100),
                      ("Primary Sound Capture Driver", 1, 2, 44100),
                      ("Microphone (Realtek)", 2, 2, 48000)])
        self.assertEqual(len(devices.list_input_devices()), 1)

    def test_wdm_ks_only_devices_are_hidden(self):
        """Stereo Mix and Line In show up only under kernel streaming; Windows
        keeps them disabled, and offering them invites a silent recording."""
        self.install([("Stereo Mix (Realtek HD Audio Stereo input)", 3, 2, 44100),
                      ("Microphone (Realtek)", 2, 2, 48000)])
        self.assertEqual([d.name for d in devices.list_input_devices()],
                         ["Microphone (Realtek)"])


class Naming(unittest.TestCase):
    def test_driver_path_names_are_reduced_to_the_friendly_part(self):
        # Backslashes literal, but a real CRLF -- that is what the driver sends.
        raw = (r"Headset (@System32\drivers\bthhfenum.sys,#2;%1 Hands-Free%0"
               "\r\n;(WH-1000XM5))")
        self.assertEqual(devices._pretty(raw), "Headset (WH-1000XM5)")

    def test_empty_parens_are_trimmed(self):
        self.assertEqual(devices._pretty("Dubbing Microphone ()"), "Dubbing Microphone")

    def test_ordinary_names_are_left_alone(self):
        name = "Microphone (Razer BlackShark V2 Pro)"
        self.assertEqual(devices._pretty(name), name)


if __name__ == "__main__":
    unittest.main()
