import os
import sys
import signal
import subprocess
import threading
import queue
import math
import tkinter as tk
from tkinter import ttk


def _is_windows():
    return os.name == "nt"


def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def list_input_devices():
    # Lazy import so this file can be imported/py_compile'd without deps.
    import sounddevice as sd  # type: ignore

    devices = sd.query_devices()
    out = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) and d.get("max_input_channels", 0) > 0:
            out.append((idx, d.get("name", f"Device {idx}")))
    return out


class DictationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Whisper Dictation")
        self.root.geometry("820x520")

        self.proc = None
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.reader_thread = None

        # Mic level meter (runs only when dictation is stopped).
        self.meter_q: "queue.Queue[float]" = queue.Queue(maxsize=8)
        self.meter_stop = threading.Event()
        self.meter_thread = None
        self.meter_device_idx = None

        self.var_model = tk.StringVar(value="turbo")
        self.var_lang = tk.StringVar(value="tr")
        self.var_paste = tk.BooleanVar(value=False)  # Deskflow: clipboard-only default
        self.var_device = tk.StringVar(value="")
        self.var_delay = tk.DoubleVar(value=0.0)

        # VAD tuning
        self.var_threshold = tk.DoubleVar(value=0.012)
        self.var_silence = tk.DoubleVar(value=1.5)
        self.var_min_voice = tk.DoubleVar(value=0.5)
        self.var_max_record = tk.DoubleVar(value=30.0)

        # Meter UI
        self.var_meter = tk.DoubleVar(value=0.0)  # 0..100
        self.var_meter_text = tk.StringVar(value="Mic level: --")

        self._build_ui()
        self._refresh_devices()
        self._poll_logs()
        self._poll_meter()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")

        left = ttk.Frame(top)
        left.pack(side="left", fill="x", expand=True)

        right = ttk.Frame(top)
        right.pack(side="right")

        meter = ttk.LabelFrame(left, text="Mic Test", padding=8)
        meter.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 10))
        ttk.Label(meter, textvariable=self.var_meter_text).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.pbar = ttk.Progressbar(meter, maximum=100.0, variable=self.var_meter)
        self.pbar.grid(row=1, column=0, sticky="we")
        meter.columnconfigure(0, weight=1)

        ttk.Label(left, text="Microphone").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        self.cmb_device = ttk.Combobox(left, textvariable=self.var_device, state="readonly", width=58)
        self.cmb_device.grid(row=1, column=1, sticky="we", pady=6)
        self.cmb_device.bind("<<ComboboxSelected>>", lambda _e: self._on_device_change())

        btn_refresh = ttk.Button(right, text="Refresh Devices", command=self._refresh_devices)
        btn_refresh.pack(padx=6, pady=6)

        ttk.Label(left, text="Model").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
        self.cmb_model = ttk.Combobox(
            left,
            textvariable=self.var_model,
            state="readonly",
            values=[
                "turbo",
                "large-v3",
                "large-v2",
                "large-v1",
                "medium",
                "small",
                "base",
                "tiny",
            ],
            width=20,
        )
        self.cmb_model.grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(left, text="Language").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=6)
        ent_lang = ttk.Entry(left, textvariable=self.var_lang, width=10)
        ent_lang.grid(row=3, column=1, sticky="w", pady=6)

        chk = ttk.Checkbutton(left, text="Auto paste (Ctrl/Cmd+V)", variable=self.var_paste)
        chk.grid(row=4, column=1, sticky="w", pady=6)

        ttk.Label(left, text="Start delay (s)").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=6)
        spn = ttk.Spinbox(left, from_=0.0, to=10.0, increment=0.5, textvariable=self.var_delay, width=8)
        spn.grid(row=5, column=1, sticky="w", pady=6)

        adv = ttk.LabelFrame(left, text="VAD Tuning", padding=8)
        adv.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))

        ttk.Label(adv, text="Threshold (RMS)").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Spinbox(adv, from_=0.001, to=0.05, increment=0.001, textvariable=self.var_threshold, width=10).grid(
            row=0, column=1, sticky="w", pady=6
        )
        ttk.Label(adv, text="Silence (s)").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Spinbox(adv, from_=0.2, to=5.0, increment=0.1, textvariable=self.var_silence, width=10).grid(
            row=1, column=1, sticky="w", pady=6
        )
        ttk.Label(adv, text="Min voice (s)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Spinbox(adv, from_=0.1, to=5.0, increment=0.1, textvariable=self.var_min_voice, width=10).grid(
            row=2, column=1, sticky="w", pady=6
        )
        ttk.Label(adv, text="Max record (s)").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Spinbox(adv, from_=5.0, to=300.0, increment=1.0, textvariable=self.var_max_record, width=10).grid(
            row=3, column=1, sticky="w", pady=6
        )

        btn_reset = ttk.Button(
            adv,
            text="Reset",
            command=lambda: (
                self.var_threshold.set(0.012),
                self.var_silence.set(1.5),
                self.var_min_voice.set(0.5),
                self.var_max_record.set(30.0),
            ),
        )
        btn_reset.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        adv.columnconfigure(1, weight=1)
        left.columnconfigure(1, weight=1)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(10, 8))
        self.btn_start = ttk.Button(btns, text="Start Listening", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)

        ttk.Label(btns, text="Clipboard-only works best with Deskflow clipboard sync.").pack(side="left", padx=12)

        log_frame = ttk.LabelFrame(frm, text="Logs", padding=8)
        log_frame.pack(fill="both", expand=True)

        self.txt = tk.Text(log_frame, wrap="word", height=18)
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(state="disabled")

    def _refresh_devices(self):
        try:
            devices = list_input_devices()
        except Exception as e:
            self._append_log(f"[UI] Failed to list devices: {e}\n")
            devices = []

        values = ["Default (System)"] + [f"{idx}: {name}" for idx, name in devices]
        self.cmb_device["values"] = values
        if values and not self.var_device.get():
            self.var_device.set(values[0])
        self._on_device_change()

    def _parse_device(self):
        try:
            v = (self.var_device.get() or "").strip()
            if not v:
                return None
            if v.lower().startswith("default"):
                return "default"
            device_val = v.split(":", 1)[0].strip()
            return int(device_val)
        except Exception:
            return None

    def _on_device_change(self):
        if self.proc is not None:
            return
        dev = self._parse_device()
        if dev is None:
            self._stop_meter()
            self.var_meter_text.set("Mic level: --")
            self.var_meter.set(0.0)
            return
        if self.meter_device_idx != dev:
            self.meter_device_idx = dev
            self._restart_meter()

    def _append_log(self, s: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _poll_logs(self):
        try:
            while True:
                s = self.log_q.get_nowait()
                self._append_log(s)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_logs)

    def _meter_worker(self, device_sel):
        # Runs in background thread.
        try:
            import numpy as np  # type: ignore
            import sounddevice as sd  # type: ignore
        except Exception as e:
            self.log_q.put(f"[UI] Mic meter unavailable (missing deps): {e}\n")
            return

        def callback(indata, frames, time_info, status):
            if self.meter_stop.is_set():
                raise sd.CallbackStop
            # Flatten to mono-ish RMS.
            data = np.asarray(indata, dtype=np.float32)
            rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
            try:
                self.meter_q.put_nowait(rms)
            except queue.Full:
                pass

        try:
            device_arg = None if device_sel == "default" else int(device_sel)

            # Pick a samplerate the device actually supports. 16kHz often fails on Windows devices.
            try:
                if device_arg is None:
                    info = sd.query_devices(kind="input")
                else:
                    info = sd.query_devices(device_arg)
                default_sr = float(info.get("default_samplerate") or 0.0)
            except Exception:
                default_sr = 0.0

            candidates = []
            if default_sr:
                candidates.append(default_sr)
            candidates += [48000.0, 44100.0, 16000.0]

            # De-dupe while preserving order.
            seen = set()
            sr_candidates = []
            for sr in candidates:
                if sr in seen:
                    continue
                seen.add(sr)
                sr_candidates.append(sr)

            last_err = None
            for sr in sr_candidates:
                if self.meter_stop.is_set():
                    return
                try:
                    sd.check_input_settings(device=device_arg, samplerate=sr, channels=1, dtype="float32")
                    blocksize = max(256, int(sr * 0.05))  # ~50ms
                    with sd.InputStream(
                        samplerate=sr,
                        channels=1,
                        dtype="float32",
                        blocksize=blocksize,
                        callback=callback,
                        device=device_arg,
                    ):
                        self.log_q.put(f"[UI] Mic meter started (device={device_sel}, sr={sr:.0f}).\n")
                        while not self.meter_stop.is_set():
                            sd.sleep(100)
                    return
                except Exception as e:
                    last_err = e

            # As a last resort, try the true system default (device=None) even if a device index was selected.
            try:
                sr = default_sr or 48000.0
                sd.check_input_settings(device=None, samplerate=sr, channels=1, dtype="float32")
                with sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    blocksize=max(256, int(sr * 0.05)),
                    callback=callback,
                    device=None,
                ):
                    self.log_q.put(f"[UI] Mic meter started (device=system-default, sr={sr:.0f}).\n")
                    while not self.meter_stop.is_set():
                        sd.sleep(100)
                return
            except Exception as e:
                last_err = e

            raise last_err if last_err else RuntimeError("Unknown mic meter error")
        except Exception as e:
            self.log_q.put(f"[UI] Mic meter error: {e}\n")

    def _stop_meter(self):
        if self.meter_thread is None:
            return
        self.meter_stop.set()
        try:
            self.meter_thread.join(timeout=1.0)
        except Exception:
            pass
        self.meter_thread = None
        self.meter_stop.clear()

    def _restart_meter(self):
        self._stop_meter()
        if self.proc is not None:
            return
        if self.meter_device_idx is None:
            return
        self.meter_thread = threading.Thread(target=self._meter_worker, args=(self.meter_device_idx,), daemon=True)
        self.meter_thread.start()

    def _poll_meter(self):
        # Update UI from meter queue. We only keep the latest sample each tick.
        rms = None
        try:
            while True:
                rms = self.meter_q.get_nowait()
        except queue.Empty:
            pass

        if rms is not None:
            # Scale RMS to 0..100. Voice RMS is usually ~0.005..0.03 on float32 input.
            rms_max = 0.05
            pct = max(0.0, min(100.0, (rms / rms_max) * 100.0))
            self.var_meter.set(pct)

            if rms <= 0:
                db = float("-inf")
                self.var_meter_text.set("Mic level: 0.0000 RMS (-inf dBFS)")
            else:
                db = 20.0 * math.log10(rms)
                self.var_meter_text.set(f"Mic level: {rms:.4f} RMS ({db:.1f} dBFS)")

        self.root.after(50, self._poll_meter)

    def _reader(self):
        assert self.proc is not None
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self.log_q.put(line)
        finally:
            try:
                code = self.proc.wait(timeout=0.2)
            except Exception:
                code = self.proc.poll()
            self.log_q.put(f"\n[UI] Process exited ({code}).\n")
            self.proc = None
            self.root.after(0, self._set_running_state, False)
            self.root.after(0, self._on_device_change)

    def _set_running_state(self, running: bool):
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")

    def _build_cmd(self):
        py = sys.executable
        script = os.path.join(_script_dir(), "realtime_dictation.py")

        dev = self._parse_device()

        args = [
            py,
            "-u",
            script,
            "--model",
            self.var_model.get().strip(),
            "--language",
            self.var_lang.get().strip(),
            "--delay",
            str(float(self.var_delay.get())),
            "--threshold",
            str(float(self.var_threshold.get())),
            "--silence",
            str(float(self.var_silence.get())),
            "--min-voice",
            str(float(self.var_min_voice.get())),
            "--max-record",
            str(float(self.var_max_record.get())),
        ]
        if isinstance(dev, int):
            args += ["--device", str(dev)]

        if self.var_paste.get():
            args += ["--paste"]
        else:
            args += ["--no-paste"]
        return args

    def start(self):
        if self.proc is not None:
            return

        # Avoid device contention: stop mic meter while dictation is running.
        self._stop_meter()

        cmd = self._build_cmd()
        self._append_log("[UI] Starting:\n  " + " ".join(cmd) + "\n\n")

        env = os.environ.copy()
        # Make child Python speak UTF-8 on Windows to avoid UnicodeEncodeError.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        self.proc = subprocess.Popen(
            cmd,
            cwd=_script_dir(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()
        self._set_running_state(True)

    def stop(self):
        if self.proc is None:
            return

        self._append_log("\n[UI] Stopping...\n")
        try:
            if self.proc.stdin:
                self.proc.stdin.write("__STOP__\n")
                self.proc.stdin.flush()
        except Exception:
            pass

        # Fallback: terminate/kill if it doesn't exit quickly.
        try:
            self.proc.wait(timeout=3.0)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def _on_close(self):
        self.stop()
        self._stop_meter()
        self.root.after(200, self.root.destroy)


def main():
    root = tk.Tk()
    app = DictationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
