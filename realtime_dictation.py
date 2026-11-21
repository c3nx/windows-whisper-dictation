import whisper
import sounddevice as sd
import numpy as np
import pyautogui
import pyperclip
import torch
import threading
import queue
import time
import re
import sys
import argparse
from scipy.signal import butter, lfilter

# Windows console codepages can be cp1252/cp1254 etc. Ensure we never crash on prints.
def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_stdio()

STOP_TOKEN = "__STOP__"

# --- AYARLAR ---
MODEL_TYPE = "turbo"  # large-v3-turbo (hizli), large-v3 (daha kaliteli) da secilebilir
LANGUAGE = "tr"        # Türkçe
SAMPLE_RATE = 16000    # Whisper 16kHz bekler
CHANNELS = 1
BLOCK_DURATION_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)

# VAD (Voice Activity Detection) Ayarları
THRESHOLD_RMS = 0.012       # Ses eşiği (iyi mikrofon için düşük)
SILENCE_DURATION_S = 1.5    # Kaç saniye sessizlik = cümle sonu
MIN_VOICE_DURATION_S = 0.5  # Minimum ses süresi (kısa sesleri yoksay)
MAX_RECORD_DURATION_S = 30  # Maksimum kayıt süresi
AUTO_PASTE = True           # False yaparsan sadece panoya kopyalar
START_DELAY_S = 3.0         # Baslamadan once hedef alana tiklamak icin bekleme
INPUT_DEVICE = None         # sounddevice input device index (None = default)

# Whisper Hallucination Filtresi - Bilinen sahte çıktılar
HALLUCINATION_PATTERNS = [
    r"altyazı\s*m\.?\s*k\.?",
    r"m\.?\s*k\.?",
    r"alt\s*yazı",
    r"abone\s*ol",
    r"beğen",
    r"subscribe",
    r"thanks\s*for\s*watching",
    r"like\s*and\s*subscribe",
    r"^[\s\.\,\!\?]+$",  # Sadece noktalama
    r"^\s*$",            # Boş
]

# Kuyruklar
audio_queue = queue.Queue()
text_queue = queue.Queue()

# Global değişkenler
is_running = True
model = None
use_fp16 = False
device = "cpu"


def list_input_devices():
    """Kullanilabilir mikrofonlari (input device) listeler."""
    devices = sd.query_devices()
    out = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) and d.get("max_input_channels", 0) > 0:
            out.append((idx, d.get("name", f"Device {idx}")))
    return out


def request_stop(reason=""):
    global is_running
    if not is_running:
        return
    is_running = False
    if reason:
        print(f"[SYSTEM] Stop requested: {reason}")
    # Unblock threads / main loop.
    try:
        audio_queue.put_nowait(None)
    except Exception:
        pass
    try:
        text_queue.put_nowait(None)
    except Exception:
        pass


def stdin_control_thread():
    # Allows GUI to stop gracefully without Ctrl+Break (which can hard-abort native libs).
    while is_running:
        try:
            line = sys.stdin.readline()
            if not line:
                return
            cmd = line.strip()
            if not cmd:
                continue
            if cmd == STOP_TOKEN or cmd.lower() in {"stop", "quit", "exit", "q"}:
                request_stop("stdin")
                return
        except Exception:
            return


def rms(data):
    """RMS (Root Mean Square) hesapla."""
    return np.sqrt(np.mean(data**2))


def is_hallucination(text):
    """Metnin Whisper hallucination olup olmadığını kontrol et."""
    if not text:
        return True

    text_lower = text.lower().strip()

    # Çok kısa metinler (3 karakterden az)
    if len(text_lower) < 3:
        return True

    # Bilinen hallucination pattern'leri
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            print(f"[FILTER] Hallucination tespit edildi: '{text}'")
            return True

    return False


def get_paste_hotkey():
    """OS'e gore yapistirma kisayolunu dondur."""
    if sys.platform == "darwin":
        return ("command", "v")
    return ("ctrl", "v")


def audio_capture_thread():
    """Ses yakalama thread'i."""
    print("[MIC] Ses yakalama başladı.")
    if INPUT_DEVICE is not None:
        print(f"[MIC] Input device: {INPUT_DEVICE}")
    else:
        try:
            d_in, d_out = sd.default.device
            print(f"[MIC] Default devices: in={d_in} out={d_out}")
        except Exception:
            pass

    state = {
        "last_speech_time": time.time(),
        "buffer": [],
        "is_speaking": False,
        "speech_start_time": 0
    }

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC] Stream durumu: {status}")

        data = indata.astype(np.float32).flatten()
        current_rms = rms(data)
        now = time.time()

        if current_rms > THRESHOLD_RMS:
            state["last_speech_time"] = now
            if not state["is_speaking"]:
                print("[MIC] Ses algılandı...")
                state["is_speaking"] = True
                state["speech_start_time"] = now
                state["buffer"] = [data.copy()]
            else:
                state["buffer"].append(data.copy())

            # Çok uzun kayıt kontrolü
            if (now - state["speech_start_time"]) > MAX_RECORD_DURATION_S and state["buffer"]:
                print("[MIC] Maksimum süre aşıldı, işleniyor.")
                audio_queue.put(np.concatenate(state["buffer"]))
                state["buffer"] = []
                state["is_speaking"] = False

        elif state["is_speaking"]:
            if (now - state["last_speech_time"]) > SILENCE_DURATION_S:
                print("[MIC] Sessizlik algılandı, işleniyor.")
                if state["buffer"] and (now - state["speech_start_time"]) >= MIN_VOICE_DURATION_S:
                    audio_queue.put(np.concatenate(state["buffer"]))
                else:
                    print("[MIC] Ses çok kısa, yoksayıldı.")
                state["buffer"] = []
                state["is_speaking"] = False
            else:
                state["buffer"].append(data.copy())

    try:
        stream_kwargs = {
            "samplerate": SAMPLE_RATE,
            "channels": CHANNELS,
            "dtype": "float32",
            "blocksize": BLOCK_SIZE,
            "callback": callback,
        }
        # Some hosts behave differently when device=None is passed explicitly.
        # Only pass device if user selected a specific input device.
        if INPUT_DEVICE is not None:
            stream_kwargs["device"] = INPUT_DEVICE

        with sd.InputStream(**stream_kwargs):
            print(f"[MIC] Dinleniyor (model: {MODEL_TYPE}, eşik: {THRESHOLD_RMS}, sessizlik: {SILENCE_DURATION_S}s)")
            while is_running:
                sd.sleep(100)
    except Exception as e:
        print(f"[HATA] Mikrofon hatası: {e}")
        print("[HATA] Mikrofonu kontrol edin.")
        text_queue.put(None)


def whisper_transcription_thread():
    """Whisper transkripsiyon thread'i."""
    global model, is_running, use_fp16, device
    print("[WHISPER] Transkripsiyon başladı.")

    try:
        print("[WHISPER] Model yükleniyor (bu biraz sürebilir)...")

        # Cihaz kontrolu
        if torch.cuda.is_available():
            device = "cuda"
            use_fp16 = True
            print("[WHISPER] CUDA mevcut - GPU kullanılacak")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            use_fp16 = False
            print("[WHISPER] Apple Silicon MPS mevcut - GPU kullanılacak")
        else:
            device = "cpu"
            use_fp16 = False
            print("[WHISPER] GPU yok - CPU kullanılacak (yavaş olabilir)")

        model = whisper.load_model(MODEL_TYPE, device=device)
        print(f"[WHISPER] Model '{MODEL_TYPE}' yüklendi.")
    except Exception as e:
        print(f"[HATA] Model yüklenemedi: {e}")
        is_running = False
        text_queue.put(None)
        return

    while is_running:
        try:
            audio_data = audio_queue.get(timeout=1)
            if audio_data is None:
                break
            if not is_running:
                break

            duration = len(audio_data) / SAMPLE_RATE
            print(f"[WHISPER] {duration:.2f}s ses işleniyor...")

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Whisper transkripsiyon - hallucination önleme parametreleri
            result = model.transcribe(
                audio_data,
                language=LANGUAGE,
                fp16=use_fp16,
                # Hallucination önleme
                condition_on_previous_text=False,  # Önceki metne bağlı olma
                no_speech_threshold=0.6,           # Sessizlik eşiği (yüksek = daha katı)
                logprob_threshold=-1.0,            # Log-probability eşiği
                compression_ratio_threshold=2.4,   # Sıkıştırma oranı (hallucination = yüksek)
                temperature=0.0,                   # Deterministic çıktı
            )

            text = result["text"].strip()

            # Hallucination kontrolü
            if text and not is_hallucination(text):
                print(f"[WHISPER] Tanındı: {text}")
                text_queue.put(text)
            else:
                print("[WHISPER] Boş veya geçersiz çıktı, yoksayıldı.")

            audio_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            if is_running:
                print(f"[HATA] Transkripsiyon hatası: {e}")
            time.sleep(0.1)


def main():
    """Ana fonksiyon."""
    global is_running
    global MODEL_TYPE, LANGUAGE, THRESHOLD_RMS, SILENCE_DURATION_S, MIN_VOICE_DURATION_S, MAX_RECORD_DURATION_S, AUTO_PASTE, START_DELAY_S, INPUT_DEVICE

    parser = argparse.ArgumentParser(description="Windows Whisper Dictation (VAD)")
    parser.add_argument("--model", default=MODEL_TYPE, help="Whisper model (e.g. turbo, large-v3, medium, small)")
    parser.add_argument("--language", default=LANGUAGE, help="Language code (e.g. tr, en)")
    parser.add_argument("--device", type=int, default=None, help="Input device index (sounddevice)")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit")

    parser.add_argument("--threshold", type=float, default=THRESHOLD_RMS, help="VAD RMS threshold")
    parser.add_argument("--silence", type=float, default=SILENCE_DURATION_S, help="Silence seconds to end utterance")
    parser.add_argument("--min-voice", type=float, default=MIN_VOICE_DURATION_S, help="Minimum voice duration (s)")
    parser.add_argument("--max-record", type=float, default=MAX_RECORD_DURATION_S, help="Maximum record duration (s)")

    parser.add_argument("--paste", dest="auto_paste", action="store_true", default=AUTO_PASTE, help="Auto paste via hotkey")
    parser.add_argument("--no-paste", dest="auto_paste", action="store_false", help="Clipboard only (no key presses)")

    parser.add_argument("--delay", type=float, default=START_DELAY_S, help="Start delay seconds (click target field)")
    parser.add_argument("--no-delay", dest="delay", action="store_const", const=0.0, help="No start delay")

    args = parser.parse_args()

    if args.list_devices:
        print("Input devices:")
        for idx, name in list_input_devices():
            print(f"  {idx}: {name}")
        return

    # Apply runtime config overrides
    MODEL_TYPE = args.model
    LANGUAGE = args.language
    THRESHOLD_RMS = args.threshold
    SILENCE_DURATION_S = args.silence
    MIN_VOICE_DURATION_S = args.min_voice
    MAX_RECORD_DURATION_S = args.max_record
    AUTO_PASTE = args.auto_paste
    START_DELAY_S = args.delay
    INPUT_DEVICE = args.device

    # Start control thread (daemon) so GUI can stop cleanly.
    threading.Thread(target=stdin_control_thread, daemon=True).start()

    print("=" * 50)
    print("  SESLE YAZMA - VAD (Sürekli Dinleme)")
    print("=" * 50)
    print(f"  Dil: {LANGUAGE}")
    print(f"  Model: {MODEL_TYPE}")
    if AUTO_PASTE:
        print(f"  Yapıştırma: {'+'.join(get_paste_hotkey())}")
    else:
        print("  Yapıştırma: Kapalı (sadece panoya kopyalar)")
    print(f"  Durdurmak için: Ctrl+C")
    print("=" * 50)
    if START_DELAY_S > 0:
        print(f"\n{START_DELAY_S:.1f} saniye içinde yazı yazmak istediğiniz yere tıklayın...\n")
        time.sleep(START_DELAY_S)

    capture_thread = threading.Thread(target=audio_capture_thread, daemon=True)
    transcribe_thread = threading.Thread(target=whisper_transcription_thread, daemon=True)

    capture_thread.start()
    transcribe_thread.start()

    try:
        while is_running:
            try:
                text = text_queue.get(timeout=0.5)
                if text is None:
                    print("[SYSTEM] Çıkış sinyali alındı.")
                    is_running = False
                    break

                if text:
                    output = text + " "
                    print(f"[YAZI] '{output}'")

                    try:
                        pyperclip.copy(output)
                        time.sleep(0.1)
                        if AUTO_PASTE:
                            pyautogui.hotkey(*get_paste_hotkey())

                    except Exception as e:
                        print(f"[HATA] Yapıştırma hatası: {e}")

                text_queue.task_done()

            except queue.Empty:
                if not capture_thread.is_alive():
                    print("[HATA] Ses yakalama durdu.")
                    is_running = False
                if not transcribe_thread.is_alive() and model is not None:
                    print("[HATA] Transkripsiyon durdu.")
                    is_running = False
                continue
            except Exception as e:
                print(f"[HATA] Ana döngü hatası: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Ctrl+C - Çıkılıyor...")
    finally:
        request_stop("finally")
        print("[SYSTEM] Threadler durduruluyor...")

        if transcribe_thread.is_alive():
            try:
                audio_queue.put_nowait(None)
            except Exception:
                pass
            transcribe_thread.join(timeout=2)

        print("[SYSTEM] Program sonlandı.")


if __name__ == "__main__":
    main()
