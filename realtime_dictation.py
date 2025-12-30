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
from scipy.signal import butter, lfilter

# --- AYARLAR ---
MODEL_TYPE = "large-v3"  # En iyi model - 4090 ile hızlı çalışır
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


def audio_capture_thread():
    """Ses yakalama thread'i."""
    print("[MIC] Ses yakalama başladı.")

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
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                          blocksize=BLOCK_SIZE, callback=callback):
            print(f"[MIC] Dinleniyor (model: {MODEL_TYPE}, eşik: {THRESHOLD_RMS}, sessizlik: {SILENCE_DURATION_S}s)")
            while is_running:
                sd.sleep(100)
    except Exception as e:
        print(f"[HATA] Mikrofon hatası: {e}")
        print("[HATA] Mikrofonu kontrol edin.")
        text_queue.put(None)


def whisper_transcription_thread():
    """Whisper transkripsiyon thread'i."""
    global model, is_running, use_fp16
    print("[WHISPER] Transkripsiyon başladı.")

    try:
        print("[WHISPER] Model yükleniyor (bu biraz sürebilir)...")

        # CUDA kontrolü
        use_fp16 = torch.cuda.is_available()
        if use_fp16:
            print(f"[WHISPER] CUDA mevcut - GPU kullanılacak")
        else:
            print(f"[WHISPER] CUDA yok - CPU kullanılacak (yavaş olabilir)")

        model = whisper.load_model(MODEL_TYPE, device=None)
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

    print("=" * 50)
    print("  SESLE YAZMA - VAD (Sürekli Dinleme)")
    print("=" * 50)
    print(f"  Dil: {LANGUAGE}")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Durdurmak için: Ctrl+C")
    print("=" * 50)
    print("\n3 saniye içinde yazı yazmak istediğiniz yere tıklayın...\n")
    time.sleep(3)

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

                        # Ctrl+V ile yapıştır
                        pyautogui.keyDown('ctrlleft')
                        time.sleep(0.05)
                        pyautogui.keyDown('v')
                        time.sleep(0.05)
                        pyautogui.keyUp('v')
                        time.sleep(0.05)
                        pyautogui.keyUp('ctrlleft')

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
        is_running = False
        print("[SYSTEM] Threadler durduruluyor...")

        if transcribe_thread.is_alive():
            audio_queue.put(None)
            transcribe_thread.join(timeout=2)

        print("[SYSTEM] Program sonlandı.")


if __name__ == "__main__":
    main()
