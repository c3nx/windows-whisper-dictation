# Windows Whisper Dictation (VAD)

Real-time voice-to-text dictation for Windows using OpenAI Whisper with Voice Activity Detection.

**Always-on listening** - detects when you speak, types where your cursor is.

## Features

- **VAD (Voice Activity Detection)** - Automatically starts recording when speech is detected
- **Auto-paste** - Types recognized text to active window via clipboard
- **Hallucination filter** - Filters out common Whisper hallucinations ("M.K.", "Subscribe", etc.)
- **GPU acceleration** - CUDA support for fast transcription
- **Multi-language** - Works with any language Whisper supports

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with CUDA (recommended) or CPU
- FFmpeg in PATH
- Microphone

## Installation

```powershell
# Clone
git clone https://github.com/c3nx/windows-whisper-dictation.git
cd windows-whisper-dictation

# Create virtual environment
python -m venv winsper_env
.\winsper_env\Scripts\Activate.ps1

# Install Whisper from GitHub (latest)
pip install git+https://github.com/openai/whisper.git

# Install other dependencies
pip install sounddevice pyautogui pyperclip scipy

# Install PyTorch with CUDA 12.4 (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install FFmpeg (if not installed)
winget install ffmpeg
```

## Usage

```powershell
# Option 1: Double-click run_dictation.bat

# Option 2: Run directly
.\winsper_env\Scripts\Activate.ps1
python realtime_dictation.py
```

1. Run the script
2. Click where you want to type (3 second delay)
3. Speak naturally
4. Text appears automatically after silence
5. Press `Ctrl+C` to stop

## Configuration

Edit `realtime_dictation.py`:

```python
MODEL_TYPE = "large-v3"    # tiny, base, small, medium, large, large-v3
LANGUAGE = "tr"            # Language code (tr, en, de, fr, es, etc.)
THRESHOLD_RMS = 0.012      # Mic sensitivity (lower = more sensitive)
SILENCE_DURATION_S = 1.5   # Seconds of silence before processing
MIN_VOICE_DURATION_S = 0.5 # Minimum speech duration
```

## Models

| Model | Size | Accuracy | Speed (GPU) |
|-------|------|----------|-------------|
| tiny | 39M | Low | Instant |
| base | 74M | Medium | Instant |
| small | 244M | Good | Instant |
| medium | 769M | Very Good | Fast |
| large | 1.5GB | Excellent | Fast |
| **large-v3** | 1.5GB | **Best** | Fast |

First run downloads the model (~3GB for large-v3).

## Troubleshooting

**CUDA not available:**
```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Hallucinations (random text like "M.K."):**
- Increase `THRESHOLD_RMS` (e.g., 0.02)
- Increase `MIN_VOICE_DURATION_S` (e.g., 0.7)
- Add patterns to `HALLUCINATION_PATTERNS` list

**Not detecting speech:**
- Decrease `THRESHOLD_RMS` (e.g., 0.008)
- Check microphone is default in Windows Settings

**Slow on CPU:**
- Use smaller model (`small` or `base`)
- Install PyTorch with CUDA support

## Hallucination Filter

The script filters known Whisper hallucinations:
- "Altyazı M.K." (Turkish subtitle watermark)
- "Subscribe", "Thanks for watching"
- Very short outputs (< 3 chars)
- Punctuation-only outputs

Add custom patterns in `HALLUCINATION_PATTERNS` list.

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper)
- Original concept: [haliul/windows-whisper-dictation](https://github.com/haliul/windows-whisper-dictation)

## License

MIT
