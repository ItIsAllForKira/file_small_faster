# Audio File Transcriber

**Offline audio-to-text transcription tool** powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).  
This desktop application transcribes local audio/video files (MP3, WAV, MP4, etc.) into English text using a local Whisper model—**no internet required after the first launch**.

> **Note**: Designed for batch processing of pre-recorded content. For live streams, use the companion app `Stream Recorder and Transcriber`.

---

## ✨ Features

- **Supports multiple formats**: MP3, WAV, MP4 (any file with audio track)
- **Automatic audio conversion**: resamples to 16 kHz mono using **PyAV**
- **Accurate transcription** using Whisper `small` model (`int8`, CPU-friendly)
- **Sentence-by-sentence output** with visual formatting (red → blue)
- **Export to `.txt`** with same name as input file (e.g., `talk.mp4` → `talk_tr.txt`)
- **Language mode selection**:
  - English → English (forced)
  - Any language → English (auto-detect + translate)
  - Russian → English (explicit)
- **Real-time timer** and progress feedback
- **Graceful stop** with logging of partial results
- **Temporary file cleanup** after processing
<img width="783" height="426" alt="image" src="https://github.com/user-attachments/assets/a61021fa-b6f7-4ef9-9431-0e61a508c2e0" />

---

## 🧠 Model Details

- Uses **faster-whisper** with the `small` model (better accuracy than `base`)
- **CPU-only**, `int8` quantization for low memory usage (~1–2 GB RAM)
- First run downloads ~500 MB model to `~/.cache/huggingface/hub`
- Model is **loaded once per session**

---

## 📦 Requirements

- Python 3.8+
- Required packages:
  ```bash
  pip install faster-whisper pyside6 pyaudio pydub numpy av
  ```
- ~600 MB free disk space (for model + temp files)
- No FFmpeg required — audio conversion handled by **PyAV**

---

## 🚀 Quick Start

1. **Download or clone** the project.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # if created
   ```
3. **Run the app**:
   ```bash
   python file.py
   ```
4. Click **File**, choose an audio/video file.
5. Select language mode and click **START**.

> On first launch, the `small` Whisper model will be downloaded automatically.

---

## 📁 Output

- **Text**: displayed in real time; each complete sentence appears in **red bold**, then turns **blue bold**.
- **Transcript file**: saved as `{original_name}_tr.txt` in the same directory as the executable.
- **Log**: `file.log` with entries like:  
  `2025-12-11 15:22:10, meeting.mp4, Length: 42:18, Duration: 03:25, Transcription completed`

Log file uses **rotating handler** (max 5 MB, 3 backups).

---

## 🛠️ UI Controls

| Element | Function |
|--------|--------|
| **Language dropdown** | Choose transcription/translation mode |
| **File browser** | Select MP3/WAV/MP4 (or any AV file with audio) |
| **START / STOP** | Begin or interrupt processing |
| **Timer** | Shows elapsed processing time |
| **Text box** | Live transcription output |

---

## ⚠️ Notes

- **Do not close the app abruptly** during processing—use **STOP** to ensure partial results are saved.
- The app **splits text at sentence boundaries** (`. ! ?`) for clean output.
- Input files with **no audio track** will trigger an error.
- For best results, use clear, speech-focused recordings (e.g., interviews, lectures, podcasts).

---

## 📜 License

This tool is for personal or research use.  
Underlying libraries:
- `faster-whisper` — [MIT License](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)
- `PyAV` — [BSD License](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE)
- `PySide6` — [LGPL/GPL]

---

> Built with Python, faster-whisper, and PyAV.  
> Version: `V201125`

---
