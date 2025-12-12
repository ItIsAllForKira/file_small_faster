import threading
import sys
import subprocess
import numpy as np
import os
import re
import wave
import time
import logging
from logging.handlers import RotatingFileHandler
import pyaudio
import warnings
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QSpinBox,
    QComboBox, QDoubleSpinBox
)
from PySide6.QtCore import QTimer, Qt, Signal, QCoreApplication, QThread
from PySide6.QtGui import QTextCharFormat, QFont, QColor, QTextCursor, QBrush
from contextlib import contextmanager
import av
cache_dir = os.path.expanduser("~/.cache/huggingface")
hf_home_dir = os.path.join(cache_dir, "hub")
os.makedirs(hf_home_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
import faster_whisper
@contextmanager
def suppress_stdout():
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

execution_logger = logging.getLogger("file_execution")
execution_logger.setLevel(logging.INFO)
if not execution_logger.handlers:
    if getattr(sys, 'frozen', False):
        log_dir = os.path.dirname(sys.executable)
    else:
        log_dir = os.path.dirname(os.path.abspath(__file__))

    log_file_path = os.path.join(log_dir, "file.log")
    handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s, %(file_name)s, Length: %(file_duration)s, Duration: %(duration)s, %(action)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    execution_logger.addHandler(handler)
    execution_logger.propagate = False
class WorkerThread(QThread):
    update_signal = Signal(str, bool, Qt.GlobalColor)
    reset_signal = Signal()
    def __init__(self, parent=None, process_func=None, args=None):
        super().__init__(parent)
        self.process_func = process_func
        self.args = args if args is not None else ()
    def run(self):
        self.process_func(*self.args)
class AudioFileTranscriberApp(QMainWindow):
    update_signal = Signal(str, bool, Qt.GlobalColor)
    reset_signal = Signal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Audio File Transcriber ("small") - V201125')
        self.setGeometry(100, 100, 800, 500)
        self.selected_audio_file = None
        self.current_chunk_ranges = []
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS
            self.output_dir = os.path.dirname(sys.executable)
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = self.base_path
        self.model = None
        self.selected_language_code = "en"
        self.transcript_filename = None
        self.running = False
        self.start_time = 0
        self.temp_wav_path = None
        self.model_lock = threading.Lock()
        self.current_sentence = ""
        self.audio_file_duration = 0
        self.was_stopped_by_user = False  # Новый флаг
        self.init_ui()
        self.update_signal.connect(self.update_status)
        self.reset_signal.connect(self.reset_text_format)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        self.elapsed_time_label = QLabel("Timer 00:00", self)
        self.elapsed_time_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.elapsed_time_label)
        self.status_label = QLabel("Status: Waiting", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Language:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English to English", "ANY to English", "Russian to English"])
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.language_combo)
        layout.addLayout(lang_layout)
        self.audio_file_label = QLabel("Select MP4, WAV, MP3 for transcription and translation into English:")
        layout.addWidget(self.audio_file_label)
        file_layout = QHBoxLayout()
        self.audio_file_entry = QLineEdit()
        self.audio_file_entry.setReadOnly(True)
        file_layout.addWidget(self.audio_file_entry)
        self.browse_button = QPushButton("File")
        self.browse_button.clicked.connect(self.open_audio_file)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)
        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setStyleSheet("background-color: white; color: black;")
        layout.addWidget(self.text_box)
        self.start_button = QPushButton("START")
        self.start_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.start_button)
        central_widget.setLayout(layout)
    def _get_language_code(self):
        selected_text = self.language_combo.currentText()
        if "Auto" in selected_text:
            return None
        elif "English" in selected_text:
            return "en"
        elif "Russian" in selected_text:
            return "en"
        return "en"
    def open_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio files (*.mp4 *.wav *.mp3);;All Files (*)"
        )
        if file_path:
            self.selected_audio_file = file_path
            self.audio_file_entry.setText(file_path)
    def toggle_recording(self):
        if self.running:
            self.stop_recording()
        else:
            if not self.selected_audio_file:
                QMessageBox.warning(self, "ERROR", "Select an audio file")
                return
            self.text_box.clear()
            if self.model is None:
                self.status_label.setText("Status: Loading the model 'small'...")
                QApplication.processEvents()
                try:
                    self.model = faster_whisper.WhisperModel("small", device="cpu", compute_type="int8")
                    self.status_label.setText("Status: The model has been loaded. Processing begins....")
                    QApplication.processEvents()
                except Exception as e:
                    QMessageBox.critical(self, "ERROR", f"Could not load Whisper model: {str(e)}")
                    return
            if self.selected_audio_file.lower().endswith('.wav'):
                try:
                    with wave.open(self.selected_audio_file, 'rb') as wf:
                        rate = wf.getframerate()
                        channels = wf.getnchannels()
                        frames = wf.getnframes()
                        if rate == 16000 and channels == 1:
                            self.audio_file_duration = frames / rate
                            self.start_time = time.time()
                            base_name = os.path.splitext(os.path.basename(self.selected_audio_file))[0]
                            self.transcript_filename = os.path.join(self.output_dir, f"{base_name}_tr.txt")
                            self.running = True
                            self.start_button.setText("STOP")
                            self.status_label.setText("Status: Audio file processing and transcription")
                            self.selected_language_code = self._get_language_code()
                            self.worker_thread = WorkerThread(self, self.process_audio_file, (self.selected_audio_file, self.selected_language_code))
                            self.worker_thread.finished.connect(self.stop_processing_audio_file)
                            self.worker_thread.start()
                            self.timer.start(1000)
                            return
                except Exception as e:
                    QMessageBox.critical(self, "ERROR", f"Could not read WAV file: {str(e)}")
                    return
            self.convert_and_process_file(self.selected_audio_file, self._get_language_code())
    def convert_and_process_file(self, file_path, lang_code):
        self.temp_wav_path = os.path.join(self.output_dir, "temp_converted_16000_mono.wav")
        container = None
        try:
            self.status_label.setText("Status: Converting file to 16000 Hz, mono...")
            container = av.open(file_path)
            
            # --- НОВЫЙ КОД: Получение реальной длительности файла ---
            audio_stream = None
            for stream in container.streams.audio:
                audio_stream = stream
                break
            if audio_stream is None:
                raise ValueError("No audio tracks found in the file")
            
            # Используем duration и time_base для получения точной длительности
            duration_seconds = audio_stream.duration * audio_stream.time_base if audio_stream.duration else container.duration * container.streams.audio[0].time_base
            if duration_seconds is None:
                # Если duration недоступна, пытаемся рассчитать из общего времени потока
                # Это может быть менее точно, но лучше, чем ничего
                duration_seconds = container.duration / av.time_base if container.duration else 0
            self.audio_file_duration = duration_seconds
            # --- /НОВЫЙ КОД ---

            # Продолжаем с конвертацией
            resampler = av.audio.resampler.AudioResampler(
                format=av.AudioFormat('s16').planar,
                layout='mono',
                rate=16000
            )
            with wave.open(self.temp_wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                for frame in container.decode(audio=0):
                    resampled_frames = resampler.resample(frame)
                    for resampled_frame in resampled_frames:
                        audio_array = resampled_frame.to_ndarray().flatten()
                        audio_bytes = audio_array.astype(np.int16).tobytes()
                        wav_file.writeframes(audio_bytes)
        except Exception as e:
            if 'container' in locals() and container is not None:
                container.close()
            QMessageBox.critical(self, "ERROR", f"Error converting with AV: {str(e)}")
            return
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception as close_e:
                    pass
        self.start_time = time.time()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.transcript_filename = os.path.join(self.output_dir, f"{base_name}_tr.txt")
        self.running = True
        self.start_button.setText("STOP")
        self.status_label.setText("Status: Audio file processing and transcription")
        self.worker_thread = WorkerThread(self, self.process_audio_file, (self.temp_wav_path, lang_code))
        self.worker_thread.finished.connect(self.stop_processing_audio_file)
        self.worker_thread.start()
        self.timer.start(1000)
    def process_audio_file(self, file_path, language_code):
        try:
            with open(self.transcript_filename, "a", encoding="utf-8") as transcript_file:
                with wave.open(file_path, "rb") as wav_file:
                    chunk_size = 16000 * 2 * 30
                    frames_total = wav_file.getnframes()
                    current_frame = 0
                    self.current_sentence = ""
                    while current_frame < frames_total and self.running:
                        remaining = frames_total - current_frame
                        read_size = min(remaining, chunk_size)
                        audio_bytes = wav_file.readframes(read_size)
                        if not audio_bytes:
                            break
                        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        with self.model_lock:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                segments, info = self.model.transcribe(audio, language=language_code, vad_filter=True)
                        for segment in segments:
                            text = segment.text.strip()
                            if text:
                                full_text = self.current_sentence + " " + text if self.current_sentence else text
                                sentences = re.split(r'(?<=[.!?])\s+', full_text)
                                for sentence in sentences[:-1]:
                                    sentence = sentence.strip()
                                    if sentence:
                                        self.update_signal.emit(f"{sentence}\n", True, Qt.GlobalColor.red)
                                        transcript_file.write(f"{sentence}\n")
                                        transcript_file.flush()
                                if sentences:
                                    last_part = sentences[-1]
                                    ends_with_punct_and_opt_quotes = re.search(r'[.!?]["\'»«]*$', last_part)
                                    if ends_with_punct_and_opt_quotes:
                                        last_part = last_part.strip()
                                        if last_part:
                                            self.update_signal.emit(f"{last_part}\n", True, Qt.GlobalColor.red)
                                            transcript_file.write(f"{last_part}\n")
                                            transcript_file.flush()
                                        self.current_sentence = ""
                                    else:
                                        self.current_sentence = last_part
                        current_frame += read_size
                    if self.current_sentence:
                        final_sentence = self.current_sentence.strip()
                        if final_sentence:
                            final_sentence_with_dot = final_sentence + "."
                            self.update_signal.emit(f"{final_sentence_with_dot}\n", True, Qt.GlobalColor.red)
                            transcript_file.write(f"{final_sentence_with_dot}\n")
                            transcript_file.flush()
                        self.current_sentence = ""
        except Exception as e:
            logging.exception("DETAILED ERROR processing file")
            logging.error(f"Error type: {type(e).__name__}")
            self.update_signal.emit(f"Error processing file: {str(e)}\n", True, Qt.GlobalColor.red)
        finally:
            self.reset_signal.emit()
    def stop_processing_audio_file(self):
        self.running = False
        self.timer.stop()
        self.status_label.setText("Status: File processing completed.")
        self.start_button.setText("START")

        # Запись в лог завершения происходит всегда при вызове этого метода
        if self.start_time > 0:
            elapsed_time = time.time() - self.start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            duration_formatted = f"{minutes:02}:{seconds:02}"
            file_name = os.path.basename(self.selected_audio_file) if self.selected_audio_file else "Unknown"
            file_duration_formatted = self.format_duration(self.audio_file_duration)
            
            # Определяем действие на основе флага
            action_text = "Transcription stopped by user" if self.was_stopped_by_user else "Transcription completed"
            
            execution_logger.info(action_text,
                                  extra={'file_name': file_name, 'action': action_text, 'duration': duration_formatted, 'file_duration': file_duration_formatted})

        if self.temp_wav_path and os.path.exists(self.temp_wav_path):
            try:
                os.remove(self.temp_wav_path)
            except Exception as e:
                logging.warning(f"Could not delete temporary file: {str(e)}")
        self.temp_wav_path = None
        self.current_sentence = ""
        QMessageBox.information(self, "Information", "File transcription completed.")
    def stop_recording(self):
        self.running = False
        self.timer.stop()
        self.status_label.setText("Status: Processing stopped.")
        self.start_button.setText("START")
        # Устанавливаем флаг, чтобы stop_processing_audio_file знал, что была остановка
        self.was_stopped_by_user = True
        self.current_sentence = ""
        QMessageBox.information(self, "Information", "Processing stopped.")
    def format_duration(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02}:{seconds:02}"
    def update_status(self, text, bold=True, color=Qt.GlobalColor.red):
        cursor = self.text_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        start_pos = cursor.position()
        fmt = QTextCharFormat()
        if bold:
            fmt.setFontWeight(QFont.Weight.Bold)
        fmt.setForeground(QBrush(color))
        cursor.insertText(text, fmt)
        end_pos = cursor.position()
        self.current_chunk_ranges.append((start_pos, end_pos))
        self.text_box.ensureCursorVisible()
    def reset_text_format(self):
        cursor = self.text_box.textCursor()
        normal_format = QTextCharFormat()
        normal_format.setFontWeight(QFont.Weight.Bold)
        normal_format.setForeground(QBrush(Qt.GlobalColor.blue))
        for start, end in self.current_chunk_ranges:
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            cursor.mergeCharFormat(normal_format)
        self.current_chunk_ranges = []
    def update_progress(self):
        elapsed_time = int(time.time() - self.start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        self.elapsed_time_label.setText(f"Timer {minutes:02}:{seconds:02}")
        QCoreApplication.processEvents()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioFileTranscriberApp()
    window.show()
    sys.exit(app.exec())