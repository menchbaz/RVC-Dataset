import os
import gradio as gr
import yt_dlp
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from pydub.silence import split_on_silence

# ایجاد پوشه‌های مورد نیاز
os.makedirs('temp', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

# دانلود مدل اگر وجود نداشته باشد
if not os.path.exists('models/drumsep.th'):
    os.system('aria2c https://huggingface.co/Eddycrack864/Drumsep/resolve/main/modelo_final.th -o models/drumsep.th')
