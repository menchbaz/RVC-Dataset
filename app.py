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
import shutil

# [Previous imports and helper functions remain the same until the Gradio interface]

# Create Gradio interface
with gr.Blocks(title="پردازشگر حرفه‌ای صدا") as app:
    gr.Markdown("# 🎵 پردازشگر حرفه‌ای صدا")
    
    with gr.Tab("جداسازی صدا"):
        gr.Markdown("لینک یوتیوب یا آپلود مستقیم فایل‌ها")
        url_input = gr.Textbox(label="لینک ویدیو (اختیاری)")
        file_input = gr.File(
            file_count="multiple",
            file_types=["audio", ".mp3", ".wav", ".m4a", ".ogg", ".aac"],
            label="آپلود فایل‌ها (اختیاری)"
        )
        model_choice = gr.Dropdown(
            choices=["BS-Roformer-1297", "BS-Roformer-1296", "Mel-Roformer-1143"],
            label="انتخاب مدل",
            value="BS-Roformer-1297"
        )
        separate_button = gr.Button("شروع جداسازی")
        separate_output = gr.Textbox(label="نتیجه")
        separate_button.click(separate_audio, [url_input, file_input, model_choice], separate_output)
    
    with gr.Tab("ترکیب صداها"):
        use_uploaded = gr.Checkbox(label="استفاده از فایل‌های آپلودی", value=False)
        audio_files = gr.File(
            file_count="multiple",
            file_types=["audio", ".mp3", ".wav", ".m4a", ".ogg", ".aac"],
            label="انتخاب فایل‌های صوتی"
        )
        combine_button = gr.Button("ترکیب و حذف سکوت")
        combined_output = gr.Audio(label="خروجی", autoplay=True)
        
        def process_and_play(use_uploaded, files):
            result = combine_and_clean(use_uploaded, files)
            return result, gr.update(autoplay=True)
            
        combine_button.click(
            process_and_play,
            [use_uploaded, audio_files],
            [combined_output, combined_output]
        )
    
    with gr.Tab("پردازش نهایی"):
        with gr.Row():
            echo_slider = gr.Slider(minimum=0.7, maximum=0.95, value=0.9, label="میزان حذف اکو")
            presence_slider = gr.Slider(minimum=0.1, maximum=0.3, value=0.1, label="میزان حضور صدا")
        process_button = gr.Button("شروع پردازش")
        final_output = gr.Audio(label="خروجی نهایی", autoplay=True)
        
        def process_and_autoplay(echo, presence):
            result = process_audio(echo, presence)
            return result, gr.update(autoplay=True)
            
        process_button.click(
            process_and_autoplay,
            [echo_slider, presence_slider],
            [final_output, final_output]
        )

    gr.Markdown("""
    ### 🎥 ما را در یوتیوب دنبال کنید
    [![YouTube Channel](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/@aigolden)
    """)

if __name__ == "__main__":
    app.launch(share=True)