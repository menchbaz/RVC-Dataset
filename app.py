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

def setup_directories():
    os.makedirs('temp', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)

def download_model():
    model_path = 'models/drumsep.th'
    if not os.path.exists(model_path):
        os.system(f'aria2c https://huggingface.co/Eddycrack864/Drumsep/resolve/main/modelo_final.th -o {model_path}')

def download_audio_from_youtube(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join('temp', '%(title)s.%(ext)s'),
            'restrictfilenames': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return [os.path.join('temp', f) for f in os.listdir('temp') if f.endswith('.wav')]
    except Exception as e:
        return str(e)

def separate_audio(url_or_files, model_choice):
    setup_directories()
    download_model()
    try:
        if isinstance(url_or_files, str) and url_or_files.startswith('http'):
            input_files = download_audio_from_youtube(url_or_files)
        else:
            input_files = [f.name for f in url_or_files]
        
        models = {
            'BS-Roformer-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
            'BS-Roformer-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
            'Mel-Roformer-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'
        }

        for file in input_files:
            os.system(f'audio-separator "{file}" --model_filename {models[model_choice]} --output_dir=output')
            if file.startswith('temp/'):
                os.remove(file)

        return "جداسازی با موفقیت انجام شد!"
    except Exception as e:
        return f"خطا: {str(e)}"

def combine_and_clean(use_uploaded_files, uploaded_files=None):
    try:
        audio_files = []
        if not use_uploaded_files:
            output_files = [f for f in os.listdir('output') if 'Vocals' in f]
            audio_files = [os.path.join('output', f) for f in output_files]
        elif uploaded_files:
            audio_files = [f.name for f in uploaded_files]

        if len(audio_files) < 2:
            return "حداقل دو فایل صوتی نیاز است!"

        combined_audio = None
        for file in audio_files:
            audio = AudioSegment.from_file(file)
            combined_audio = audio if combined_audio is None else combined_audio + audio

        chunks = split_on_silence(
            combined_audio,
            min_silence_len=1000,
            silence_thresh=-40,
            keep_silence=100
        )

        final_audio = chunks[0]
        for chunk in chunks[1:]:
            final_audio += chunk

        output_path = "output/combined_vocals.wav"
        final_audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        return f"خطا: {str(e)}"

def process_audio(echo_reduction=0.9, presence=0.1):
    try:
        input_path = "output/combined_vocals.wav"
        if not os.path.exists(input_path):
            return "لطفا ابتدا از بخش ترکیب صداها استفاده کنید"

        audio, sr = librosa.load(input_path, sr=44100, mono=True)
        
        echo_reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=echo_reduction,
            stationary=False,
            n_fft=2048,
            win_length=2048,
            n_std_thresh_stationary=1.2
        )

        b1, a1 = butter(2, [200/22050, 8000/22050], btype='band')
        b2, a2 = butter(2, 4000/22050, btype='high')
        
        filtered = filtfilt(b1, a1, echo_reduced)
        high_freq = filtfilt(b2, a2, echo_reduced) * 0.2
        enhanced = filtered + (high_freq * presence)
        
        final_audio = librosa.util.normalize(enhanced) * 0.95
        
        output_path = "output/final_processed.wav"
        sf.write(output_path, final_audio, sr, 'PCM_24')
        
        return output_path
    except Exception as e:
        return f"خطا: {str(e)}"

with gr.Blocks(title="پردازشگر حرفه‌ای صدا") as app:
    gr.Markdown("# 🎵 پردازشگر حرفه‌ای صدا")
    
    with gr.Tab("جداسازی صدا"):
        url_input = gr.Textbox(label="لینک ویدیو (اختیاری)")
        file_input = gr.File(file_count="multiple", label="آپلود فایل‌ها (اختیاری)")
        model_choice = gr.Dropdown(choices=["BS-Roformer-1297", "BS-Roformer-1296", "Mel-Roformer-1143"], label="انتخاب مدل", value="BS-Roformer-1297")
        separate_button = gr.Button("شروع جداسازی")
        separate_output = gr.Textbox(label="نتیجه")
        separate_button.click(separate_audio, [url_input, model_choice], separate_output)
    
    with gr.Tab("ترکیب صداها"):
        use_uploaded = gr.Checkbox(label="استفاده از فایل‌های آپلودی", value=False)
        audio_files = gr.File(file_count="multiple", label="انتخاب فایل‌های صوتی")
        combine_button = gr.Button("ترکیب و حذف سکوت")
        combined_output = gr.Audio(label="خروجی")
        combine_button.click(combine_and_clean, [use_uploaded, audio_files], combined_output)
    
    with gr.Tab("پردازش نهایی"):
        echo_slider = gr.Slider(minimum=0.7, maximum=0.95, value=0.9, label="میزان حذف اکو")
        presence_slider = gr.Slider(minimum=0.1, maximum=0.3, value=0.1, label="میزان حضور صدا")
        process_button = gr.Button("شروع پردازش")
        final_output = gr.Audio(label="خروجی نهایی")
        process_button.click(process_audio, [echo_slider, presence_slider], final_output)

if __name__ == "__main__":
    app.launch(share=True)
