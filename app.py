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
def separate_audio(url, model_choice):
    # تنظیمات دانلود از یوتیوب
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join('temp', '%(title)s.%(ext)s'),
    }
    
    # دیکشنری مدل‌ها
    models = {
        'BS-Roformer-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'BS-Roformer-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
        'Mel-Roformer-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'
    }
    
    try:
        # دانلود فایل
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # پیدا کردن فایل دانلود شده
        downloaded_file = [f for f in os.listdir('temp') if f.endswith('.wav')][0]
        input_path = os.path.join('temp', downloaded_file)
        
        # جداسازی صدا
        os.system(f'audio-separator "{input_path}" --model_filename {models[model_choice]} --output_dir=output')
        
        # پاکسازی فایل‌های موقت
        os.remove(input_path)
        
        return "جداسازی با موفقیت انجام شد!"
    except Exception as e:
        return f"خطا: {str(e)}"
def combine_and_clean(audio_files):
    if len(audio_files) < 2:
        return "حداقل دو فایل صوتی نیاز است!"
    
    try:
        # ترکیب فایل‌ها
        combined_audio = None
        for file in audio_files:
            audio = AudioSegment.from_file(file.name)
            if combined_audio is None:
                combined_audio = audio
            else:
                combined_audio += audio
        
        # حذف سکوت‌ها
        chunks = split_on_silence(
            combined_audio,
            min_silence_len=1000,
            silence_thresh=-40,
            keep_silence=100
        )
        
        # ترکیب قطعات بدون سکوت
        final_audio = chunks[0]
        for chunk in chunks[1:]:
            final_audio += chunk
            
        output_path = "output/combined_vocals.wav"
        final_audio.export(output_path, format="wav")
        
        return output_path
    except Exception as e:
        return f"خطا: {str(e)}"
def process_audio(input_audio, echo_reduction=0.85, presence=0.15):
    try:
        # خواندن فایل صوتی
        audio, sr = librosa.load(input_audio.name, sr=44100, mono=True)
        
        # حذف اکو
        echo_reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=echo_reduction,
            stationary=False,
            n_fft=2048,
            win_length=2048,
            n_std_thresh_stationary=1.2
        )
        
        # تنظیم فضای طبیعی
        b1, a1 = butter(2, [200/22050, 8000/22050], btype='band')
        b2, a2 = butter(2, 4000/22050, btype='high')
        
        filtered = filtfilt(b1, a1, echo_reduced)
        high_freq = filtfilt(b2, a2, echo_reduced) * 0.2
        enhanced = filtered + (high_freq * presence)
        
        # نرمالایز نهایی
        final_audio = librosa.util.normalize(enhanced) * 0.95
        
        output_path = "output/final_processed.wav"
        sf.write(output_path, final_audio, sr, 'PCM_24')
        
        return output_path
    except Exception as e:
        return f"خطا: {str(e)}"
# تعریف رابط کاربری
with gr.Blocks(title="پردازشگر حرفه‌ای صدا") as app:
    gr.Markdown("# 🎵 پردازشگر حرفه‌ای صدا")
    
    with gr.Tab("جداسازی صدا"):
        url_input = gr.Textbox(label="لینک ویدیو")
        model_choice = gr.Dropdown(
            choices=["BS-Roformer-1297", "BS-Roformer-1296", "Mel-Roformer-1143"],
            label="انتخاب مدل",
            value="BS-Roformer-1297"
        )
        separate_button = gr.Button("شروع جداسازی")
        separate_output = gr.Textbox(label="نتیجه")
        separate_button.click(separate_audio, [url_input, model_choice], separate_output)
    
    with gr.Tab("ترکیب صداها"):
        audio_files = gr.File(file_count="multiple", label="انتخاب فایل‌های صوتی")
        combine_button = gr.Button("ترکیب و حذف سکوت")
        combined_output = gr.Audio(label="خروجی")
        combine_button.click(combine_and_clean, audio_files, combined_output)
    
    with gr.Tab("پردازش نهایی"):
        input_audio = gr.File(label="انتخاب فایل صوتی")
        with gr.Row():
            echo_slider = gr.Slider(minimum=0.7, maximum=0.95, value=0.85, label="میزان حذف اکو")
            presence_slider = gr.Slider(minimum=0.1, maximum=0.3, value=0.15, label="میزان حضور صدا")
        process_button = gr.Button("شروع پردازش")
        final_output = gr.Audio(label="خروجی نهایی")
        process_button.click(process_audio, [input_audio, echo_slider, presence_slider], final_output)

# اجرای برنامه
if __name__ == "__main__":
    app.launch(share=True)
