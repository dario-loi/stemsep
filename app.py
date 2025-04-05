import os
import tempfile
import logging
import numpy as np
import subprocess
from pathlib import Path

import gradio as gr
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile
import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stem_separator')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    logger.info("Loading htdemucs_6s model")
    model = get_model("htdemucs_6s")
    logger.info(f"Model loaded on device {device}")
    model.eval()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def create_response(status="success", message="", paths=None, error_code=None):
    """
    Create a standardized response dictionary.
    
    Args:
        status (str): Status of the operation ('success' or 'error')
        message (str): Status message or error description
        paths (dict): Dictionary containing paths to audio files
        error_code (str, optional): Error code for better error handling
        
    Returns:
        dict: Standardized response dictionary
    """
    if paths is None:
        paths = {
            "drums": None,
            "bass": None, 
            "other": None,
            "vocals": None,
            "guitar": None,
            "original": None
        }
    
    return {
        "status": status,
        "message": message,
        "error_code": error_code,
        "paths": paths
    }

def custom_save_audio(wav, path, samplerate):
    try:
        import soundfile as sf
        logger.info(f"Saving audio to {path}, type: {type(wav)}, shape: {wav.shape if hasattr(wav, 'shape') else 'unknown'}")
        
        if torch.is_tensor(wav):
            logger.info("Converting tensor to numpy array")
            wav = wav.detach().cpu().numpy()
        
        if not isinstance(wav, np.ndarray):
            logger.error(f"Expected numpy array, got {type(wav)}")
            raise TypeError(f"Expected numpy array, got {type(wav)}")
        
        if wav.ndim == 1:
            wav = wav[None]
            logger.info(f"Added channel dimension, new shape: {wav.shape}")
        
        wav = wav.T
        logger.info(f"Transposed wav array, shape now: {wav.shape}")
        
        sf.write(path, wav, samplerate)
        logger.info(f"Successfully saved audio to {path}")
        return path
    except Exception as e:
        logger.error(f"Error saving audio to {path}: {e}")
        raise

def process_audio(audio_path):
    logger.info(f"Processing audio from {audio_path}")
    try:
        audio_file = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
        logger.info(f"Audio loaded: type={type(audio_file)}, shape={audio_file.shape}")
        
        if not isinstance(audio_file, np.ndarray):
            logger.warning(f"Expected numpy array after reading audio, got {type(audio_file)}")
            audio_file = np.array(audio_file, dtype=np.float32)
        
        audio_tensor = torch.from_numpy(audio_file).to(torch.float32)
        logger.info(f"Converted to tensor: shape={audio_tensor.shape}")
        
        logger.info(f"Applying model to audio")
        sources = apply_model(model, audio_tensor[None], device=device, progress=True)[0]
        logger.info(f"Model applied, sources tensor shape: {sources.shape}")
        
        sources_np = sources.detach().cpu().numpy()
        logger.info(f"Converted sources to numpy: shape={sources_np.shape}")
        
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        stems = {}
        
        for i, name in enumerate(model.sources):
            if i < len(sources_np):
                output_path = os.path.join(temp_dir, f"{name}.wav")
                logger.info(f"Saving stem {name} (shape: {sources_np[i].shape})")
                custom_save_audio(sources_np[i], output_path, model.samplerate)
                stems[name] = output_path
            else:
                logger.warning(f"Skipping stem {name} as it's out of bounds")
        
        original_output = os.path.join(temp_dir, "original.wav")
        logger.info("Saving original audio")
        custom_save_audio(audio_file, original_output, model.samplerate)
        
        return stems, original_output
    except Exception as e:
        logger.error(f"Error in process_audio: {e}")
        raise

def download_from_youtube(url):
    if not url:
        logger.warning("No YouTube URL provided")
        return None
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "audio")
    logger.info(f"Downloading from YouTube: {url} to {temp_file}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': temp_file,
        'quiet': False,
        'verbose': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Starting YouTube download")
            ydl.download([url])
        audio_path = f"{temp_file}.wav"
        logger.info(f"YouTube download complete: {audio_path}")
        
        if os.path.exists(audio_path):
            return audio_path
        else:
            logger.error(f"Downloaded file not found: {audio_path}")
            return None
    except Exception as e:
        logger.error(f"Error downloading from YouTube: {e}")
        return None

def download_from_spotify(url):
    if not url:
        logger.warning("No Spotify URL provided")
        return None
    
    try:
        logger.info(f"Processing Spotify URL with spotify-dl: {url}")
        temp_dir = tempfile.mkdtemp()
        
        cmd = [
            "spotify_dl", 
            "-l", url, 
            "-o", temp_dir,
            "-s", "y",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"spotify-dl failed: {result.stderr}")
            logger.debug(f"spotify-dl output: {result.stdout}")
            return None
            
        audio_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith((".mp3", ".m4a", ".webm", ".wav")):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            logger.warning("No audio file found after spotify-dl download")
            return None
            
        audio_path = audio_files[0]
        logger.info(f"Found downloaded file: {audio_path}")
        
        if not audio_path.endswith(".wav"):
            wav_path = os.path.join(temp_dir, "converted.wav")
            logger.info(f"Converting {audio_path} to WAV: {wav_path}")
            
            ffmpeg_cmd = ["ffmpeg", "-i", audio_path, wav_path]
            ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if ffmpeg_result.returncode != 0:
                logger.error(f"ffmpeg conversion failed: {ffmpeg_result.stderr}")
                return audio_path
            
            logger.info(f"Conversion successful: {wav_path}")
            return wav_path
        
        return audio_path
    except Exception as e:
        logger.error(f"Error processing Spotify link with spotify-dl: {e}")
        return None

def process_audio_file(audio_path):
    if not audio_path:
        logger.warning("No audio path provided")
        return create_response(
            status="error", 
            message="Please upload an audio file", 
            error_code="NO_AUDIO_FILE"
        )
    
    try:
        logger.info(f"Processing audio file: {audio_path}")
        stems, original_path = process_audio(audio_path)
        
        paths = {
            "drums": stems.get("drums"),
            "bass": stems.get("bass"),
            "other": stems.get("other"),
            "vocals": stems.get("vocals"),
            "guitar": stems.get("guitar"),
            "original": original_path
        }
        
        logger.info(f"Available stems: {list(stems.keys())}")
        logger.info(f"Returning stems: {paths}")
        
        return create_response(
            status="success",
            message="Processing complete!",
            paths=paths
        )
    except Exception as e:
        error_msg = f"Error processing audio: {str(e)}"
        logger.error(error_msg)

        return create_response(
            status="error",
            message=error_msg,
            error_code="PROCESSING_ERROR"
        )

def process_youtube(youtube_url):
    try:
        if not youtube_url:
            return create_response(
                status="error",
                message="Please enter a YouTube URL",
                error_code="NO_YOUTUBE_URL"
            )
            
        logger.info(f"Processing YouTube URL: {youtube_url}")
        audio_path = download_from_youtube(youtube_url)
        if not audio_path:
            logger.warning("Failed to download from YouTube")
            return create_response(
                status="error",
                message="Failed to download from YouTube",
                error_code="YOUTUBE_DOWNLOAD_ERROR"
            )
        
        return process_audio_file(audio_path)
    except Exception as e:
        error_msg = f"Error processing YouTube URL: {str(e)}"
        logger.error(error_msg)
        return create_response(
            status="error",
            message=error_msg,
            error_code="YOUTUBE_PROCESSING_ERROR"
        )

def process_spotify(spotify_url):
    try:
        if not spotify_url:
            return create_response(
                status="error",
                message="Please enter a Spotify track URL",
                error_code="NO_SPOTIFY_URL"
            )
            
        logger.info(f"Processing Spotify URL: {spotify_url}")
        audio_path = download_from_spotify(spotify_url)
        if not audio_path:
            logger.warning("Failed to download from Spotify")
            return create_response(
                status="error",
                message="Failed to download from Spotify",
                error_code="SPOTIFY_DOWNLOAD_ERROR"
            )
        
        return process_audio_file(audio_path)
    except Exception as e:
        error_msg = f"Error processing Spotify URL: {str(e)}"
        logger.error(error_msg)
        return create_response(
            status="error",
            message=error_msg,
            error_code="SPOTIFY_PROCESSING_ERROR"
        )

def create_interface():
    logger.info("Creating Gradio interface")
    with gr.Blocks(css="""
        .contain { display: flex; flex-direction: column; }
        .audio-row { display: flex; flex-direction: row; }
        .audio-col { flex: 1; margin: 5px; }
    """,
    title="Audio Source Separation with Demucs",
    theme="soft"
    ) as app:
        
        gr.Markdown("# Audio Source Separation with Demucs")
        gr.Markdown("Upload an audio file or provide a YouTube/Spotify URL to separate the audio into different stems.")
        gr.Markdown("## Instructions")
        gr.Markdown("1. Upload an audio file or enter a YouTube/Spotify URL.")
        gr.Markdown("2. Click 'Process' to start the separation.")
        gr.Markdown("3. Download the separated audio stems.")
        
        status_info = gr.Textbox(label="Status", interactive=False, value="Ready")
        
        # Helper function to extract values from response dictionary for Gradio
        def extract_for_gradio(response):
            paths = response.get("paths", {})
            message = response.get("message", "")
            return (
                paths.get("drums"),
                paths.get("bass"),
                paths.get("other"),
                paths.get("vocals"),
                paths.get("guitar"),
                paths.get("original"),
                message
            )
        
        with gr.Tab("Upload Audio File"):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            process_btn = gr.Button("Process")
            
            with gr.Group(elem_classes="contain"):
                with gr.Row(elem_classes="audio-row"):
                    with gr.Column(elem_classes="audio-col"):
                        drums_output = gr.Audio(label="Drums", show_download_button=True)
                    with gr.Column(elem_classes="audio-col"):
                        bass_output = gr.Audio(label="Bass", show_download_button=True)
                
                with gr.Row(elem_classes="audio-row"):
                    with gr.Column(elem_classes="audio-col"):
                        other_output = gr.Audio(label="Other", show_download_button=True)
                    with gr.Column(elem_classes="audio-col"):
                        vocals_output = gr.Audio(label="Vocals", show_download_button=True)
                
                guitar_output = gr.Audio(label="Guitar", show_download_button=True)
                original_output = gr.Audio(label="Original", show_download_button=True)
            
            process_btn.click(
                fn=lambda x: extract_for_gradio(process_audio_file(x)),
                inputs=[audio_input],
                outputs=[drums_output, bass_output, other_output, vocals_output, guitar_output, original_output, status_info]
            )
        
        with gr.Tab("YouTube URL"):
            youtube_input = gr.Textbox(label="YouTube URL")
            yt_process_btn = gr.Button("Process")
            
            with gr.Group(elem_classes="contain"):
                with gr.Row(elem_classes="audio-row"):
                    with gr.Column(elem_classes="audio-col"):
                        yt_drums_output = gr.Audio(label="Drums", show_download_button=True)
                    with gr.Column(elem_classes="audio-col"):
                        yt_bass_output = gr.Audio(label="Bass", show_download_button=True)
                
                with gr.Row(elem_classes="audio-row"):
                    with gr.Column(elem_classes="audio-col"):
                        yt_other_output = gr.Audio(label="Other", show_download_button=True)
                    with gr.Column(elem_classes="audio-col"):
                        yt_vocals_output = gr.Audio(label="Vocals", show_download_button=True)
                
                yt_guitar_output = gr.Audio(label="Guitar", show_download_button=True)
                yt_original_output = gr.Audio(label="Original", show_download_button=True)
            
            yt_process_btn.click(
                fn=lambda x: extract_for_gradio(process_youtube(x)),
                inputs=[youtube_input],
                outputs=[yt_drums_output, yt_bass_output, yt_other_output, yt_vocals_output, yt_guitar_output, yt_original_output, status_info]
            )
        
        with gr.Tab("Spotify URL"):
            spotify_input = gr.Textbox(label="Spotify Track URL")
            sp_process_btn = gr.Button("Process")
            
            gr.Markdown("### Spotify Authentication")
            gr.Markdown(""" IMPORTANT: You **need** to define the environment variable SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET for authentication.
                            You can get them from the Spotify Developer Dashboard
                            https://developer.spotify.com/dashboard/applications""")
            
            with gr.Group(elem_classes="contain"):
                with gr.Row(elem_classes="audio-row"):
                    with gr.Column(elem_classes="audio-col"):
                        sp_drums_output = gr.Audio(label="Drums", show_download_button=True)
                    with gr.Column(elem_classes="audio-col"):
                        sp_bass_output = gr.Audio(label="Bass", show_download_button=True)
                
                with gr.Row(elem_classes="audio-row"):
                    with gr.Column(elem_classes="audio-col"):
                        sp_other_output = gr.Audio(label="Other", show_download_button=True)
                    with gr.Column(elem_classes="audio-col"):
                        sp_vocals_output = gr.Audio(label="Vocals", show_download_button=True)
                
                sp_guitar_output = gr.Audio(label="Guitar", show_download_button=True)
                sp_original_output = gr.Audio(label="Original", show_download_button=True)
            
            sp_process_btn.click(
                fn=lambda x: extract_for_gradio(process_spotify(x)),
                inputs=[spotify_input],
                outputs=[sp_drums_output, sp_bass_output, sp_other_output, sp_vocals_output, sp_guitar_output, sp_original_output, status_info]
            )
        
    
    return app

if __name__ == "__main__":
    try:
        import soundfile
        logger.info("Found soundfile library")
    except ImportError:
        logger.error("soundfile library not found. Please install it: pip install soundfile")
        print("Error: soundfile library not found. Please install it: pip install soundfile")
        exit(1)
        
    try:
        subprocess.run(["spotify_dl", "--help"], capture_output=True, text=True)
        logger.info("spotify-dl found and working")
    except Exception as e:
        logger.error(f"spotify-dl not properly installed: {e}")
        print("Error: spotify-dl not properly installed. Please install it: pip install spotify-dl")
        exit(1)
        
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        logger.info("ffmpeg found and working")
    except Exception as e:
        logger.error(f"ffmpeg not properly installed: {e}")
        print("Error: ffmpeg not properly installed. Please install it before running this application.")
        exit(1)
        
    try:
        app = create_interface()
        logger.info("Launching Gradio app")
        app.launch(show_error=True, server_name="0.0.0.0", pwa=True)
    except Exception as e:
        logger.error(f"Error launching app: {e}")
        print(f"Error launching app: {e}")