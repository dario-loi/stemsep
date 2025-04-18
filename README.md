# Audio Stem Separator

A web application that separates audio into stems (vocals, drums, bass, guitar, other) using Demucs. Supports uploads, YouTube URLs, and Spotify links.


![frontend](https://github.com/user-attachments/assets/4faf4b66-30e4-4164-994c-3f0b38a3aaa3)
*A preview of the interface*

## Features

- Upload audio files for stem separation
- Extract stems from YouTube videos
- Extract stems from Spotify tracks
- High-quality separation using [Demucs](https://github.com/facebookresearch/demucs)' `htdemucs_6s` model
- Easy-to-use web interface built with Gradio


## Installation

### Option 1: Using Docker **(Best)**

The most convenient way to run the application is using Docker. Two commands and you're done!

1. Get the latest Docker image:
   ```bash
   docker pull darioloi/stemsep
   ```

2. Run the container:
   ```bash
   docker run -p 7860:7860 darioloi/stemsep
   ```
   This will run the app without GPU support.

3. If you want to use GPU acceleration, ensure you have NVIDIA Container Toolkit installed. Then run:
   ```bash
    docker run --gpus all -p 7860:7860 darioloi/stemsep
    ```

4. Access the web interface at http://localhost:7860

### Option 2: Using pip

If you prefer to run the application locally, you can install it using pip. It's recommended to use a virtual environment to avoid conflicts with other packages. Also check out [uv](https://github.com/astral-sh/uv) for a better local python experience.

> [!WARNING]
> Running locally (or in a .venv) will automatically use CUDA if available. If you want to run it without GPU, change the line in `app.py`:
> ```python
> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> ```
> to:
> ```python
> device = torch.device("cpu")
> ```

1. Clone the repository:
   ```bash
   git clone https://github.com/dario-loi/stemsep.git
   cd stemsep
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install ffmpeg (required for audio processing):
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the web interface at http://localhost:7860


### Option 3: Using Docker Compose

1. Run with Docker Compose, this will automatically enable GPU support
   ```bash
   docker-compose up --build
   ```
   
2. If you want to run it without GPU, use:
   ```bash
   docker-compose -f docker-compose.cpu.yml up --build
   ```

3. Access the web interface at http://localhost:7860

## Spotify API Setup

> [!NOTE]
> While `spotify-dl` doesn't require API keys for downloading, it still needs them to access Spotify metadata.

1. Visit [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications)
2. Create a new application
3. Set environment variables:
   ```bash
   export SPOTIFY_CLIENT_ID="your_client_id"
   export SPOTIFY_CLIENT_SECRET="your_client_secret"
   ```

For Docker, add these variables to your environment:
```bash
docker run -p 7860:7860 \
  -e SPOTIFY_CLIENT_ID="your_client_id" \
  -e SPOTIFY_CLIENT_SECRET="your_client_secret" \
  audio-stem-separator
```

For Docker Compose, add to docker-compose.yml:
```yaml
services:
  stem-separator:
    environment:
      - SPOTIFY_CLIENT_ID=your_client_id
      - SPOTIFY_CLIENT_SECRET=your_client_secret
```

## Usage

1. Access the web interface at http://localhost:7860
2. Choose a tab based on your input (Upload, YouTube, or Spotify)
3. Provide the audio/URL and click "Process"
4. Download the separated stems
