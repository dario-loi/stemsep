# Audio Stem Separator

A web application that separates audio into stems (vocals, drums, bass, guitar, other) using Demucs. Supports uploads, YouTube URLs, and Spotify links.

## Features

- Upload audio files for stem separation
- Extract stems from YouTube videos
- Extract stems from Spotify tracks
- High-quality separation using Demucs htdemucs_6s model
- Easy-to-use web interface built with Gradio

## Installation

### Option 1: Using pip

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install ffmpeg (required for audio processing):
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

3. Run the application:
   ```bash
   python app.py
   ```

### Option 2: Using Docker

1. Build and run with Docker:
   ```bash
   docker build -t audio-stem-separator .
   docker run -p 7860:7860 audio-stem-separator
   ```

### Option 3: Using Docker Compose

1. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Spotify API Setup

**Important**: While spotify-dl doesn't require API keys for downloading, it still needs them to access Spotify metadata.

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
