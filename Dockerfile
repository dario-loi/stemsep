FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
SHELL ["/bin/bash", "-c"]
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
RUN uv pip install --system --no-cache-dir --requirements requirements.txt 


RUN python -c "from demucs.pretrained import get_model; model = get_model('htdemucs_6s')"

COPY app.py .

EXPOSE 7860

CMD ["uv", "run", "app.py"]