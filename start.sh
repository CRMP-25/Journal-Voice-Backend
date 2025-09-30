#!/bin/bash

echo "🚀 Bootstrapping Journal Voice Backend..."

# ✅ Install all dependencies (safe to rerun)
python3 -m pip install --no-cache-dir -r requirements.txt

# ✅ Pre-warm Whisper model (optional: avoids cold downloads inside pod)
if [ ! -d "/root/.cache/huggingface" ]; then
    echo "🔄 Downloading Whisper model (${MODEL_SIZE:-large-v3})..."
    python3 - <<'PY'
from faster_whisper import WhisperModel
import os
model = WhisperModel(
    os.getenv("MODEL_SIZE", "large-v3"),
    device=os.getenv("DEVICE", "cpu"),
    compute_type=os.getenv("COMPUTE_TYPE", "int8"),
    cpu_threads=int(os.getenv("NUM_THREADS", "4"))
)
print("✅ Whisper model ready.")
PY
else
    echo "✅ Whisper model already cached."
fi

# ✅ Start FastAPI using uvicorn
PORT=${PORT:-7860}
echo "🚀 Launching FastAPI on port $PORT ..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT
