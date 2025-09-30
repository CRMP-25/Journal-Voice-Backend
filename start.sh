#!/bin/bash

echo "🚀 Bootstrapping Journal Voice Backend..."

# ✅ Install all dependencies (safe to rerun)
python3 -m pip install --no-cache-dir -r requirements.txt

# ✅ Pre-warm Whisper model (optional: avoids cold downloads inside pod)
if [ ! -d "/root/.cache/huggingface" ]; then
    echo "🔄 Downloading Whisper model (${MODEL_SIZE:-large-v3})..."
    python3 - <<'PY'
import os
from faster_whisper import WhisperModel

device = os.getenv("DEVICE", "cpu")
compute_type = os.getenv("COMPUTE_TYPE", "int8")
model_size = os.getenv("MODEL_SIZE", "large-v3")
num_threads = int(os.getenv("NUM_THREADS", "4"))

kwargs = dict(device=device, compute_type=compute_type)
if device == "cpu":
    kwargs["cpu_threads"] = num_threads  # only on CPU

WhisperModel(model_size, **kwargs)
print(f"✅ Whisper model ready on {device} ({compute_type}).")
PY

else
    echo "✅ Whisper model already cached."
fi

# ✅ Start FastAPI using uvicorn
PORT=${PORT:-7860}
echo "🚀 Launching FastAPI on port $PORT ..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT
