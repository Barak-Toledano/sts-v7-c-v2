# OpenAI Realtime Assistant Environment Configuration

# API Keys and Authentication
# --------------------------
OPENAI_API_KEY=

# Model Configuration
# --------------------------
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview-2024-12-17
OPENAI_API_BASE_URL=https://api.openai.com/v1

# Voice Configuration
# --------------------------
# Options: alloy, ash, ballad, coral, echo, sage, shimmer, verse
VOICE=alloy

# Audio Settings
# --------------------------
AUDIO_FORMAT=pcm16
AUDIO_SAMPLE_RATE=24000
AUDIO_CHANNELS=1
AUDIO_CHUNK_SIZE=4096

# Voice Activity Detection
# --------------------------
VAD_THRESHOLD=0.85
SILENCE_DURATION_MS=1200
PREFIX_PADDING_MS=500

# Logging Settings
# --------------------------
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE_ENABLED=true
LOG_CONSOLE_ENABLED=true

# Application Settings
# --------------------------
DEBUG_MODE=false
DATA_DIR=data
