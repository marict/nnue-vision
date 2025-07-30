#!/usr/bin/env bash
# Persist-safe NNUE-Vision setup and training entry-point
# Saves logs and checkpoints in /runpod-volume so they survive POD restarts.

set -euo pipefail
exec 2>&1
mountpoint -q /runpod-volume || echo "/runpod-volume not mounted"

# Initialize timing and logging
start_time=$(date +%s)
log() { printf '[%6ss] %s\n' "$(( $(date +%s) - start_time ))" "$*"; }
err() { log "ERROR: $*" >&2; }
trap 'err "failed at line $LINENO"' ERR

# Check environment
log "Container started"
cd /workspace/repo
[[ -d .git ]] || { err "Repository not found in /workspace/repo"; exit 1; }

# Clean up problematic NVIDIA repositories and update apt
log "updating apt repositories"
find /etc/apt -name "*.list*" -exec grep -l "nvidia\|cuda" {} \; -delete 2>/dev/null || true
rm -rf /var/lib/apt/lists/* || true
apt-get update || { log "apt update failed, retrying..."; dpkg --configure -a || true; apt-get update || true; }
apt-get install -y --no-install-recommends tree htop || true

# Set up pip cache in persistent storage for faster subsequent installs
export PIP_CACHE_DIR="/runpod-volume/pip-cache"
mkdir -p "$PIP_CACHE_DIR"

log "using pip cache dir: $PIP_CACHE_DIR"
pip install -r requirements-dev.txt

# Install Python dependencies
log "installing python dependencies"
pip install -r requirements-dev.txt || { 
    log "pip install failed, upgrading pip and retrying"
    pip install --upgrade pip
    pip install -r requirements-dev.txt
}

# Set debugging env vars
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set up persistent logging
log_dir="/runpod-volume/nnue-vision-logs"
mkdir -p "$log_dir/checkpoints"
[[ -L lightning_logs ]] || { rm -rf lightning_logs 2>/dev/null || true; ln -sf "$log_dir" lightning_logs; }

# Process arguments
filtered_args=("$@")
[[ "${filtered_args[*]}" == *"--log_dir"* ]] || filtered_args+=("--log_dir=$log_dir")

# Check WANDB_API_KEY
[[ -n "${WANDB_API_KEY:-}" ]] || {
    log "ERROR: WANDB_API_KEY environment variable is not set"
    exit 1
}

# Run training
log_file="/runpod-volume/train_$(date +%Y%m%d_%H%M%S).log"
log "starting training -> $log_file"
training_exit_code=0
python -u "${filtered_args[@]}" 2>&1 | tee "$log_file" || training_exit_code=$?

# Keep container alive indefinitely - it's up to the code to kill runpod
log "keeping container alive indefinitely"
tail -f /dev/null