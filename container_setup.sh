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

# Install Python dependencies
log "installing python dependencies"
pip install -q -r requirements-dev.txt || { 
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

# Process arguments and check for keep-alive
keep_alive_enabled=false
filtered_args=()
for arg in "$@"; do
    [[ "$arg" == "--keep-alive" ]] && keep_alive_enabled=true || filtered_args+=("$arg")
done
[[ "${filtered_args[*]}" == *"--log_dir"* ]] || filtered_args+=("--log_dir=$log_dir")

# Check WANDB_API_KEY
[[ -n "${WANDB_API_KEY:-}" ]] || {
    log "ERROR: WANDB_API_KEY environment variable is not set"
    [[ "$keep_alive_enabled" == "true" ]] && { log "keeping alive despite error"; exec tail -f /dev/null; }
    exit 1
}

# Run training
log_file="/runpod-volume/train_$(date +%Y%m%d_%H%M%S).log"
log "starting training -> $log_file"
training_exit_code=0
python -u "${filtered_args[@]}" 2>&1 | tee "$log_file" || training_exit_code=$?

# Log result and save model
[[ $training_exit_code -eq 0 ]] && log "training completed successfully" || log "training failed (exit code: $training_exit_code)"
[[ -f visual_wake_words_model.pt ]] && { cp visual_wake_words_model.pt /runpod-volume/; log "model saved to /runpod-volume/"; }

# Keep-alive if requested
if [[ "$keep_alive_enabled" == "true" ]]; then
    log "keep-alive enabled - container will run indefinitely"
    log "files saved to /runpod-volume/ - use 'docker stop' or terminate pod to exit"
    while true; do sleep 300; log "heartbeat"; done
fi

# Exit with training result if no keep-alive
exit $training_exit_code 