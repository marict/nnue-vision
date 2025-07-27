#!/usr/bin/env bash
# Persist-safe NNUE-Vision setup and training entry-point
# Saves logs and checkpoints in /runpod-volume so they survive POD restarts.

set -xe            # prints every command before it runs
mountpoint -q /runpod-volume || echo "/runpod-volume not mounted"

set -euo pipefail
exec 2>&1                     # merge stderr into stdout

# Initialize timing and define logging functions first
start_time=$(date +%s)

log()   { printf '[%6ss] %s\n'  "$(( $(date +%s) - start_time ))" "$*"; }
err()   { log "ERROR: $*" >&2; }
trap 'err "failed at line $LINENO"' ERR

# Add more verbose logging
log "Container started - checking environment"
log "Python version: $(python --version)"
log "Pip version: $(pip --version)"
log "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'torch not available yet')"

#---------------------------------------------------------------------------#
# workspace & repo
#---------------------------------------------------------------------------#
cd /workspace/repo
log "cwd $(pwd)"

# Repository should already be cloned by the calling script
if [[ ! -d .git ]]; then
    err "Repository not found. Expected to be in /workspace/repo"
    exit 1
fi

#---------------------------------------------------------------------------#
# system pkgs (guard against readonly images)
#---------------------------------------------------------------------------#
# Robust apt update (NVIDIA repo occasionally has sync issues)
log "updating apt repositories"

# Proactive cleanup of problematic NVIDIA repositories before first attempt
log "cleaning up NVIDIA repository sources that may cause conflicts"
find /etc/apt/sources.list.d -type f -iname "*nvidia*" -delete || true
find /etc/apt/sources.list.d -type f -iname "*cuda*" -delete || true
sed -i '/nvidia/d' /etc/apt/sources.list || true
sed -i '/cuda/d' /etc/apt/sources.list || true

# Clear apt cache to ensure fresh repository data
rm -rf /var/lib/apt/lists/* || true

if ! apt-get update; then
    log "apt-get update failed even after NVIDIA cleanup – performing additional cleanup"
    
    # More aggressive cleanup
    find /etc/apt -name "*.list*" -exec grep -l "nvidia\|cuda" {} \; -delete || true
    
    # Remove any broken or incomplete package lists
    rm -rf /var/lib/apt/lists/partial/* || true
    
    # Try to fix any broken packages
    dpkg --configure -a || true
    
    # Final retry (ignore failure so script continues)
    apt-get update || true
fi

# Install useful CLI tools (best-effort – do not abort on failure)
apt-get install -y --no-install-recommends tree htop || true

#---------------------------------------------------------------------------#
# python deps
#---------------------------------------------------------------------------#
log "installing python deps"
if ! pip install -q -r requirements-dev.txt; then
    log "ERROR: pip install failed - attempting pip upgrade and retry"
    pip install --upgrade pip
    if ! pip install -r requirements-dev.txt; then
        log "ERROR: pip install failed even after upgrade"
        exit 1
    fi
fi
log "python dependencies installed successfully"

#---------------------------------------------------------------------------#
# debugging env vars
#---------------------------------------------------------------------------#
export CUDA_LAUNCH_BLOCKING=1  # helpful for catching async CUDA errors
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # help with GPU memory fragmentation

#---------------------------------------------------------------------------#
# Set up logging directory
#---------------------------------------------------------------------------#
log_dir="/runpod-volume/nnue-vision-logs"
mkdir -p "$log_dir"

# Symlink lightning_logs to persistent storage
if [[ ! -L lightning_logs ]]; then
    rm -rf lightning_logs 2>/dev/null || true
    ln -sf "$log_dir" lightning_logs
fi

# Create checkpoints directory
mkdir -p "$log_dir/checkpoints"

#---------------------------------------------------------------------------#
# training
#---------------------------------------------------------------------------#
log_file="/runpod-volume/train_$(date +%Y%m%d_%H%M%S).log"
log "starting NNUE-Vision training – output -> $log_file"

# Check if keep-alive flag was passed (before we remove it)
keep_alive_enabled=false
if [[ "$*" == *"--keep-alive"* ]]; then
    keep_alive_enabled=true
fi

# Remove --keep-alive flag from arguments since train.py doesn't recognize it
filtered_args=()
for arg in "$@"; do
    if [[ "$arg" != "--keep-alive" ]]; then
        filtered_args+=("$arg")
    fi
done

# Update log directory argument if not already specified
if [[ "${filtered_args[*]}" != *"--log_dir"* ]]; then
    filtered_args+=("--log_dir=$log_dir")
fi

# Ensure WANDB_API_KEY is set before training
if [[ -z "$WANDB_API_KEY" ]]; then
    log "ERROR: WANDB_API_KEY environment variable is not set"
    exit 1
fi

log "starting python training with args: ${filtered_args[*]}"
python -u "${filtered_args[@]}" 2>&1 | tee "$log_file"

log "training completed in $(( $(date +%s)-start_time ))s"

# Save final model to persistent storage
if [[ -f visual_wake_words_model.pt ]]; then
    cp visual_wake_words_model.pt /runpod-volume/
    log "copied final model to /runpod-volume/"
fi

# Apply keep-alive behavior if flag was passed
if [[ "$keep_alive_enabled" == "true" ]]; then
    log "keep-alive mode enabled – keeping container alive"
    log "final model and logs saved to /runpod-volume/"
    log "to access files: ls -la /runpod-volume/"
    tail -f /dev/null
fi 