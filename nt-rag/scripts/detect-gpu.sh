#!/usr/bin/env bash
# Detect NVIDIA GPU availability for Docker (Windows Docker Desktop / Linux).
# Source this file: source scripts/detect-gpu.sh

detect_nvidia_gpu_for_docker() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  if docker info 2>/dev/null | grep -qi nvidia; then
    return 0
  fi
  if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi \
    >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

should_use_gpu_compose() {
  local mode="${OLLAMA_GPU_MODE:-auto}"
  case "$mode" in
    cpu)
      return 1
      ;;
    gpu)
      if detect_nvidia_gpu_for_docker; then
        return 0
      fi
      echo "ERROR: OLLAMA_GPU_MODE=gpu but no NVIDIA GPU is usable by Docker." >&2
      echo "  Enable GPU in Docker Desktop (WSL2) and install NVIDIA drivers." >&2
      return 2
      ;;
    auto)
      detect_nvidia_gpu_for_docker
      return $?
      ;;
    *)
      echo "WARNING: Unknown OLLAMA_GPU_MODE='$mode' (use auto|gpu|cpu). Falling back to CPU." >&2
      return 1
      ;;
  esac
}

# Prints compose -f arguments (one per line); use with mapfile. Exit 2 if gpu forced but unavailable.
compose_file_args() {
  echo "-f"
  echo "docker-compose.yml"
  should_use_gpu_compose
  local s=$?
  if [[ $s -eq 2 ]]; then
    return 2
  fi
  if [[ $s -eq 0 ]]; then
    echo "-f"
    echo "docker-compose.gpu.yml"
  fi
}
