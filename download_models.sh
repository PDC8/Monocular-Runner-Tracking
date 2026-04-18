#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models_dir="${MODELS_DIR:-$repo_root/models}"
github_repo="${GITHUB_REPO:-PDC8/Monocular_Runner_Tracking}"
release_tag="${MODEL_RELEASE_TAG:-models-v1}"
base_url="https://github.com/${github_repo}/releases/download/${release_tag}"

files=(
  "base_best.pt:ad45194246367488ac2f9207d23cccd8e90362019117466b37fc217cbdbb2822"
  "player_detect_yolov8l_best.pt:ad369fd8a3a4e528cc79c93a6a99b3243bea5a49b6ce0547ed373b5c1598795e"
  "sam2.1_l.pt:ab7e1ac9cb9f6eb3bcf197ece044f06a707ec49129361a2b47e93e1db6989efd"
)

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download model assets." >&2
  exit 1
fi

if command -v shasum >/dev/null 2>&1; then
  hash_cmd=(shasum -a 256)
elif command -v sha256sum >/dev/null 2>&1; then
  hash_cmd=(sha256sum)
else
  echo "Either shasum or sha256sum is required to verify model assets." >&2
  exit 1
fi

checksum() {
  "${hash_cmd[@]}" "$1" | awk '{print $1}'
}

download_model() {
  local name="$1"
  local expected_sha="$2"
  local url="${base_url}/${name}"
  local dest="${models_dir}/${name}"
  local partial="${dest}.part"

  if [[ -f "$dest" ]]; then
    local current_sha
    current_sha="$(checksum "$dest")"
    if [[ "$current_sha" == "$expected_sha" ]]; then
      echo "Using existing ${name}"
      return 0
    fi
    echo "Existing ${name} failed checksum verification; downloading a fresh copy."
  fi

  echo "Downloading ${name}"
  curl -fL --retry 3 --retry-delay 2 -C - -o "$partial" "$url"

  local downloaded_sha
  downloaded_sha="$(checksum "$partial")"
  if [[ "$downloaded_sha" != "$expected_sha" ]]; then
    echo "Checksum mismatch for ${name}." >&2
    echo "Expected: ${expected_sha}" >&2
    echo "Actual:   ${downloaded_sha}" >&2
    exit 1
  fi

  mv "$partial" "$dest"
  echo "Saved ${dest}"
}

mkdir -p "$models_dir"

for entry in "${files[@]}"; do
  name="${entry%%:*}"
  sha="${entry#*:}"
  download_model "$name" "$sha"
done

echo "Model download complete."
