#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models_dir="${MODELS_DIR:-$repo_root/models}"
github_repo="${GITHUB_REPO:-PDC8/Monocular_Runner_Tracking}"
release_tag="${MODEL_RELEASE_TAG:-models-v1}"
base_url="https://github.com/${github_repo}/releases/download/${release_tag}"

files=(
  "base_best.pt|base_best.pt|ad45194246367488ac2f9207d23cccd8e90362019117466b37fc217cbdbb2822"
  "player_detect_best.pt|player_detect_best.pt|ad369fd8a3a4e528cc79c93a6a99b3243bea5a49b6ce0547ed373b5c1598795e"
  "sam2.1_l.pt|sam2.1_l.pt|ab7e1ac9cb9f6eb3bcf197ece044f06a707ec49129361a2b47e93e1db6989efd"
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

download_file() {
  local url="$1"
  local partial="$2"
  local mode="${3:-resume}"

  if [[ "$mode" == "fresh" ]]; then
    rm -f "$partial"
    curl -fL --retry 3 --retry-delay 2 -o "$partial" "$url"
  else
    curl -fL --retry 3 --retry-delay 2 -C - -o "$partial" "$url"
  fi
}

download_model() {
  local asset_name="$1"
  local local_name="$2"
  local expected_sha="$3"
  local url="${base_url}/${asset_name}"
  local dest="${models_dir}/${local_name}"
  local partial="${dest}.part"
  local downloaded_sha=""
  local legacy_dest="${models_dir}/${asset_name}"

  if [[ "$asset_name" != "$local_name" && -f "$legacy_dest" && ! -f "$dest" ]]; then
    downloaded_sha="$(checksum "$legacy_dest")"
    if [[ "$downloaded_sha" == "$expected_sha" ]]; then
      mv "$legacy_dest" "$dest"
      echo "Renamed existing ${asset_name} to ${local_name}"
      return 0
    fi
  fi

  if [[ -f "$dest" ]]; then
    local current_sha
    current_sha="$(checksum "$dest")"
    if [[ "$current_sha" == "$expected_sha" ]]; then
      echo "Using existing ${local_name}"
      return 0
    fi
    echo "Existing ${local_name} failed checksum verification; downloading a fresh copy."
    rm -f "$dest"
  fi

  if [[ -f "$partial" ]]; then
    downloaded_sha="$(checksum "$partial")"
    if [[ "$downloaded_sha" == "$expected_sha" ]]; then
      mv "$partial" "$dest"
      echo "Recovered ${dest} from an existing partial download"
      return 0
    fi
  fi

  echo "Downloading ${local_name}"
  download_file "$url" "$partial" resume

  downloaded_sha="$(checksum "$partial")"
  if [[ "$downloaded_sha" != "$expected_sha" ]]; then
    echo "Checksum mismatch for ${local_name} after resume; retrying from scratch."
    download_file "$url" "$partial" fresh
    downloaded_sha="$(checksum "$partial")"
    if [[ "$downloaded_sha" != "$expected_sha" ]]; then
      echo "Checksum mismatch for ${local_name}." >&2
      echo "Expected: ${expected_sha}" >&2
      echo "Actual:   ${downloaded_sha}" >&2
      echo "The release asset does not match the checksum baked into download_models.sh." >&2
      exit 1
    fi
  fi

  mv "$partial" "$dest"
  echo "Saved ${dest}"
}

mkdir -p "$models_dir"

for entry in "${files[@]}"; do
  asset_name="${entry%%|*}"
  rest="${entry#*|}"
  local_name="${rest%%|*}"
  sha="${entry##*|}"
  download_model "$asset_name" "$local_name" "$sha"
done

echo "Model download complete."
