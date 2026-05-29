#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
DATA_DIR="${MNIST_DATA_DIR:-data/MNIST/raw}"

FILES=(
  "train-images-idx3-ubyte"
  "train-labels-idx1-ubyte"
  "t10k-images-idx3-ubyte"
  "t10k-labels-idx1-ubyte"
)

download() {
  local url="$1"
  local destination="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --output "$destination" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$destination" "$url"
  else
    echo "Error: curl or wget is required to download MNIST." >&2
    exit 1
  fi
}

mkdir -p "$DATA_DIR"

for file in "${FILES[@]}"; do
  destination="$DATA_DIR/$file"
  if [[ -f "$destination" ]]; then
    echo "$destination already exists; skipping."
    continue
  fi

  gz_file="$destination.gz"
  if [[ ! -f "$gz_file" ]]; then
    echo "Downloading $gz_file..."
    download "$BASE_URL/$file.gz" "$gz_file"
  fi

  echo "Decompressing $gz_file..."
  gzip -dk "$gz_file"
done

echo "MNIST IDX files are ready in $DATA_DIR."
