#!/bin/bash

INPAINT_URL="https://www.dropbox.com/scl/fi/htifzvxjewg637magznr4/maps_train_inpaint1000.zip?rlkey=n7ju0a1ut4da4voxzp92mq2d1&st=tirhlyj1&dl=1"
ZIP_FILE="maps_train_inpaint.zip"


echo "Downloading $ZIP_FILE..."
if command -v curl >/dev/null 2>&1; then
    curl -L "$INPAINT_URL" -o "$ZIP_FILE"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$ZIP_FILE" "$INPAINT_URL"
else
    echo "Error: Neither curl nor wget found. Install one and try again."
    exit 1
fi

echo "Extracting $ZIP_FILE ..."
unzip -o "$ZIP_FILE" -d "./dataset"

echo "Cleaning up..."
rm -f "$ZIP_FILE"

echo "✅ Done! Files extracted to $DEST_DIR."
