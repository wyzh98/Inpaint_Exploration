#!/bin/bash

INPAINT_URL="https://www.dropbox.com/scl/fi/hwknf4yhub6zw3t3fzggz/maps_train_inpaint1000.zip?rlkey=tv5dsrlgpfvb7yz7ufetnsbh4&st=3awdhbgq&dl=1"
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
