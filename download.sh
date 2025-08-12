#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_TOKEN=''

huggingface-cli login --token $HUGGINGFACE_TOKEN

MODEL_NAME="microsoft/phi-3-mini-4k-instruct"
LOCAL_DIR="LLM/phi-3-mini-4k-instruct"
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Creating directory: $LOCAL_DIR"
    mkdir -p $LOCAL_DIR
fi

while true; do
    echo "Attempting to download model: $MODEL_NAME"
    
    huggingface-cli download --resume-download $MODEL_NAME \
        --local-dir $LOCAL_DIR --local-dir-use-symlinks False
    
    if [ "$(ls -A $LOCAL_DIR)" ]; then
        echo "Model downloaded successfully?"
    else
        echo "Download interrupted or failed. Retrying in 10 seconds..."
        sleep 10
    fi
done
