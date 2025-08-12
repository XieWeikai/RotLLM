#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

REPO_TYPE="dataset"
REPO_NAME="cimec/lambada"
LOCAL_DIR="Dataset/lambada"
TOKEN=""

while true; do
    echo "尝试下载数据集..."
    huggingface-cli download --repo-type $REPO_TYPE --resume-download $REPO_NAME --local-dir $LOCAL_DIR --token $TOKEN
    
    if [ $? -eq 0 ]; then
        echo "下载成功!"
        break
    else
        echo "下载失败，正在重试..."
        sleep 5
    fi
done