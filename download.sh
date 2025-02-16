#!/bin/bash

# 定义要替换的 X 值
values=(0 100 200 300 400 500 600 700 800 900)

# 最大重试次数
MAX_RETRIES=3

# 下载函数，支持重试
download_with_retry() {
    local url=$1
    local output_file=$2
    local retries=0

    while [[ $retries -lt $MAX_RETRIES ]]; do
        echo "Downloading: $url -> $output_file (Attempt $((retries + 1))/$MAX_RETRIES)"
        aria2c -x 16 -s 16 -k 1M "$url" -o "$output_file"
        if [[ $? -eq 0 ]]; then
            echo "Download successful: $output_file"
            return 0
        else
            retries=$((retries + 1))
            echo "Download failed, retrying in 2 seconds..."
            sleep 2  # 等待 2 秒后重试
        fi
    done

    echo "Download failed after $MAX_RETRIES attempts: $output_file"
    return 1
}

# 解压函数，支持重试
extract_with_retry() {
    local file=$1
    local retries=0

    while [[ $retries -lt $MAX_RETRIES ]]; do
        echo "Extracting: $file (Attempt $((retries + 1))/$MAX_RETRIES)"
        tar -xvf "$file"
        if [[ $? -eq 0 ]]; then
            echo "Extraction successful: $file"
            return 0
        else
            retries=$((retries + 1))
            echo "Extraction failed, retrying in 2 seconds..."
            sleep 2  # 等待 2 秒后重试
        fi
    done

    echo "Extraction failed after $MAX_RETRIES attempts: $file"
    return 1
}

# 遍历每个值
for X in "${values[@]}"; do

    # 下载并解压 ObjectFolder[X+1]-[X+100]KiloOSF.tar.gz
    download_with_retry "https://download.cs.stanford.edu/viscam/ObjectFolder/ObjectFolder$((X+1))-$((X+100)).tar.gz" "ObjectFolder$((X+1))-$((X+100)).tar.gz"
    if [[ $? -eq 0 ]]; then
        extract_with_retry "ObjectFolder$((X+1))-$((X+100)).tar.gz"
    fi
done

