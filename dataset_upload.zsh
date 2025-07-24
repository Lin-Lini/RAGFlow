#!/bin/zsh

url="http://127.0.0.1:8000/upload"
patch="/home/sergey/HACKATON/train_data_mediawise/Media_Digital/"

for i in {0..69}; do
  filename="${i}.pdf"
  
  if [[ -f "$patch$filename" ]]; then
    curl -v -X POST "$url" -F "file=@$patch$filename"
  else
    echo "Файл $patch$filename не найден, пропуск..."
  fi
done
