file=${1}
filename=$(basename ${file%.*})
folder="data/video_frames/${filename}"
mkdir -p ${folder}
ffmpeg -i "${file}" "${folder}/%07d.png"
