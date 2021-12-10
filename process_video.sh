file=${1}
filename="${file%.*}"
folder="data/video_frames/${filename}"
mkdir ${folder}
ffmpeg -i "${file}" "${folder}/%07d.png"
