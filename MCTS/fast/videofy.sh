ffmpeg \
  -framerate 10 \
  -i $1%04d.png \
  -c:v libx264 \
  -pix_fmt yuv420p \
  $1video.mp4
