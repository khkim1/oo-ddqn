ffmpeg \
  -framerate 10 \
  -i sim_screen_%04d.png \
  -c:v libx264 \
  -pix_fmt yuv420p \
  video.mp4
