ffmpeg \
  -framerate 20 \
  -i $1%05d.png \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -vf "drawtext=fontfile=/System/Library/Fonts/Monaco.dfont: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099" \
  $1video.mp4
