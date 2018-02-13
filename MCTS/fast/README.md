## How to build

1. Clone the repository
1. Download platform-specific Tensorflow library files 
   (`libtensorflow_cc.so`, `libtensorflow_framework.so`)
   from Team Drive to tf{linux,mac}/lib
1. make -f Makefile.{linux,mac} <target>

If the compiler complains about missing library, you may need to
create a symlink to libtensorflow\* files in the current folder.

## How to run

```
./atariUCTPlanner -rom_path=../atari_roms/pong.bin -num_traj=100 -depth=100
```

```
./objectPlanner -rom_path=../atari_roms/pong.bin \
  -state_model=../../DQN-model/saves/trained_Q/1000/episode_1000.ckpt \
  -reward_model=../reward_model/reward_model -num_traj=100 -depth=100
```
