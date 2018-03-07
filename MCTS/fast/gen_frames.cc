#include <gflags/gflags.h>
#include <iostream>
#include <stdio.h>

#include "constants.h"
#include "atari_sim.h"
#include "util.h"

DEFINE_string(rom_path, "", "Game Rom File");
DEFINE_string(frame_prefix, "", "Prefix for saved screen frames in PNG.");
DEFINE_int32(frameskip, 1, "Frame skip");

using namespace std;
using namespace oodqn;

void log(AtariSim* sim, const string& prefix, int step) {
  char frame_fn[80];
  sprintf(frame_fn, "%s%04d.png", prefix.c_str(), step);
  sim->saveFrame(frame_fn);
  cout << frame_fn << " ";

  const Vec state = AleScreenToObjState(sim->getScreen());
  for (int i = 0; i < state.size(); ++i) {
    if (i > 0) cout << ",";
    cout << std::to_string(state[i]);
  }

  cout << endl;
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  srand(time(0));
  const bool save_frame = (FLAGS_frame_prefix != "");

  AtariSim* sim = new AtariSim(FLAGS_rom_path, FLAGS_frameskip);
  if (save_frame) log(sim, FLAGS_frame_prefix, 0);

  double reward;
  Action* action = nullptr;
  for (int step = 1; step < 300; ++step) {
    action = sim->getRandomAction();
    reward = sim->act(action);

    // Save screen if flag is true.
    if (save_frame) log(sim, FLAGS_frame_prefix, step);
  }

  delete sim;
  return 0;
}
