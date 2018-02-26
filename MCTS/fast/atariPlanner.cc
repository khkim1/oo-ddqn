#include "atari_sim.h"
#include "mcts.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
//#define STRIP_FLAG_HELP 1 
#include <time.h>
#include <sys/stat.h>

DEFINE_string(rom_path, "", "Game Rom File");
DEFINE_int32(num_traj, 20, "Sample trajectory numbers");
DEFINE_int32(depth, 10, "Planning depth");
DEFINE_double(ucb, 0.1, "Planning scalar");
DEFINE_double(leaf, 0, "Leaf Value");
DEFINE_double(gamma, 0.99, "Discount factor used in UCT");
DEFINE_bool(save_data, false, "True to save state and action pairs");
DEFINE_string(save_path, "output", "Path to save training data pairs");

using namespace std;
using namespace oodqn;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  srand(time(0));
  cout << "Rom: " << FLAGS_rom_path << endl;
  cout << "Num of traj: " << FLAGS_num_traj << endl;
  cout << "Depth: " << FLAGS_depth << endl;
  cout << "UCB: " << FLAGS_ucb << endl;
  cout << "Leaf: " << FLAGS_leaf << endl;
  string rom_path(FLAGS_rom_path);
  AtariSim* sim = new AtariSim(rom_path, 4);
  AtariSim* sim_in_plan = new AtariSim(rom_path, 4);
  MCTSPlanner mcts(
      sim_in_plan, FLAGS_depth, FLAGS_num_traj, FLAGS_ucb, FLAGS_gamma);
    
  int steps = 0;
  bool prev_planned = false;
  Action* prev_action = NULL;
  double r = 0;
  double rwd ;
  int data_index = 0;
  const int max_steps = 20000;
  while (!sim->isTerminal() && steps < max_steps) {
    steps++;
    Action* action = NULL;
    if (!sim->actionIgnored()) {
      LOG(INFO) << "Taking planned action";
      if (prev_planned && (!mcts.terminalRoot())) {
        mcts.prune(prev_action, -1);
      } else {
        mcts.setRootNode(sim->getState(), sim->getActions(), rwd, sim->isTerminal());
      }
      mcts.plan();
      action = mcts.getAction();
      // LOG(INFO) << "Chosen action: " << action->str();
      prev_planned = true;
    } else {
      LOG(INFO) << "Taking random action";
      action = sim->getRandomAction();
      prev_planned = false;
    }

    prev_action = action;
    cout << "step: " << steps << " act: " << action->str();

    rwd = sim->act(action);
    r += rwd;
    cout << " rwd: " << rwd << " total: " << r << endl;
    // << " time: " << (total_time / (time_count + 1e-8)) << endl;
  }
  cout <<  "steps: " << steps << "\nr: " << r << endl;
  delete sim;
  delete sim_in_plan;
  return 0;
}
