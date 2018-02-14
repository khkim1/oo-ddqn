#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
// #include <time.h>
// #include <sys/stat.h>

#include "constants.h"
#include "mcts.h"
#include "atari.h"
#include "obj_sim.h"

// test case 1 deterministic property (done)
// test case 2 tree structure (done)
// test case 3 real game play (done)
// test case 4 pseudo death (done)
// test case 5 visualize (done)
// new function: cut branches --> test (done)
// new function: actDiffer --> test (done)

DEFINE_string(rom_path, "", "Game Rom File");
DEFINE_int32(num_traj, 20, "Sample trajectory numbers");
DEFINE_int32(depth, 10, "Planning depth");
DEFINE_int32(num_acts, 6, "Number of actions");
DEFINE_double(ucb, 0.1, "UCB1 coefficient");
DEFINE_double(gamma, 0.99, "Discount factor used in UCT");
DEFINE_string(state_model, "", "Prefix to state model ckpt");
DEFINE_string(reward_model, "", "Prefix to reward model ckpt");
DEFINE_string(plan_sim, "model", "Simulator to use for planning. "
                        "Either \"real\" or \"model\"");
// DEFINE_double(leaf, 0, "Leaf Value");
// DEFINE_bool(save_data, false, "True to save state and action pairs");
// DEFINE_string(save_path, "output", "Path to save training data pairs");

// TODO: Super hardcoded for Pong. What's the best way to
// convert actions?
AtariAction* AtariToObjAction(const ObjectAction* obj_action,
                                         AtariSimulator* atari_sim) {
  // XXX: Correct way to convert action?
  AtariAction* conv_act;
  const int _a = obj_action->act_;
  if (_a != 3 && _a != 4) {
    return new AtariAction(atari_sim->actSet_[-2]);
  } else {
    return new AtariAction(atari_sim->actSet_[_a-2]);
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  srand(time(0));
  cout << "Rom: " << FLAGS_rom_path << endl
       << "Num of traj: " << FLAGS_num_traj << endl
       << "Depth: " << FLAGS_depth << endl
       << "UCB1 coefficient: " << FLAGS_ucb << endl
       << "Number of actions: " << FLAGS_num_acts << endl
       << "Discount (gamma): " << FLAGS_gamma << endl
       << "State model prefix: " << FLAGS_state_model << endl
       << "Reward model prefix: " << FLAGS_reward_model << endl
       << endl;

  AtariSimulator* real_sim = new AtariSimulator(
      FLAGS_rom_path, false, false, 4);
  Simulator* plan_sim; 
  if (FLAGS_plan_sim == "real") {
    // Question: Why different parameters from real_sim?
    plan_sim = new AtariSimulator(FLAGS_rom_path, true, true, 4);
  }
  else {
    plan_sim = new ObjectSimulator(
        FLAGS_state_model, FLAGS_reward_model, true, 4, FLAGS_num_acts);
  }
  MCTSPlanner mcts(plan_sim, FLAGS_depth, FLAGS_num_traj, FLAGS_ucb,
                   FLAGS_gamma);

  int steps = 0;
  bool prev_planned = false;
  SimAction* prev_action = NULL;
  double r = 0;
  double rwd = 0;
  const int max_steps = 20000;
  const bool using_model = (FLAGS_plan_sim == "model");
  Vec obj_state;
  // int data_index = 0;
  if (using_model) {
    obj_state = AleScreenToObjState(real_sim->getScreen());
  }

  while (!real_sim->isTerminal() && steps < max_steps) {
    steps++;
    SimAction* action = NULL;
    // XXX
    if (plan_sim->actDiffer()) {
    // if (real_sim->actDiffer()) {
      if (prev_planned && (!mcts.terminalRoot())) {
        mcts.prune(prev_action, HISTORY_SIZE);
      } else {
        if (using_model) {
          mcts.setRootNode(new ObjectState(obj_state),
                           plan_sim->getActions(),
                           rwd, // XXX: correct?
                           real_sim->isTerminal());
        }
        else {
          mcts.setRootNode(real_sim->getState(),
                           real_sim->getActions(),
                           rwd, // XXX: correct?
                           real_sim->isTerminal());
        }
      }
      mcts.plan();
      action = mcts.getAction();
      prev_planned = true;
      // ++data_index;
      // if(FLAGS_save_data) {
      //   sim->recordData(FLAGS_save_path, data_index, action);
      // }
      
    } else {
      action = real_sim->getRandomAct();
      prev_planned = false;
    }

    if (using_model) {
      ObjectAction* obj_action = dynamic_cast<ObjectAction*>(action);
      AtariAction* conv_act = AtariToObjAction(obj_action, real_sim);
      prev_action = action;
      cout << "step: " << steps << " live: " << real_sim->lives() << " act: ";
      obj_action->print();

      // Observe reward and next state from true env.
      rwd = real_sim->act(conv_act);
      obj_state = AleScreenToObjState(real_sim->getScreen());
      delete conv_act;
    }
    else {
      prev_action = action;
      cout << "step: " << steps << " live: " << real_sim->lives() << " act: ";
      action->print();
      rwd = real_sim->act(action);
    }

    // real_sim->ale_->saveScreenPNG("screen_" + std::to_string(steps));
    r += rwd;
    cout << " rwd: " << rwd << " total: " << r << endl;
  }
  cout <<  "steps: " << steps << "\nr: " << r << endl;
  delete real_sim;
  delete plan_sim;
  return 0;
}
