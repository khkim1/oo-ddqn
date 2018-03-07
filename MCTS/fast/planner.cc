#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <string>
#include <vector>

#include "constants.h"
#include "mcts.h"
#include "atari_sim.h"
#include "atari_obj_sim.h"

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
DEFINE_bool(obj_state, false,
            "Whether to use object state representation or not");
DEFINE_string(frame_prefix, "", "Prefix for saved screen frames in PNG.");
DEFINE_string(rollout_prefix, "", "Prefix for saved rollout screen frames.");
DEFINE_int32(frameskip, 4, "Frame skip");

using namespace std;
using namespace oodqn;

// TODO: Super hardcoded for Pong. What's the best way to
// convert actions?
/*
AtariAction* ObjToAtariAction(const ObjectAction* obj_action,
                              AtariSimulator* atari_sim) {
  AtariAction* conv_act;
  const int _a = obj_action->act_;
  if (_a == 0) {
    return new AtariAction(atari_sim->actSet_[0]);
  } else if (_a == 3 || _a == 4) {
    return new AtariAction(atari_sim->actSet_[_a-2]);
  } else {
		int idx = rand() % atari_sim->actVect_.size();
    return new AtariAction(atari_sim->actSet_[idx]);
  }
}
*/

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  srand(time(0));
  const bool save_frame = (FLAGS_frame_prefix != "");
  const bool save_rollout = (FLAGS_rollout_prefix != "");
  char frame_fn[80];

  cout << "\n >>> Configuration <<<"
       << "\n  * Rom: " << FLAGS_rom_path
       << "\n  * Num of traj: " << FLAGS_num_traj
       << "\n  * Depth: " << FLAGS_depth
       << "\n  * UCB1 coefficient: " << FLAGS_ucb
       << "\n  * Number of actions: " << FLAGS_num_acts
       << "\n  * Discount (gamma): " << FLAGS_gamma
       << "\n  * State model prefix: " << FLAGS_state_model
       << "\n  * Reward model prefix: " << FLAGS_reward_model
       << "\n  * State representation: " << (FLAGS_obj_state ? "object" : "raw")
       << "\n  * Saving screen frames: "
       << (save_frame ? FLAGS_frame_prefix : "no")
       << "\n  * Saving rollout frames: "
       << (save_rollout ? FLAGS_rollout_prefix : "no")
       << endl;

  Simulator *real_sim, *plan_sim; 
  if (FLAGS_obj_state) {
    real_sim = new AtariObjSim(FLAGS_rom_path, FLAGS_frameskip);
  } else {
    real_sim = new AtariSim(FLAGS_rom_path, FLAGS_frameskip);
  }

  if (FLAGS_plan_sim == "real") {
    if (FLAGS_obj_state) {
      plan_sim = new AtariObjSim(FLAGS_rom_path, FLAGS_frameskip);
    } else {
      plan_sim = new AtariSim(FLAGS_rom_path, FLAGS_frameskip);
    }
  }
  else {
    // If we're using trained model of the world, at least the reward model
    // should be given.
    CHECK(FLAGS_reward_model != "") << "Reward model not specified.";

    // We're only using the reward model with the true transition model.
    if (FLAGS_state_model == "") {
      plan_sim = new AtariObjSim(FLAGS_rom_path, FLAGS_frameskip,
                                 FLAGS_reward_model);
    }
    // else {
    //   plan_sim = new ObjectSimulator(
    //       FLAGS_state_model, FLAGS_reward_model, true, FLAGS_frameskip,
    //       FLAGS_num_acts, AleScreenToObjState(real_sim->getScreen()));
    // }
  }
  MCTSPlanner mcts(plan_sim, FLAGS_depth, FLAGS_num_traj, FLAGS_ucb,
                   FLAGS_gamma);
  if (save_rollout) {
    mcts.setRolloutPrefix(FLAGS_rollout_prefix);
  }

  int step = 0;
  bool prev_planned = false;
  Action* prev_action = NULL;
  double ret = 0;
  double rwd = 0;
  const int max_steps = 20000;
  const bool using_state_model =
    (FLAGS_plan_sim == "model" && FLAGS_state_model != "");
  Vec obj_state;
  // int data_index = 0;

  /*
  if (using_state_model) {
    // Initial object state vector
    obj_state = AleScreenToObjState(real_sim->getScreen());

    // Skip loading frames
    SimAction* tmp_act;
    while (obj_state[2] == 0. ||
           obj_state[5] == 0. ||
           obj_state[8] == 0.) {
      tmp_act = real_sim->getRandomAct();
      real_sim->act(tmp_act);
      obj_state = AleScreenToObjState(real_sim->getScreen());
      cout << "[dbg] Skipping frame by taking random action..." << endl;
    }
  }
  */

  cout << "\n *** Starting planning *** \n" << endl;
  while (!real_sim->isTerminal() && step < max_steps) {
    step++;
    Action* action = nullptr;

    // With learned dynamics model
    if (using_state_model) {
      /*
      if (plan_sim->actDiffer()) {
        if (prev_planned && (!mcts.terminalRoot())) {
          cout << "[dbg] Real step" << endl;
          ObjectState* newStateFromTrueEnv = new ObjectState(obj_state);
          mcts.realStep(prev_action, newStateFromTrueEnv, rwd, real_sim->isTerminal());
          // vector<State*> hist = mcts.root_->stateHistory(12);
          // for (int i = 0; i < hist.size(); i++) {
          //   cout << "[dbg] State Hist " << i << ": ";
          //   hist[i]->print();
          //   cout << endl;
          // }
        } else {
          cout << "[dbg] Set new root" << endl;
          mcts.setRootNode(new ObjectState(obj_state),
                           plan_sim->getActions(),
                           // Question: correct to use real reward here?
                           rwd,
                           // Question: What about isTermianl from real_sim?
                           real_sim->isTerminal());
        }
        mcts.plan();
        action = mcts.getAction();
        prev_planned = true;
        cout << "[dbg] Taking planned action" << endl;
      } else {
        cout << "[dbg] Taking random action" << endl;
        action = real_sim->getRandomAct();
        prev_planned = false;
      }

      ObjectAction* obj_action = dynamic_cast<ObjectAction*>(action);
      AtariAction* conv_act = ObjToAtariAction(obj_action, real_sim);
      prev_action = action;
      cout << "\nstep: " << step << " live: " << real_sim->lives() << " act: ";
      conv_act->print();

      // Observe reward and next state from true env.
      rwd = real_sim->act(conv_act);
      obj_state = AleScreenToObjState(real_sim->getScreen());
      delete conv_act;
      */
    }

    // With real simulator
    else {
      // XXX
      if (real_sim->actionIgnored()) {
        LOG(INFO) << "Taking random action";
        action = real_sim->getRandomAction();
        prev_planned = false;
      } else {
        LOG(INFO) << "Taking planned action";
        if (prev_planned && (!mcts.terminalRoot())) {
          mcts.prune(prev_action, -1);
        } else {
          mcts.setRootNode(real_sim->getState(), real_sim->getActions(),
                           rwd, real_sim->isTerminal());
        }
        if (save_rollout)
          mcts.plan(step);
        else
          mcts.plan(-1);
        action = mcts.getAction();
        prev_planned = true;
      }

      prev_action = action;
      rwd = real_sim->act(action);

      ret += rwd;
      cout << "step: " << step << " act: " << action->str()
           << " rwd: " << rwd << " total: " << ret << endl;
    }

    // Save screen if flag is true.
    if (save_frame) {
      sprintf(frame_fn, "%s%05d.png", FLAGS_frame_prefix.c_str(), step);
      real_sim->saveFrame(frame_fn);
    }
  }
  cout << "Final reward: " << rwd <<" total: " << ret << endl;

  cout <<  "step: " << step << "\nr: " << ret << endl;
  delete real_sim;
  delete plan_sim;
  return 0;
}
