#include <string>
#include <vector>
#include <glog/logging.h>
#include "atari_sim.h"
#include "atari_obj_sim.h"
#include "constants.h"
#include "util.h"

using namespace std;
namespace tf = tensorflow;

namespace oodqn {

// 
// AtariObjState
// 

AtariObjState::AtariObjState(const string& snapshot, const Vec& objvec) {
  VLOG(2) << "Creating AtariObjState from snapshot and objvec " << objvec.str();
  set_values(snapshot, objvec);
}
AtariObjState::AtariObjState(const string& snapshot,
                             const ale::ALEScreen& screen) {
  set_values(snapshot, AleScreenToObjState(screen));
}
void AtariObjState::setSnapshotAndObjvec(const string& snapshot,
                                         const Vec& objvec) {
  set_values(snapshot, objvec);
}
bool AtariObjState::equals(const State* rhs) const {
  const AtariObjState* other = static_cast<const AtariObjState*>(rhs);
  return (snapshot_ == other->snapshot_);
}
bool AtariObjState::equals_planning(const State* const rhs) const {
  const AtariObjState* other = static_cast<const AtariObjState*>(rhs);
  return (objvec_ == other->objvec_);
}
AtariObjState* AtariObjState::clone() const {
  return new AtariObjState(*this);
}
string AtariObjState::str() const {
  return objvec_.str();
}
void AtariObjState::print() const {
  cout << str() << endl;
}
void AtariObjState::set_values(const std::string& snapshot, const Vec& objvec) {
  snapshot_ = snapshot;
  objvec_ = objvec;
}

//
// AtariObjSim
//

AtariObjSim::AtariObjSim(const std::string& rom_file, int frameskip) :
  frameskip_(frameskip),
  use_reward_model_(false),
  reward_model_(nullptr)
{
  VLOG(2) << "Initializing AtariObjSim with rom " << rom_file;
  ale_ = new ale::ALEInterface(rom_file);
  VLOG(2) << "ALE created";
  current_state_ = new AtariObjState(ale_->getSnapshot(), ale_->getScreen());
  VLOG(2) << "Initial state populated";
  for (const auto& ale_action : ale_->getMinimalActionSet()) {
    actions_.push_back(new AtariAction(ale_action));
  }
  VLOG(2) << "Action list populated";

  // Print information about the created simulator
  cout << "\n >>> AtariObjSim Initialized <<<"
       << "\n  * ROM: " << rom_file
       << "\n  * Min reward: " << ale_->minReward() 
       << "\n  * Max reward: " << ale_->maxReward()
       << "\n  * Available actions: " << stringifyActions(actions_)
       << endl;
}
AtariObjSim::AtariObjSim(const std::string& rom_file, int frameskip,
                         const string& reward_model_prefix) : 
  frameskip_(frameskip),
  use_reward_model_(true)
{
  VLOG(2) << "Initializing AtariObjSim with rom " << rom_file
          << " and reward model " << reward_model_prefix;

  ale_ = new ale::ALEInterface(rom_file);
  VLOG(2) << "ALE created";
  current_state_ = new AtariObjState(ale_->getSnapshot(), ale_->getScreen());
  VLOG(2) << "Initial state populated";
  for (const auto& ale_action : ale_->getMinimalActionSet()) {
    actions_.push_back(new AtariAction(ale_action));
  }
  VLOG(2) << "Action list populated";

  reward_model_ = new TFModel(reward_model_prefix);
  VLOG(2) << "Reward model loaded";

  // Print information about the created simulator
  cout << "\n >>> AtariObjSim Initialized <<<"
       << "\n  * ROM: " << rom_file
       << "\n  * Min reward: " << ale_->minReward() 
       << "\n  * Max reward: " << ale_->maxReward()
       << "\n  * Reward model: " << reward_model_->getModelPrefix()
       << "\n  * Available actions: " << stringifyActions(actions_)
       << endl;
}
AtariObjSim::~AtariObjSim() {
  delete ale_;
  if (use_reward_model_) {
    delete reward_model_;
  }
}
ale::ALEScreen AtariObjSim::getScreen() const {
  return ale_->getScreen();
}
void AtariObjSim::screenToPNG(const string& filename) const {
  if (!ale_->screenToPNG(filename)) {
    LOG(ERROR) << "Failed to save screen to " << filename << endl;
  }
}
const State* AtariObjSim::getState() const {
  return current_state_;
}
void AtariObjSim::setState(const State* new_state) {
  CHECK(current_state_->getType() == new_state->getType())
    << "Mismatched type for AtariObjSim::setState: "
    << current_state_->getType() << " vs " << new_state->getType();
  const AtariObjState* state = dynamic_cast<const AtariObjState*>(new_state);
  *current_state_ = *state;
  ale_->restoreSnapshot(current_state_->getSnapshot());
}
double AtariObjSim::act(const Action* action) {
  VLOG(5) << "Taking action: " << action->str();
  float reward = 0.f;
  const AtariAction* atari_action = dynamic_cast<const AtariAction*>(action);
  const ale::Action ale_action = atari_action->getAleAction();
  for (int i = 0; i < frameskip_ && !ale_->gameOver(); ++i) {
    if (use_reward_model_) {
      VLOG(5) << "Getting reward from a trained model";
      ale_->act(ale_action);
      reward += predictReward(AleScreenToObjState(ale_->getScreen()));
    }
    else {
      VLOG(5) << "Getting reward from ALE";
      reward += ale_->act(ale_action);
    }
  }
  current_state_->setSnapshotAndObjvec(
      ale_->getSnapshot(), AleScreenToObjState(ale_->getScreen()));
  // Clip reward if using reward model.
  if (use_reward_model_) {
    if (reward > 0) reward = 1;
    if (reward < 0) reward = -1;
  }
  return reward;
}
void AtariObjSim::reset() {
  ale_->resetGame();
  current_state_->setSnapshotAndObjvec(
      ale_->getSnapshot(),
      AleScreenToObjState(ale_->getScreen()));
}
bool AtariObjSim::isTerminal() const {
  return ale_->gameOver();
}
const std::vector<const Action*>& AtariObjSim::getActions() const {
  return actions_;
}
bool AtariObjSim::actionIgnored() const {
  const std::string cur_ss = current_state_->getSnapshot();
  std::string prev_ss = "";
  for (const Action* _a : actions_) {
    const AtariAction* action = static_cast<const AtariAction*>(_a);
    ale_->restoreSnapshot(cur_ss);
    for (int j = 0; j < frameskip_ && !ale_->gameOver(); ++j) {
      ale_->act(action->getAleAction());
    }
    const std::string ss = ale_->getSnapshot();
    if ((prev_ss != "") && (prev_ss != ss)) {
      ale_->restoreSnapshot(cur_ss);
      return false;
    }
    prev_ss = ss;
  }
  ale_->restoreSnapshot(cur_ss);
  return true;
}
void AtariObjSim::saveFrame(const string& filename) const {
  if (!ale_->screenToPNG(filename)) {
    LOG(ERROR) << "Failed to save screen to " << filename;
  }
}
// XXX: Hardcoded for Pong!
double AtariObjSim::predictReward(const Vec& objvec) {
  tf::Tensor out;
  reward_model_->RunMatrix(objvec, tf::TensorShape({1, 9}),
                           kInputPlaceholderReward, kRewardTensor, &out);
  float neg, zero, pos, reward;
  neg = out.matrix<float>()(0, 0);
  zero = out.matrix<float>()(0, 1);
  pos = out.matrix<float>()(0, 2);

  if (neg > zero && neg > pos)
    reward = -1;
  else if (pos > neg && pos > zero)
    reward = 1;
  else
    reward = 0;

  VLOG(5) << "Predicted reward: " << reward;
  return reward;
}

}  // namespace oodqn
