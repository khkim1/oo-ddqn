#include <string>
#include <vector>
#include <glog/logging.h>
#include "atari_sim.h"
#include "atari_obj_sim.h"
#include "constants.h"
#include "util.h"

using namespace std;

namespace oodqn {

// 
// AtariObjState
// 

AtariObjState::AtariObjState(const string& snapshot, const Vec& objvec) {
  VLOG(2) << "Creating AtariObjState from snapshot and objvec " << objvec.str();
  snapshot_ = snapshot;
  objvec_ = objvec;
}
AtariObjState::AtariObjState(const string& snapshot,
                             const ale::ALEScreen& screen) {
  snapshot_ = snapshot;
  objvec_ = AleScreenToObjState(screen);
}
void AtariObjState::setSnapshotAndObjvec(const string& snapshot,
                                         const Vec& objvec) {
  snapshot_ = snapshot;
  objvec_ = objvec;
}
bool AtariObjState::equals(const State* rhs) const {
  // if (getType() != rhs->getType())
  //   return false;
  const AtariObjState* other = static_cast<const AtariObjState*>(rhs);
  return snapshot_ == other->snapshot_;
}
bool AtariObjState::equals_planning(const State* const rhs) const {
  // if (getType() != rhs->getType())
  //   return false;
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

//
// AtariObjSim
//

AtariObjSim::AtariObjSim(const std::string& rom_file, int frameskip) :
  frameskip_(frameskip) {
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
       << "\n"
       << endl;
}
AtariObjSim::~AtariObjSim() {
  delete ale_;
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
// const State* AtariObjSim::getPlanningState() const {
//   return current_state_;
// }
void AtariObjSim::setState(const State* new_state) {
  CHECK(current_state_->getType() == new_state->getType())
    << "Mismatched type for AtariObjSim::setState: "
    << current_state_->getType() << " vs " << new_state->getType();
  const AtariObjState* state = dynamic_cast<const AtariObjState*>(new_state);
  *current_state_ = *state;
  ale_->restoreSnapshot(current_state_->getSnapshot());
}
double AtariObjSim::act(const Action* action) {
  float reward = 0.f;
  const AtariAction* atari_action = dynamic_cast<const AtariAction*>(action);
  const ale::Action ale_action = atari_action->getAleAction();
  for (int i = 0; i < frameskip_ && !ale_->gameOver(); ++i) {
    reward += ale_->act(ale_action);
  }
  current_state_->setSnapshotAndObjvec(
      ale_->getSnapshot(),
      AleScreenToObjState(ale_->getScreen()));
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

}  // namespace oodqn
