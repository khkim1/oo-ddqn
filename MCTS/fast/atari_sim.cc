#include <glog/logging.h>
#include <vector>
#include <string>
#include "atari_sim.h"
// #include "mcts.h"
// #include <ale_interface.hpp>
#include "constants.h"

using namespace std;

string stringifyActions(const vector<const oodqn::Action*>& actions) {
  string out = "";
  if (actions.size() > 0) {
    out += actions[0]->str();
    for (int i = 1; i < actions.size(); ++i) {
      out += ", " + actions[i]->str();
    }
  }

  return ("[" + out + "]");
}

namespace oodqn {

// 
// AtariState
// 

AtariState::AtariState(const string& snapshot) {
  snapshot_ = snapshot;
}
void AtariState::setSnapshot(const string& snapshot) {
  snapshot_ = snapshot;
}
string AtariState::getSnapshot() const {
  return snapshot_;
}
bool AtariState::equals(const State* rhs) const {
  const AtariState* other = static_cast<const AtariState*>(rhs);
  return snapshot_ == other->snapshot_;
}
bool AtariState::equals_planning(const State* rhs) const {
  return equals(rhs);
}
AtariState* AtariState::clone() const {
  return new AtariState(*this);
}
string AtariState::str() const {
  return snapshot_;
}
void AtariState::print() const {
  cout << str() << endl;
}
string AtariState::getType() const {
  return "AtariState";
}

// 
// AtariAction
// 

AtariAction::AtariAction(const ale::Action action) {
  action_ = action;
}
bool AtariAction::equals(const Action* rhs) const {
  const AtariAction* other = static_cast<const AtariAction*>(rhs);
  return action_ == other->action_;
}
Action* AtariAction::clone() const {
  return new AtariAction(*this);
}
std::string AtariAction::str() const {
  return ALE_ACTION_NAMES[action_];
}
void AtariAction::print() const {
  cout << str() << endl;
}

//
// AtariSim
//

AtariSim::AtariSim(const std::string& rom_file, int frameskip) :
  frameskip_(frameskip) {
  ale_ = new ale::ALEInterface(rom_file);
  current_state_ = new AtariState(ale_->getSnapshot());
  for (const auto& ale_action : ale_->getMinimalActionSet()) {
    actions_.push_back(new AtariAction(ale_action));
  }

  // Print information about the created simulator
  cout << "\n >>> AtariSim Initialized <<<"
       << "\n  * ROM: " << rom_file
       << "\n  * Min reward: " << ale_->minReward() 
       << "\n  * Max reward: " << ale_->maxReward()
       << "\n  * Available actions: " << stringifyActions(actions_)
       << "\n"
       << endl;
}
AtariSim::~AtariSim() {
  delete ale_;
}
ale::ALEScreen AtariSim::getScreen() const {
  return ale_->getScreen();
}
void AtariSim::screenToPNG(const string& filename) const {
  if (!ale_->screenToPNG(filename)) {
    LOG(ERROR) << "Failed to save screen to " << filename << endl;
  }
}
const State* AtariSim::getState() const {
  return current_state_;
}
// const State* AtariSim::getPlanningState() const {
//   return current_state_;
// }
void AtariSim::setState(const State* new_state) {
  // XXX: ???
  const AtariState* state = dynamic_cast<const AtariState*>(new_state);
  current_state_->setSnapshot(state->getSnapshot());
  ale_->restoreSnapshot(current_state_->getSnapshot());
}
double AtariSim::act(const Action* action) {
  float reward = 0.f;
  const AtariAction* atari_action = dynamic_cast<const AtariAction*>(action);
  const ale::Action ale_action = atari_action->getAleAction();
  for (int i = 0; i < frameskip_ && !ale_->gameOver(); ++i) {
    reward += ale_->act(ale_action);
  }
  current_state_->setSnapshot(ale_->getSnapshot());
  return reward;
}
void AtariSim::reset() {
  ale_->resetGame();
  current_state_->setSnapshot(ale_->getSnapshot());
}
bool AtariSim::isTerminal() const {
  return ale_->gameOver();
}
const std::vector<const Action*>& AtariSim::getActions() const {
  return actions_;
}
bool AtariSim::actionIgnored() const {
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
void AtariSim::saveFrame(const string& filename) const {
  if (!ale_->screenToPNG(filename)) {
    LOG(ERROR) << "Failed to save screen to " << filename;
  }
}


/*
class AtariSimulator: public Simulator {
public:

	AtariSimulator (
      const std::string& romFile,
      // bool updateFrame,
      bool pseudoGameover,
      bool scaleReward,
      int numRepeats,
      bool useObjectState
      ):
		// updateFrame_(updateFrame),
		pseudoGameover_(pseudoGameover), 
		scaleReward_(scaleReward), 
		numRepeats_(numRepeats), 
    useObjectState_(useObjectState),
		reward_(0) {

		ale_ = new ale::ALEInterface(romFile);
		currentState_ = new AtariState(ale_->getSnapshot());
		actSet_ = ale_->getMinimalActionSet();
		minReward_ = ale_->minReward();
		maxReward_ = ale_->maxReward();
		// frameBuffer_ = new FrameBuffer(4);
		lifeLost_ = false;

		for (int i = 0 ; i < actSet_.size(); ++i) {
			actVect_.push_back(new AtariAction(actSet_[i]));
			// act_map[actSet_[i]] = i;
		}

		cout << "Simulator Information: "
		     << "\nROM: " << romFile
		     << "\nPseudo GameOver: " << (pseudoGameover_ ? "True" : "False")
		     << "\nScale Reward: " << (scaleReward_ ? "True" : "False" )
		     << "\nMax R: " << maxReward_ << "  Min R: " << minReward_
		     << "\nAct Size: " << actSet_.size() << "[";
		for (int i = 0; i < actSet_.size(); ++i) {
			cout << actSet_[i] << ", ";
		}
		cout << "]\n" << endl;
	}

	virtual ~AtariSimulator () {
		delete ale_;
		delete currentState_;
		for (int i = 0; i < actVect_.size(); ++i) {
			delete actVect_[i];
		}
		// delete frameBuffer_;
	}

	int lives() {
		return ale_->lives();
	}

  ale::ALEScreen getScreen() {
    return ale_->getScreen();
  }

  void screenToPNG(const string& filename) {
    if (!ale_->screenToPNG(filename)) {
      cout << "Failed to save screen to " << filename << endl;
    }
  }

  virtual void setTerminal(bool) {}

	virtual void setState(State* state) {
		const AtariState* other = dynamic_cast<const AtariState*> (state);
		currentState_->snapshot_ = other->snapshot_;
		ale_->restoreSnapshot(currentState_->snapshot_);
		lifeLost_ = false;
	}

  virtual void setHistory(vector<State*> history) {
    // Nothing to do for AtariSimulator.
  }

	virtual State* getState() {
		return currentState_;
	}

	virtual double act(const SimAction* action) {
		reward_ = 0;
		int prevLives = ale_->lives();
		const AtariAction* other = dynamic_cast<const AtariAction*>(action);
		ale::Action act = other->act_;
		for (int i = 0; i < numRepeats_ && !ale_->gameOver(); ++i) {
			reward_ += ale_->act(act);
		}
		// update screen buffer
		// if (updateFrame_) {
		// 	frameBuffer_->pushFrame(ale_->getScreen());
		// }
		currentState_->snapshot_ = ale_->getSnapshot();
		lifeLost_ = (prevLives != ale_->lives());
		if(scaleReward_) {
			if(reward_ > 0) return 1;
			if(reward_ < 0) return -1;
			return 0;
		}

		return reward_;
	}

	// void setUpdateFrame() {
	// 	updateFrame_ = true;
	// }

	// minimize impact on other parts
	// return true if next state is not unique
	// conflict with pseudo death
	// solution: fix life counter in xitari
	// currently fixed: MsPacMan
	virtual bool actDiffer() {
		std::string currentSnapshot = currentState_->snapshot_;
		std::string prevSnapshot;
		for (int i = 0; i < actSet_.size(); ++i) {
			ale_->restoreSnapshot(currentSnapshot);
			for (int j = 0; j < numRepeats_ && !ale_->gameOver(); ++j) {
				ale_->act(actSet_[i]);
			}
			std::string snapshot = ale_->getSnapshot();
			if ((i > 0) && (prevSnapshot.compare(snapshot) != 0)) {
				ale_->restoreSnapshot(currentSnapshot);
				return true;
			}
			prevSnapshot = snapshot;
		}
		ale_->restoreSnapshot(currentSnapshot);
		return false;
	}

	// random action
	SimAction* getRandomAct() {
		int index = rand() % actVect_.size();
		return actVect_[index];
	}

	virtual vector<SimAction*>& getActions() {
		return actVect_;
	}

	virtual bool isTerminal() {
		return ale_->gameOver() || ( pseudoGameover_ && lifeLost_);
	}

	virtual void reset() {
		ale_->resetGame();
		currentState_->snapshot_ = ale_->getSnapshot();
		lifeLost_ = false;
	}

	// void recordData(string path, int index, const SimAction* action) {
	// 	char buffer[20];
	// 	sprintf(buffer, "/frames/%d.txt", index);
	// 	string input = path + buffer;
	// 	// this->frameBuffer_->writeToFile(input.c_str());
	// 	sprintf(buffer, "/act/%d.label", index);
	// 	string label = path + buffer;
	// 	const AtariAction* other = dynamic_cast<const AtariAction*>(action);
	// 	ale::Action act = other->act_;
	// 	ofstream myfile;
	// 	myfile.open(label.c_str());
	// 	if(myfile.fail()) {
	// 		cout << "Cannot access: " << label << endl;
	// 		myfile.clear();
	// 		exit(0);
	// 	}
	// 	myfile << act_map[act] << endl;
	// 	myfile.close();
  //
	// }

	bool getPseudoDeath() {
		return lifeLost_;

	}
	// control the condition of terminal states, true then lost live will termintate 
	const bool pseudoGameover_;
	// number of action repeats
	const int numRepeats_;
	// 
	const bool scaleReward_;
	double minReward_;
	double maxReward_;
	// true if just losing one life
	bool lifeLost_;
	//
	double reward_;
	// true then update frameBuffer_
	// bool updateFrame_;
	// FrameBuffer* frameBuffer_;
	// 
	ale::ALEInterface* ale_;
	// coding constraint: always equal ale state
	AtariState* currentState_; 
	// action set
	ale::ActionVect actSet_;
	vector<SimAction*> actVect_;

  const bool useObjectState_;
	
	// map<ale::Action, int> act_map;
	
};
*/

}  // namespace oodqn
