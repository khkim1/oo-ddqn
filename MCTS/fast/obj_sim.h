#ifndef __OBJ_SIM_H__
#define __OBJ_SIM_H__

#include "mcts.h"
#include <string>
#include "tf_util.h"
#include "constants.h"

using namespace std;
using namespace tensorflow;

class ObjectState: public State {
  public:
    ObjectState(const Vec& objState) {
      objState_ = objState;
      // TODO: Hardcoded for Pong
      assert(objState_.size() == 9);
    }

    ObjectState(const float* objState, int size) {
      objState_.clear();
      for (int i = 0; i < size; i++) {
        objState_.push_back(objState[i]);
      }
      // TODO: Hardcoded for Pong
      assert(objState_.size() == 9);
    }

    virtual bool equal(State* state) {
      const ObjectState* other = dynamic_cast<const ObjectState*>(state);
      if ((other == NULL) || (objState_ != other->objState_)) {
        return false;
      }
      return true;
    }

    virtual State* duplicate() {
      return new ObjectState(objState_);
    }

    virtual void print() const {
      std::cout << "[";
      for (int i = 0; i < objState_.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << objState_[i];
      }
      std::cout << "]";
    }
          
    ~ObjectState() {}

    Vec objState_;
};


class ObjectAction: public SimAction {
  public:
    int act_;
    ObjectAction (int & act): act_(act) {}

    virtual SimAction* duplicate() {
      return new ObjectAction(act_);
    }

    virtual void print() const {
      std::cout << act_;
    }

    virtual bool equal(SimAction* action) {
      ObjectAction* other = dynamic_cast<ObjectAction*>(action);
      return other->act_ == act_;
    }
          
    ~ObjectAction() {}
};


class ObjectSimulator : public Simulator {
  public:
    double minReward_;
    double maxReward_;
    // true if just losing one life
    bool lifeLost_;
    //
    double reward_;

    bool isTerminal_;

    // Control the condition of terminal states:
    // true then lost live will terminate.
    // const bool pseudoGameover_;

    // number of action repeats
    const int numRepeats_;
    const bool scaleReward_;

    // TODO
    // true then update frameBuffer_
    // bool updateFrame_;
    // FrameBuffer* frameBuffer_;

    Vec history_;
    ObjectState* currentState_;
    vector<SimAction*> actionSet_;
    string stateModelPrefix_;
    string rewardModelPrefix_;
    std::unique_ptr<TFModel> stateModel_;
    std::unique_ptr<TFModel> rewardModel_;

	  ObjectSimulator(const string& stateModelPrefix,
                    const string& rewardModelPrefix,
                    bool scaleReward,
                    int numRepeats,
                    int num_actions
                    // bool updateFrame,
                    // bool pseudoGameover, 
                    ):
    stateModelPrefix_(stateModelPrefix),
    rewardModelPrefix_(rewardModelPrefix),
		scaleReward_(scaleReward), 
		numRepeats_(numRepeats), 
		// updateFrame_(updateFrame),
		// pseudoGameover_(pseudoGameover), 
		reward_(0) {

    cout << "[dqn] LOADING STATE MODEL" << endl;
    stateModel_.reset(new TFModel(stateModelPrefix_));
    cout << "[dqn] LOADING REWARD MODEL" << endl;
    rewardModel_.reset(new TFModel(rewardModelPrefix_));
    cout << "[dqn] SETTING CURRENT STATE" << endl;
		currentState_ = new ObjectState(
        kResetState, sizeof(kResetState) / sizeof(kResetState[0]));
    cout << "[dqn] SETTING ACTION SET" << endl;
    for (int i = 0; i < num_actions; ++i) {
      actionSet_.push_back(new ObjectAction(i));
    }
		lifeLost_ = false;
    isTerminal_ = false;

		cout << "Simulator Information: "
		     << "\nState model: " << stateModelPrefix_
		     << "\nReward model: " << rewardModelPrefix_
		     << "\nScale Reward: " << (scaleReward_ ? "True" : "False" )
		     << "\nNum actions: " << actionSet_.size() << "\n"
		     // << "\nPseudo GameOver: " << (pseudoGameover_ ? "True" : "False")
		     << endl;
	}

	~ObjectSimulator () {
		delete currentState_;
	}

  // TODO: Needed?
	int lives() {
		// return ale_->lives();
		return -1;
	}

	virtual void setState(State* state) {
		const ObjectState* other = dynamic_cast<const ObjectState*> (state);
    currentState_->objState_ = other->objState_;
		lifeLost_ = false;
	}

  virtual void setHistory(vector<State*> history) {
    history_.clear();
    for (const State* state : history) {
      const ObjectState* _state = dynamic_cast<const ObjectState*> (state);
      for (int i = 0; i < _state->objState_.size(); ++i) {
        history_.push_back(_state->objState_[i]);
      }
    }
  }

	virtual State* getState() {
		return currentState_;
	}

	SimAction* getRandomAct() {
		return actionSet_[rand() % actionSet_.size()];
	}

	virtual vector<SimAction*>& getActions() {
		return actionSet_;
	}

  // TODO: How do we check for terminal condition for non-Pong games?
	virtual bool isTerminal() {
		// return ale_->gameOver() || ( pseudoGameover_ && lifeLost_);
    return isTerminal_;
	}

	virtual void reset() {
    if (currentState_ != nullptr) {
      delete currentState_;
    }
		currentState_ = new ObjectState(
        kResetState, sizeof(kResetState) / sizeof(kResetState[0]));
		lifeLost_ = false;
    isTerminal_ = false;
	}


  // TODO: The following prediction logic is hardcoded for Pong!!
  Vec predictState(const Vec& input) {
    Tensor out;
    Vec nextState(9);

    // Get next Ant paddle state
    stateModel_->RunVector(input, kInputPlaceholderAnt,
                           kNextStateTensorAnt, &out);
    nextState[0] = out.matrix<float>()(0, 0);
    nextState[1] = out.matrix<float>()(0, 1);
    nextState[2] = out.matrix<float>()(0, 2);

    // Get next ball state
    stateModel_->RunVector(input, kInputPlaceholderBall,
                           kNextStateTensorBall, &out);
    nextState[3] = out.matrix<float>()(0, 0);
    nextState[4] = out.matrix<float>()(0, 1);
    nextState[5] = out.matrix<float>()(0, 2);

    // Get next Pro paddle state
    stateModel_->RunVector(input, kInputPlaceholderPro,
                           kNextStateTensorPro, &out);
    nextState[6] = out.matrix<float>()(0, 0);
    nextState[7] = out.matrix<float>()(0, 1);
    nextState[8] = out.matrix<float>()(0, 2);

    return nextState;
  }

  float predictReward(const vector<float>& stateVec) {
    Tensor out;
    rewardModel_->RunMatrix(stateVec, TensorShape({1, 9}),
                            kInputPlaceholderReward, kRewardTensor, &out);
    float neg, zero, pos;
    neg = out.matrix<float>()(0, 0);
    zero = out.matrix<float>()(0, 1);
    pos = out.matrix<float>()(0, 2);

    if (neg > zero && neg > pos)
      return -1;
    else if (pos > neg && pos > zero)
      return 1;
    else
      return 0;
  }

	virtual double act(const SimAction* action) {
		reward_ = 0;
		const ObjectAction* objAction = dynamic_cast<const ObjectAction*>(action);
		for (int i = 0; i < numRepeats_; ++i) {

      // TODO: MUCH OF THIS FUNCTION HARDCODED FOR PONG
      // Pad history and append onehot-encoded action.
      Vec input;
      const int pad_len = 108 - history_.size();
      for (int j = 0; j < pad_len; ++j) {
        input.push_back(0.f);
      }
      for (int j = 0; j < history_.size(); ++j) {
        input.push_back(history_[j]);
      }
      AppendOnehotAction(&input, objAction->act_, 6);

      assert(input.size() == 114);

      // Get next state
      const Vec nextState = predictState(input);
      // Get reward for the new state
      int single_r = predictReward(nextState);
      // Update history
      if (history_.size() == 108) {
        std::rotate(history_.begin(), history_.begin()+9, history_.end());
        history_.erase(history_.end()-9, history_.end());
        assert(history_.size() == 99);
      }
      for (int j = 0; j < nextState.size(); ++j) {
        history_.push_back(nextState[j]);
      }
      // std::copy(nextState.begin(), nextState.end(), history_.end());
      assert(history_.size() <= 108 && history_.size() % 9 == 0);
      currentState_->objState_ = nextState;

      if (single_r != 0) {
        cout << "[dqn] reward " << single_r << " received!! breaking" << endl;
        // For Pong, just terminate when we get a reward for now.
        reward_ = single_r;
        isTerminal_ = true;
        // TODO: Fix the following gameover logic for other games.
        break;
      }
		}

		// TODO: update screen buffer
		// if (updateFrame_) {
		// 	frameBuffer_->pushFrame(ale_->getScreen());
		// }

		// lifeLost_ = (prevLives != ale_->lives());
    // 
		if(scaleReward_) {
			if(reward_ > 0) return 1;
			if(reward_ < 0) return -1;
			return 0;
		}

		return reward_;
	}

	// minimize impact on other parts
	// return true if next state is not unique
	// conflict with pseudo death
	// solution: fix life counter in xitari
	// currently fixed: MsPacMan
	virtual bool actDiffer() {
		// std::vector<float> currentObjVec = currentState_->objState_;
		// std::vector<float> prevObjVec;
    return true;
    
    // `XX: rewrite this
    /*
		for (int i = 0; i < actionSet_.size(); ++i) {
			// for (int j = 0; j < numRepeats_ && !ale_->gameOver(); ++j) {
			for (int j = 0; j < numRepeats_; ++j) {
				ale_->act(actionSet_[i]);
			}
			std::string snapshot = ale_->getSnapshot();
			if ((i > 0) && snapshot != prevObjVec)) {
				ale_->restoreSnapshot(currentSnapshot);
				return true;
			}
			prevSnapshot = snapshot;
		}
		ale_->restoreSnapshot(currentSnapshot);
		return false;
    */
	}

  // TODO
	// void recordData(string path, int index, const SimAction* action) {}

};


#endif
