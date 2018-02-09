#ifndef __MODEL_SIM_HPP__
#define __MODEL_SIM_HPP__

#include "uct.hpp"
#include <string>
#include "atari_images.hpp"
#include "tf_util.h"

using namespace std;
using namespace tensorflow;

class ObjectState: public State {
  public:
    ObjectState(const std::vector<float>& objVec) {
      objVec_ = objVec;
    }

    virtual bool equal(State* state) {
      const ObjectState* other = dynamic_cast<const ObjectState*>(state);
      if ((other == NULL) || (objVec_ != other->objVec_)) {
        return false;
      }
      return true;
    }

    virtual State* duplicate() {
      return new ObjectState(objVec_);
    }

    virtual void print() const {
      std::cout << "[";
      for (int i = 0; i < objVec_.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << objVec_[i];
      }
      std::cout << "]";
    }
          
    ~ObjectState() {}

    std::vector<float> objVec_;
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
    // // set the state of simulator
    // virtual void setState(State* state) = 0;
    // // get current state of simulator, no state is created
    // virtual State* getState() = 0;
    // // one step simulation
    // virtual double act(const SimAction* action) = 0;
    // // get all available actions for current state
    // // need to handle memory of actions
    // virtual vector<SimAction*>& getActions() = 0;
    // // return whether current state is terminal state
    // virtual bool isTerminal() = 0;
    // // reset the state of the simulator
    // virtual void reset() = 0;


    double minReward_;
    double maxReward_;
    // true if just losing one life
    bool lifeLost_;
    //
    double reward_;


    // Control the condition of terminal states:
    // true then lost live will terminate.
    const bool pseudoGameover_;
    // number of action repeats
    const int numRepeats_;
    const bool scaleReward_;

    // true then update frameBuffer_
    bool updateFrame_;
    // TODO
    // FrameBuffer* frameBuffer_;

    ObjectState* initialState_;
    ObjectState* currentState_;
    vector<SimAction*> actionSet_;
    TFModel model_;

	  ObjectSimulator (const string& model_prefix, bool updateFrame,
                  bool pseudoGameover, bool scaleReward, int numRepeats,
                  const vector<float>& initialState_vec,
                  int num_actions):
		updateFrame_(updateFrame),
		pseudoGameover_(pseudoGameover), 
		scaleReward_(scaleReward), 
		numRepeats_(numRepeats), 
    model_(model_prefix),
		reward_(0) {

    initialState_ = new ObjectState(initialState_vec),
		currentState_ = new ObjectState(initialState_vec);
    for (int i = 0; i < num_actions; ++i) {
      // TODO: PONG
      actionSet_.push_back(new ObjectAction(i));
    }
		lifeLost_ = false;

		cout << "Simulator Information: "
		     << "\nModel prefix: " << model_.model_prefix_
		     << "\nPseudo GameOver: " << (pseudoGameover_ ? "True" : "False")
		     << "\nScale Reward: " << (scaleReward_ ? "True" : "False" )
		     << "\nNum actions: " << actionSet_.size() << "\n"
         << "\nInitial state: ";
    initialState_->print();
		cout << endl;
	}

	~ObjectSimulator () {
		delete currentState_;
    delete initialState_;
	}

  // TODO: Needed?
	int lives() {
		// return ale_->lives();
		return -1;
	}

	virtual void setState(State* state) {
		const ObjectState* other = dynamic_cast<const ObjectState*> (state);
    currentState_->objVec_ = other->objVec_;
		lifeLost_ = false;
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

  // TODO: How do we check for terminal condition?
	virtual bool isTerminal() {
		// return ale_->gameOver() || ( pseudoGameover_ && lifeLost_);
    return false;
	}

	virtual void reset() {
    if (currentState_ != nullptr) {
      delete currentState_;
    }
		currentState_ = dynamic_cast<ObjectState*>(initialState_->duplicate());
		lifeLost_ = false;
	}

	virtual double act(const SimAction* action) {
		reward_ = 0;
		// int prevLives = ale_->lives();
		const ObjectAction* other = dynamic_cast<const ObjectAction*>(action);
		int act = other->act_;
    // XXX: how to check for game over
		// for (int i = 0; i < numRepeats_ && !ale_->gameOver(); ++i) {
		for (int i = 0; i < numRepeats_; ++i) {
      Tensor out;
      // XXX: Call model -- pred = ?
      if (model_.Run(currentState_->objVec_, "pred", &out).ok()) {
        for (int j = 0; j < currentState_->objVec_.size(); ++j) {
          currentState_->objVec_[j] = out.vec<float>()(j);
        }
      }
      // XXX: How to get reward?
			// reward_ += ale_->act(act);
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
		std::vector<float> currentObjVec = currentState_->objVec_;
		std::vector<float> prevObjVec;
    return false;
    
    // XXX
		// for (int i = 0; i < actSet_.size(); ++i) {
		// 	ale_->restoreSnapshot(currentSnapshot);
		// 	for (int j = 0; j < numRepeats_ && !ale_->gameOver(); ++j) {
		// 		ale_->act(actSet_[i]);
		// 	}
		// 	std::string snapshot = ale_->getSnapshot();
		// 	if ((i > 0) && (prevSnapshot.compare(snapshot) != 0)) {
		// 		ale_->restoreSnapshot(currentSnapshot);
		// 		return true;
		// 	}
		// 	prevSnapshot = snapshot;
		// }
		// ale_->restoreSnapshot(currentSnapshot);
		// return false;
	}

  // TODO
	void recordData(string path, int index, const SimAction* action) {}

};


#endif
