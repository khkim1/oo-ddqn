#ifndef __ATARI_OBJ_SIM_H__
#define __ATARI_OBJ_SIM_H__

#include <string>
#include <ale_interface.hpp>
#include "mcts.h"
#include "simulator.h"

namespace oodqn {

class AtariObjState : public State {
public:
  AtariObjState() = delete;
  // Constructing from ale::getSnapshot() and object vector
  AtariObjState(const std::string& snapshot, const Vec& objvec);
  // Constructing from ale::getSnapshot() and ale::ALEScreen
  AtariObjState(const std::string& snapshot, const ale::ALEScreen& screen);
  void setSnapshotAndObjvec(const std::string&, const Vec&);
  std::string getSnapshot() const { return snapshot_; }
  Vec getObjvec() const { return objvec_; }
  ~AtariObjState() {}

  AtariObjState& operator=(const AtariObjState &rhs) {
    snapshot_ = rhs.snapshot_;
    objvec_ = rhs.objvec_;
  }

  // Overrides from base class
  AtariObjState* clone() const override;
  bool equals(const State*) const override;
  bool equals_planning(const State*) const override;
  std::string str() const override;
  void print() const override;
  std::string getType() const override;

protected:
  std::string snapshot_;
  Vec objvec_;
};

/*
class AtariAction : public Action {
public:
  // Constructing from ale::Action (enum)
  AtariAction(const ale::Action);
  ~AtariAction() {};
  ale::Action getAleAction() const {
    return action_;
  }

  // Overrides from base class
  bool equals(const Action*) const override;
  Action* clone() const override;
  std::string str() const override;
  void print() const override;

protected:
  ale::Action action_;
};
*/

class AtariObjSim : public Simulator {
public:
  AtariObjSim() = delete;
  AtariObjSim(const std::string& romFile, int frameskip);
  ale::ALEScreen getScreen() const;
  void screenToPNG(const std::string& filename) const;
  virtual ~AtariObjSim();

  // Overrides from base class
  const State* getState() const override;
  void setState(const State*) override;
  // const State* getPlanningState() const override;
  // void setInternalState(const State&) override;
  double act(const Action*) override;
  void reset() override;
  bool isTerminal() const override;
  const std::vector<const Action*>& getActions() const override;
  bool actionIgnored() const override;
  void saveFrame(const std::string&) const override;

	// Frameskip determines how many times actions are repeated.
	const int frameskip_;

	// double reward_;

  // Instance of ALE simulator.
	ale::ALEInterface* ale_;

  // This should always match the ALE state.
	AtariObjState* current_state_; 

	// List of available actions.
  std::vector<const Action*> actions_;
};

} // namespace oodqn

#endif
