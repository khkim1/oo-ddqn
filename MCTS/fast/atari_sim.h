#ifndef __ATARI_SIM_H__
#define __ATARI_SIM_H__

#include <string>
#include <ale_interface.hpp>
#include "simulator.h"
#include "mcts.h"
// #include "constants.h"

namespace oodqn {

class AtariState : public State {
public:
  AtariState() {};
  // Constructing from ale::getSnapshot()
  AtariState(const std::string&);
  void setSnapshot(const std::string&);
  std::string getSnapshot() const;
  ~AtariState() {}

  // Overrides from base class
  AtariState* clone() const override;
  bool equals(const State*) const override;
  bool equals_planning(const State*) const override;
  std::string str() const override;
  void print() const override;
  std::string getType() const override;

protected:
  std::string snapshot_;
};

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

class AtariSim : public Simulator {
public:
  AtariSim() = delete;
  AtariSim(const std::string& romFile, int frameskip);
  ale::ALEScreen getScreen() const;
  void screenToPNG(const std::string& filename) const;
  virtual ~AtariSim();

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

	// Frameskip determines how many times actions are repeated.
	const int frameskip_;

	// double reward_;

  // Instance of ALE simulator.
	ale::ALEInterface* ale_;

	// This should always match the ALE state.
	AtariState* current_state_; 

	// List of available actions.
  std::vector<const Action*> actions_;
};

} // namespace oodqn

#endif
