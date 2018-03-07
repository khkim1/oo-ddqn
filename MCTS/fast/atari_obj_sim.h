#ifndef __ATARI_OBJ_SIM_H__
#define __ATARI_OBJ_SIM_H__

#include <string>
#include <ale_interface.hpp>
#include "simulator.h"
#include "mcts.h"
#include "tf_util.h"

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
    if (this != &rhs) {
      snapshot_ = rhs.snapshot_;
      objvec_ = rhs.objvec_;
    }
    return *this;
  }

  // Overrides from base class
  AtariObjState* clone() const override;
  bool equals(const State*) const override;
  bool equals_planning(const State*) const override;
  std::string str() const override;
  void print() const override;
  inline std::string getType() const override {
    return "AtariObjState";
  }

protected:
  std::string snapshot_;
  Vec objvec_;

private:
  void set_values(const std::string& snapshot, const Vec& objvec);
};

class AtariObjSim : public Simulator {
public:
  AtariObjSim() = delete;
  AtariObjSim(const std::string& romFile, int frameskip);
  // Use trained reward model
  AtariObjSim(const std::string& romFile, int frameskip,
              const std::string& reward_model_prefix);
  ale::ALEScreen getScreen() const;
  void screenToPNG(const std::string& filename) const;
  virtual ~AtariObjSim();

  // Overrides from base class
  const State* getState() const override;
  void setState(const State*) override;
  double act(const Action*) override;
  void reset() override;
  bool isTerminal() const override;
  const std::vector<const Action*>& getActions() const override;
  bool actionIgnored() const override;
  void saveFrame(const std::string&) const override;

	// Frameskip determines how many times actions are repeated.
	const int frameskip_;

  // Instance of ALE simulator.
	ale::ALEInterface* ale_;

  // This should always match the ALE state.
	AtariObjState* current_state_;

	// List of available actions.
  std::vector<const Action*> actions_;

  // Reward model
  bool use_reward_model_;
  TFModel* reward_model_;

  // **** For debugging reward model ****
  int reward_dbg_total = 0;
  int reward_dbg_diff = 0;

private:
  double predictReward(const Vec& objvec);
};

} // namespace oodqn

#endif
