#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <string>
#include <typeinfo>
#include <vector>

namespace oodqn {

class State {
public:
  bool operator==(const State& other) = delete;

  // Check if this state is same as another state.
  virtual bool equals(const State*) const = 0;
  // Check if this state is same as another state *FOR PLANNING*.
  virtual bool equals_planning(const State*) const = 0;

  // Create and return a copy of this state.
  virtual State* clone() const = 0;

  virtual std::string str() const = 0;
  virtual void print() const = 0;
  virtual std::string getType() const = 0;
  virtual ~State(){};
};

class Action {
public:
  bool operator==(const Action& other) const {
    return typeid(*this) == typeid(other) && equals(&other);
  }

  // Check if this action is same as another action.
  virtual bool equals(const Action*) const = 0;

  // Create and return a copy of this action.
  virtual Action* clone() const = 0;

  virtual std::string str() const = 0;
  virtual void print() const = 0;

  virtual ~Action() {};
};

class Simulator {
public:

  // ----- State-related -----

  // virtual const State* getInternalState() const = 0;
  virtual const State* getState() const = 0; 
  virtual void setState(const State*) = 0;

  // ----- Simulation-related -----
  
  // Take action in the simulator, return reward, and internally
  // update states (which then can be fetched).
  virtual double act(const Action*) = 0;

  // Reset the simulation.
  virtual void reset() = 0;

  // Whether the simulation has reached a terminal state.
  virtual bool isTerminal() const = 0;

  // Return all available actions from current simulator state.
  virtual const std::vector<const Action*>& getActions() const = 0;

  Action* getRandomAction() {
    const std::vector<const Action*>& actions = getActions();
    const int idx = rand() % actions.size(); 
    return actions[idx]->clone();
  }

  // ----- Miscellaneous -----
  
  // Tries all possible actions and checks if action makes any difference
  // in the next state. If not, we can just take any action and skip planning
  // for this step.
  virtual bool actionIgnored() const = 0;
  virtual ~Simulator() {};

};

} // namespace oodqn

#endif
