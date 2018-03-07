#ifndef __MCTS_H__
#define __MCTS_H__

#include <cmath>
#include <vector>
#include "constants.h"
#include "simulator.h"

namespace oodqn {

class ActionNode;

class StateNode {
public:
  StateNode(ActionNode* _parent, const State* _state,
            const std::vector<const Action*>& _actions,
            double _reward, bool _terminal);
  virtual ~StateNode();

  // True if all child actions have been explored.
  bool explored() const;

  // Create a new child ActionNode for the next unexplored action.
  // Return the index of new action in A(s)
  int addActionNode();

  // Parent action that led to this state.
  ActionNode* parent_;

  // Corresponding simulator state for this state node.
  State* state_;

  // Reward received after transitioning into this state,
  // i.e. R(s, a, s') where (s,a) are ancestors of this node.
  const double reward_;

  // Number of times this node was visited: N(s)
  int visits_;

  // Index of the next action to explore.
  int next_action_idx_;

  // List of actions that can be taken from this state.
  std::vector<Action*> actions_;

  // Child action node for each action tried.
  std::vector<ActionNode*> children_;

  // Whether this node is terminal.
  bool terminal_;
};

class ActionNode {
public:
  ActionNode(StateNode* _state);
  ~ActionNode() ;
  // Return the child StateNode containing the given state *for planning*
  StateNode* getStateNode(const State* state) const;

  // Create and return a child StateNode for the given next state.
  StateNode* addStateNode(const State* state,
                          const std::vector<const Action*>& actions,
                          double reward, bool terminal);

  // Return whether the given state is in {s' ~ T(s,a)} *for planning*
  bool containsNextState(const State* state);

  // Parent state node.
  StateNode* const parent_;

  // List of children.
  // NOTE: This should always be size at most 1 for deterministic envs!
  std::vector<StateNode*> children_;

  // Estimated Q(s,a)
  double q_value_;

  // Number of visits: N(s,a)
  int visits_;
};

class MCTSPlanner
{
public:
  MCTSPlanner(Simulator* sim, int max_depth, int num_traj, double ucb_scale,
              double gamma);

  // We don't own sim_, so don't delete it here.
  virtual ~MCTSPlanner(); 

  // Set the new root node.
  void setRootNode(const State* state,
                   const std::vector<const Action*>& actions,
                   double reward, bool terminal); 

  // Plan a single step.  If step >= 0, record the first rollout
  // used during planning.
  void plan(int step = -1);

  // Obtain the planned action for root after planning
  Action* getAction() const;

  // Return the action with highest value for root node
  int getGreedyBranchIndex() const;

  // Return the index of action to sample.
  // Add a new action node to tree if the action wass never sampled
  int getUCTBranchIndex(StateNode* node);

  // Backpropagate along the path from leaf to root based on the
  // MC sample of return by updating values and counters.
  void updateValues(StateNode* node, double mc_return);
  // Sample trajectory to the specified depth.
  double rollout(StateNode* node, int depth, int step, int traj);

  // Memory management and search tree pruning.
  void prune(Action* acttion, int history_size);
  void pruneState(StateNode* node);
  void pruneAction(ActionNode* node);
  void pruneAncestors(int history_size);

  // Wipe the search tree.
  void clearTree();

  inline bool terminalRoot() const {
    return root_->terminal_;
  }

  inline void setRolloutPrefix(const std::string& prefix) {
    rollout_prefix_ = prefix;
  }

protected:
  // Simulator interface.
  Simulator* sim_;

  // UCT hyperparameters.
  int max_depth_;
  int num_traj_;
  double ucb_scale_;
  double gamma_;
  StateNode* root_;

  // File name prefix for saved frames from all rollouts.
  std::string rollout_prefix_;

  // void printRootValues() {
  //   int size = root_->nodeVect_.size();
  //   for (int i = 0; i < size; ++i) {
  //     double val = root_->nodeVect_[i]->avgReturn_;
  //     int numVist = root_->nodeVect_[i]->numVisits_;
  //     cout << "(";
  //     root_->actVect_[i]->print();
  //     cout << "," << val << "," << numVist << ") ";
  //   }
  //   cout << root_->isTerminal_ ;
  //   // cout << endl;
  // }
};

}  // namespace oodqn

#endif
