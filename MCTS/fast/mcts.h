#ifndef __MCTS_H__
#define __MCTS_H__

#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include "constants.h"
#include "simulator.h"

namespace oodqn {

class ActionNode;

class StateNode {
public:
  StateNode(ActionNode* _parent, State* _state, const vector<Action*>& _actions,
            double _reward);
  virtual ~StateNode();

  // True if all child actions have been explored.
  bool explored() const;

  // Create a new child ActionNode for the next unexplored action.
  // XXX: return the index of new action in A(s)
  int addActionNode();

protected:
  // Parent action that led to this state.
  ActionNode* parent_;

  // Corresponding simulator state for this state node.
  State* state_;

  // Reward received after transitioning into this state,
  // i.e. R(s, a, s') where (s,a) are ancestors of this node.
  const double reward_;

  // Number of times this node was visited: N(s)
  int numVisits_;

  // Index of the next action to explore.
  int next_action_idx_;

  // List of actions that can be taken from this state.
  vector<Action*> actions_;

  // Child action node for each action tried.
  vector<ActionNode*> children_;

  // XXX
  // isTerminal(s)
  // bool isTerminal_;
};

class ActionNode {
public:
  ActionNode(StateNode* _state);
  ~ActionNode() ;
  // Return the child StateNode containing the given state.
  StateNode* getStateNode(State* state) const;

  // Create and return a child StateNode for the given next state.
  StateNode* addStateNode(State* _state, const vector<Action*>& _actions,
                          double _reward) ;

protcted:
  // Parent state node.
  StateNode* const parent_;

  // List of children.
  // XXX: This should always be size at most 1 for deterministic envs!
  vector<StateNode*> children_;

  // Estimated Q(s,a)
  double q_value_;

  // Number of visits: N(s,a)
  int visits_;

  // XXX: Not ported yet since we assume deterministic dynamics
  // return whether sx in {s' ~ T(s,a)}
  // bool containNextState(State* state);
};

class MCTSPlanner
{
public:
  // Simulator interfaces
  Simulator* sim_;
  // uct parameters
  int maxDepth_;
  int numRuns_;
  double ucbScalar_;
  double gamma_;
  double leafValue_;
  double endEpisodeValue_;
  // rand seed value
  StateNode* root_;

  MCTSPlanner(Simulator* _sim, int _maxDepth, int _numRuns, double _ucbScalar,
      double _gamma, double _leafValue = 0, double _endEpisodeValue = 0):
    sim_(_sim),
    maxDepth_(_maxDepth),
    numRuns_(_numRuns),
    ucbScalar_(_ucbScalar),
    gamma_(_gamma),
    leafValue_(_leafValue),
    endEpisodeValue_(_endEpisodeValue),
    root_(NULL) {}

  // does not handle sim_
  ~MCTSPlanner() {
    clearTree();
  }

  // set the root node in UCT
  void setRootNode(State* _state, vector<SimAction*> _actVect, double _reward,
      bool _isTerminal) {
    if (root_ != NULL) {
      clearTree();
    }
    root_ = new StateNode(NULL, _state, _actVect, _reward, _isTerminal);
  }

  // start planning
  void plan();

  // get the planned action for root
  // called after planning
  SimAction* getAction() {
    int idx = getGreedyBranchIndex();
    // cout << "[dbg] MCTS: Greedy branch index: " << idx << endl;
    // cout << "[dbg] MCTS: Corresponding act: ";
    // root_->actVect_[idx]->print();
    // cout << endl;
    return root_->actVect_[idx];
  }

  // return the most visited action for root node
  int getMostVisitedBranchIndex();
  // return the most visited action for root node
  int getGreedyBranchIndex();
  // return the index of action in root
  // add a new action node to tree if the action is never sampled
  int getUCTRootIndex(StateNode* node);
  // return the index of action to sample
  // add a new action node to tree if the action is never sampled
  int getUCTBranchIndex(StateNode* node);

  // update the values along the path from leaf to root
  // update the counters
  void updateValues(StateNode* node, double mcReturn);
  // sample trajectory to a depth
  double MC_Sampling(StateNode* node, int depth);
  // sample to the end of an episode
  double MC_Sampling(StateNode* node);

  // modify the reward function
  // currently it does nothing
  double modifyReward(double orig) {
    return orig;
  }

  void printRootValues() {
    int size = root_->nodeVect_.size();
    for (int i = 0; i < size; ++i) {
      double val = root_->nodeVect_[i]->avgReturn_;
      int numVist = root_->nodeVect_[i]->numVisits_;
      cout << "(";
      root_->actVect_[i]->print();
      cout << "," << val << "," << numVist << ") ";
    }
    cout << root_->isTerminal_ ;
    // cout << endl;
  }

  // release all nodes
  void clearTree() {
    if (root_ != NULL) {
      pruneState(root_);
    }
    root_ = NULL;
  }
  // adjust UCT tree by pruning out all other branches
  bool terminalRoot() {
    return root_->isTerminal_;
  }
  void prune(SimAction* act, int historySize);
  // prune out state node and its children
  void pruneState(StateNode* state);
  // prune out action node and its children
  void pruneAction(ActionNode* act);

  void pruneAncestors(int historySize);

  void realStep(SimAction* act, State* newRealState, float rwd, bool isTerminal);
};

}  // namespace oodqn

#endif
