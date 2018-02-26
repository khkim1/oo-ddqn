#include <algorithm>
#include <cassert>
#include <iostream>
#include <glog/logging.h>
#include "mcts.h"
#include "constants.h"

using namespace std;

namespace oodqn {

//
// StateNode
// 

StateNode::StateNode(
    ActionNode* _parent, const State* _state,
    const std::vector<const Action*>& _actions, double _reward, bool _terminal):
  parent_(_parent),
  state_(_state->clone()),
  reward_(_reward),
  terminal_(_terminal),
  next_action_idx_(0) {
  // Copy actions.
  const int size = _actions.size();
  for (int i = 0; i < size; ++i) {
    actions_.push_back(_actions[i]->clone());
  }

  // Shuffle actions for tie-breaking.
  std::random_shuffle(actions_.begin(), actions_.end());

  VLOG(2) << "StateNode " << this
          << " created with " << actions_.size() << " actions";
}
StateNode::~StateNode() {
  delete state_;

  int size = actions_.size();
  for (int i = 0; i < size; ++i) {
    Action* tmp = actions_[i];
    delete tmp;
  }

  size = children_.size();
  for (int i = 0; i < size; ++i) {
    ActionNode* tmp = children_[i];
    delete tmp;
  }

  actions_.clear();
  children_.clear();
}
bool StateNode::explored() const {
  VLOG(2) << "explored() called. next_action_idx: " << next_action_idx_
          << ", actions.size(): " << actions_.size();
  return (next_action_idx_ == actions_.size());
}
int StateNode::addActionNode() {
  CHECK(next_action_idx_ < actions_.size());
  children_.push_back(new ActionNode(this));
  ++next_action_idx_;
  return next_action_idx_ - 1;
}
/*
vector<State*> StateNode::stateHistory(int length) {
  int cnt = 0;
  vector<State*> hist;
  StateNode* cur = this;
  assert(cur != nullptr);

  while (cnt < length) {
    hist.push_back(cur->state_);
    cnt++;
    if (cur->parentAct_ == nullptr) {
      break;
    }
    cur = cur->parentAct_->parentState_;
    assert(cur != nullptr);
  }
  std::reverse(hist.begin(), hist.end());
  return hist;
}
*/

//
// ActionNode
// 

ActionNode::ActionNode(StateNode* _parent) :
  parent_(_parent), q_value_(0), visits_(0) {}
ActionNode::~ActionNode() {
  const int size = children_.size();
  for (int i = 0; i < size; ++i) {
    StateNode* tmp = children_[i];
    delete tmp;
  }
  children_.clear();
}
StateNode* ActionNode::getStateNode(const State* state) const {
  const int size = children_.size();
  for (int i = 0; i < size; ++i) {
    if (children_[i]->state_->equals_planning(state)) {
      return children_[i];
    }
  }
  return nullptr;
}
StateNode* ActionNode::addStateNode(const State* _state,
                                    const vector<const Action*>& _actions,
                                    double _reward, bool _terminal) {
  const int size = children_.size();
  if (size > 0) {
    LOG(INFO) << "Multiple states for same (s,a) pair detected!"
              << "Total count: " << size+1 << endl;
  }
  children_.push_back(
      new StateNode(this, _state, _actions, _reward, _terminal));
  return children_[size];
}
bool ActionNode::containsNextState(const State* state) {
  return (getStateNode(state) != nullptr);
}

//
// MCTSPlanner
// 

void MCTSPlanner::plan() {
  VLOG(1) << "Planning started.";
  CHECK(root_ != NULL) << "Root is NULL!";

  // XXX: What's this logic??
  // to avoid treat root differently than other nodes, when first see a root,
  // just draw a MC sample trajectory
  int offset = root_->visits_;
  if (offset == 0) {
    root_->visits_++;
    offset++;
  }

  for (int traj = offset; traj < num_traj_; ++traj) {
    VLOG(1) << "Planning trajectory: " << traj;
    StateNode* current = root_;
    double mc_return = 0;
    int depth = 0;
    while (true) {
      ++depth;
      if (current->terminal_) {
        VLOG(1) << "Terminal node reached, stopping rollout with return "
                << mc_return;
        break;
      }

      // All children have been explored already, so use UCB1.
      if (current->explored()) {
        const int uct_idx = getUCTBranchIndex(current);
        VLOG(1) << "All children explored; using UCB1 index " << uct_idx;

        sim_->setState(current->state_);
        const double reward = sim_->act(current->actions_[uct_idx]);

        if (reward != 0.) {
          VLOG(2) << "Non-zero reward in planning: " << reward;
        }
        const State* next_state = sim_->getState();

        // If this state has already been visited, then just follow the path.
        if (current->children_[uct_idx]->containsNextState(next_state)) {
          VLOG(2) << "New state has been visited already -- following the path";
          current = current->children_[uct_idx]->getStateNode(next_state);
          continue;
        }
        
        // We encountered a new next state, so add a state node and rollout.
        else {
          StateNode* next_node = current->children_[uct_idx]->addStateNode(
              next_state, sim_->getActions(), reward, sim_->isTerminal());

          mc_return = rollout(next_node, max_depth_);
          current = next_node;
          break;
        }
      }
      
      // There is an unexplored action, so we perform rollout from that action.
      else {
        const int action_idx = current->addActionNode();
        VLOG(2) << "New ActionNode added: " << current->children_[action_idx];
        VLOG(2) << "Exploring next child: "
                << current->actions_[action_idx]->str();
        sim_->setState(current->state_);
        const double reward = sim_->act(current->actions_[action_idx]);
        StateNode* next_node = current->children_[action_idx]->addStateNode(
            sim_->getState(), sim_->getActions(), reward, sim_->isTerminal());

        VLOG(2) << "New StateNode added: " << next_node;

        mc_return = rollout(next_node, max_depth_);
        current = next_node;
        break;
      }
    }  // max_depth_ 

    // Backpropagate
    updateValues(current, mc_return);
  }  // num_traj_
}

int MCTSPlanner::getGreedyBranchIndex() {
  CHECK(root_ != nullptr) << "Root is NULL";
  vector<double> maximizer; //maximizer.clear();
  int size = root_->children_.size();
  for (int i = 0; i < size; ++i) {
    maximizer.push_back(root_->children_[i]->q_value_);
  }
  vector<double>::iterator max_it = std::max_element(maximizer.begin(),
                                                     maximizer.end());
  int index = std::distance(maximizer.begin(), max_it);
  VLOG(2) << "Chosen (greedy) child action: "
          << root_->actions_[index]->str();

  return index;
}

int MCTSPlanner::getUCTBranchIndex(StateNode* node) {
  VLOG(2) << "Calculating UCB1 index";
  double det = log((double)node->visits_);

  vector<double> maximizer;
  int size = node->children_.size();
  VLOG(2) << "Number of available actions for " << node << " : " << size;
  for (int i = 0; i < size; ++i) {
    double val = node->children_[i]->q_value_;
    VLOG(2) << "Action " << i << " Q value: " << val;
    val += ucb_scale_ * sqrt(det / (double)node->children_[i]->visits_);
    VLOG(2) << "Action " << i << " ucb1 value: " << val;
    maximizer.push_back(val);
  }
  vector<double>::iterator max_it = std::max_element(maximizer.begin(),
                                                     maximizer.end());
  int index = std::distance(maximizer.begin(), max_it);
  VLOG(2) << "Chosen (UCB1) child action: "
          << root_->actions_[index]->str();

  return index;
}

void MCTSPlanner::updateValues(StateNode* node, double mc_return) {
  double totalReturn = mc_return;
  node->visits_++;
  while (node->parent_ != nullptr) {
    ActionNode* parent_action = node->parent_;
    parent_action->visits_++;
    totalReturn *= gamma_;
    totalReturn += node->reward_;
    // Update average Q value.
    parent_action->q_value_ += 
      (totalReturn - parent_action->q_value_) / parent_action->visits_;
    node = parent_action->parent_;
    node->visits_++;
  }
}

double MCTSPlanner::rollout(StateNode* node, int depth) {
  double ret = 0;
  sim_->setState(node->state_);
  VLOG(2) << "Rollout begin";
  double discnt = 1;
  for (int i = 0; i < depth; i++) {
    if (sim_->isTerminal()) {
      break;
    }
    const vector<const Action*>& actions = sim_->getActions();
    // Print the state vector
    // VLOG(2) << sim_->getState()->str();
    int action_idx = rand() % actions.size();
    const double reward = sim_->act(actions[action_idx]);
    ret += discnt * reward;
    discnt *= gamma_;
  }
  VLOG(2) << "Rollout reward: " << ret;
  return ret;
}


void MCTSPlanner::pruneState(StateNode * state) {
  int size = state->children_.size();
  VLOG(2) << "Pruning State with size " << size;
  for (int i = 0; i < size; ++i) {
    ActionNode* tmp = state->children_[i];
    pruneAction(tmp);
  }
  state->children_.clear();
  VLOG(2) << "Children all pruned";
  delete state;
  VLOG(2) << "State itself deleted";
}

void MCTSPlanner::pruneAction(ActionNode * action) {
  VLOG(2) << "Pruning Action ";
  int sizeNode = action->children_.size();
  for (int i = 0; i < sizeNode; ++i) {
    StateNode* tmp = action->children_[i];
    pruneState(tmp);
  }
  action->children_.clear();
  VLOG(2) << "Children all pruned";
  delete action;
  VLOG(2) << "Action itself deleted";
}

// XXX: finish this.
// Prune all ancestors, except keeps `historySize` of them to use with
// the state prediction model.
void MCTSPlanner::pruneAncestors(int history_size) {
  // int cnt = 0;
  // StateNode* current = root_;
  // while (cnt < historySize && current->parentAct_ != NULL) {
  //   assert(current->parentAct_->parentState_ != NULL);
  //   cnt++;
  //   current = current->parentAct_->parentState_;
  // }
  // pruneState(current);
}

/*
void MCTSPlanner::realStep(SimAction* act, State* newRealState, float rwd, bool isTerminal) {
  int size = root_->nodeVect_.size();
  StateNode * nextRoot = nullptr;
  for (int i = 0 ; i < size; ++i) {
    if (act->equal(root_->actVect_[i])) {
      assert(root_->nodeVect_[i]->stateVect_.size() == 1);
      delete root_->nodeVect_[i]->stateVect_[0];
      root_->nodeVect_[i]->stateVect_.clear();
      root_->nodeVect_[i]->addStateNode(
          newRealState, sim_->getActions(), rwd, isTerminal);
      nextRoot = root_->nodeVect_[i]->stateVect_[0];
      assert(root_->nodeVect_[i]->stateVect_.size() == 1);
      break;
      // ActionNode* tmp = root_->nodeVect_[i];
      // delete tmp;
    }
  }
  assert(nextRoot != nullptr);
  // cout << "[dbg] MCTS: Prev root: " << root_
  //      << ", New root: " << nextRoot
  //      << ", New root terminal: " << root_->isTerminal_
  //      << endl;
  root_ = nextRoot;
  // root_->parentAct_ = NULL;
  sim_->setState(newRealState);
  sim_->setTerminal(isTerminal);
}
*/

// XXX: Fix memory leaks
void MCTSPlanner::prune(Action* act, int history_size) {
  VLOG(2) << "Pruning started for action " << act->str()
          << ", history size " << history_size
          << " from state node " << root_;
  StateNode * next_root = nullptr;
  int size = root_->children_.size();
  for (int i = 0 ; i < size; ++i) {
    // if (act->equals(root_->actions_[i])) {
    if (*act == *root_->actions_[i]) {
      VLOG(2) << "Action idx " << i << " "
              << root_->actions_[i]->str() << " is the new root: "
              << root_->children_[i]->children_[0];
      // Deterministic environment.
      CHECK(root_->children_[i]->children_.size() == 1);
      next_root = root_->children_[i]->children_[0];
      ActionNode* tmp = root_->children_[i];
      // delete tmp;
    } else {
      VLOG(2) << "Action idx " << i << " "
              << root_->actions_[i]->str() << " is pruned: "
              << root_->actions_[i];
      ActionNode* tmp = root_->children_[i];
      // pruneAction(tmp);
    }
  }

  if (history_size > 0) {
    pruneAncestors(history_size);
  }

  CHECK(next_root != nullptr);

  VLOG(2) << "Old root: " << root_;
  // delete root_;
  root_ = next_root;
  root_->parent_ = nullptr;

  VLOG(2) << "New root: " << root_;
}

}  // namespace oodqn
