#include "mcts.h"
#include "constants.h"

using namespace std;
//=================== state node part ===============

// init state node: shuffle actions as required
StateNode::StateNode(ActionNode* _parentAct, State* _state,
    vector<SimAction*>& _actVect, double _reward, bool _isTerminal):
  parentAct_(_parentAct),
  state_(_state->duplicate()),
  reward_(_reward),
  isTerminal_(_isTerminal),
  numVisits_(0),
  actPtr_(0) {
  // copy _actVect to actVect
  int _size = _actVect.size();
  for (int i = 0; i < _size; ++i) {
    actVect_.push_back(_actVect[i]->duplicate());
  }
  // shuffle actions
  std::random_shuffle( actVect_.begin(), actVect_.end() );
}

// free s, A(s) & A(s) nodes
StateNode::~StateNode() {
  delete state_;

  int sizeAct = actVect_.size();
  for (int i = 0; i < sizeAct; ++i) {
    SimAction* tmp = actVect_[i];
    delete tmp;
  }
  actVect_.clear();
}

// return whether all actions are sampled
bool StateNode::isFull() {
  return (actPtr_ == actVect_.size());
}

// create a node for next new action in A(s)
// return the index of new action in A(s)
int StateNode::addActionNode() {
  assert(actPtr_ < actVect_.size());
  nodeVect_.push_back(new ActionNode(this));
  ++actPtr_;
  return actPtr_ - 1;
}

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

//============ end of state node ====================

//============ action node part ======================
// init action node
ActionNode::ActionNode(StateNode* _parentState):
  parentState_(_parentState),
  avgReturn_(0),
  numVisits_(0) {}

// free state nodes
ActionNode::~ActionNode() {
  // int sizeNode = stateVect_.size();
  // for (int i = 0; i < sizeNode; ++i) {
  //   StateNode* tmp = stateVect_[i];
  //   delete tmp;
  // }
  // stateVect_.clear();
}

// return whether sx in {s' ~ T(s,a)}
bool ActionNode::containNextState(State* state) {
  int size = stateVect_.size();
  for (int i = 0; i < size; ++i) {
    if (state->equal(stateVect_[i]->state_)) {
      return true;
    }
  }
  return false;
}

// return the state node containing the next state
StateNode* ActionNode::getNextStateNode(State* state) {
  int size = stateVect_.size();
  for (int i = 0; i < size; ++i) {
    if (state->equal(stateVect_[i]->state_)) {
      return stateVect_[i];
    }
  }
  return NULL;
}

// create a state node for the next state
// return the new state node
StateNode* ActionNode::addStateNode(State* _state, vector<SimAction*>& _actVect, double _reward, bool _isTerminal) {
  int index = stateVect_.size();
  if (index > 0) {
    cout << "Added state node, total: " << index+1 << endl;
  }
  stateVect_.push_back(new StateNode(this, _state, _actVect,  _reward, _isTerminal));
  return stateVect_[index];
}
// =============== end of action node ================

// ================  uct part =========================
void MCTSPlanner::plan() {
  cout << "[dbg] MCTS: Planning started" << endl;
  assert(root_ != NULL);

  // // to avoid treat root differently than other nodes, when first see a root, just draw a MC sample trajectory
  // int rootOffset = root_->numVisits_;
  // if (rootOffset == 0) {
  //   root_->numVisits_++;
  //   rootOffset++;
  // }
  int rootOffset = 0;

  for (int trajectory = rootOffset; trajectory < numRuns_; ++trajectory) {
    // cout << "[dbg] MCTS: Planning trajectory: " << trajectory << endl;
    StateNode* current = root_;
    double mcReturn = leafValue_;
    int depth = 0;
    while (true) {
      ++depth;
      if (current->isTerminal_) {
        cout << "[dbg] MCTS: Current " << current
             << " is terminal, breaking out" << endl;
        mcReturn = endEpisodeValue_;
        break;
      }

      if (current->isFull()) { //follow the UCT tree
        // cout << "[dbg] MCTS: All children explored" << endl;
        //sample a node using UCB1
        int uctBranch = getUCTBranchIndex(current);
        // int uctBranch = 0;
        // if (current == root_) {
        //   assert (depth == 1);
        //   uctBranch = getUCTRootIndex(current);
        // } else {
        //   uctBranch = getUCTBranchIndex(current);
        // }

        sim_->setHistory(current->stateHistory(HISTORY_SIZE));
        sim_->setState(current->state_);
        double r = sim_->act(current->actVect_[uctBranch]);
        State* nextState = sim_->getState();

        //if old node
        if (current->nodeVect_[uctBranch]->containNextState(nextState)) {
          //follow path
          current = current->nodeVect_[uctBranch]->getNextStateNode(nextState);
          continue;
        } else { //new s'
          //add new state node
          //then MC sampling
          StateNode* nextNode = current->nodeVect_[uctBranch]->addStateNode(
              nextState, sim_->getActions(), r, sim_->isTerminal());

          if (-1 == maxDepth_) {
            mcReturn = MC_Sampling(nextNode);
          } else {
            mcReturn = MC_Sampling(nextNode, maxDepth_ - depth);
          }
          current = nextNode;
          break;
        }
      } else { //start MC-Sampling for the new action
        int actID = current->addActionNode();
        // cout << "[dbg] MCTS: Exploring next child: ";
        // current->actVect_[actID]->print();
        // cout<< endl;
        sim_->setHistory(current->stateHistory(HISTORY_SIZE));
        sim_->setState(current->state_);
        double r = sim_->act(current->actVect_[actID]);
        StateNode* nextNode = current->nodeVect_[actID]->addStateNode(
            sim_->getState(), sim_->getActions(), r, sim_->isTerminal());

        if (-1 == maxDepth_) {
          mcReturn = MC_Sampling(nextNode);
        } else {
          mcReturn = MC_Sampling(nextNode, maxDepth_ - depth);
        }
        current = nextNode;
        break;
      }
    }// end of for (depth)
    // update tree values
    updateValues(current, mcReturn);
  }// end of for (numRuns)
}

int MCTSPlanner::getMostVisitedBranchIndex() {
  assert(root_ != NULL);
  vector<double> maximizer;//maximizer.clear();
  int size = root_->nodeVect_.size();
  for (int i = 0; i < size; ++i) {
    maximizer.push_back(root_->nodeVect_[i]->numVisits_);
  }
  vector<double>::iterator max_it = std::max_element(maximizer.begin(), maximizer.end());
  int index = std::distance(maximizer.begin(), max_it);

  return index;
}

int MCTSPlanner::getGreedyBranchIndex() {
  assert(root_ != NULL);
  vector<double> maximizer;//maximizer.clear();
  int size = root_->nodeVect_.size();
  // cout << "[dbg] Root child scores: ";
  for (int i = 0; i < size; ++i) {
    maximizer.push_back(root_->nodeVect_[i]->avgReturn_);
    // cout << root_->nodeVect_[i]->avgReturn_ << " ";
  }
  // cout << endl;
  vector<double>::iterator max_it = std::max_element(maximizer.begin(), maximizer.end());
  int index = std::distance(maximizer.begin(), max_it);
  // cout << "[dbg] Chosen root child index: " << index << endl;

  return index;

}

// int MCTSPlanner::getUCTRootIndex(StateNode* node) {
//   double det = log((double)node->numVisits_);
//
//   vector<double> maximizer;
//   int size = node->nodeVect_.size();
//   for (int i = 0; i < size; ++i) {
//     //double val = node->nodeVect[i]->avgReturn + node->internalR[i];
//     double val = node->nodeVect_[i]->avgReturn_;
//     val += ucbScalar_ * sqrt(det / (double)node->nodeVect_[i]->numVisits_);
//     maximizer.push_back(val);
//   }
//   vector<double>::iterator max_it = std::max_element(maximizer.begin(), maximizer.end());
//   int index = std::distance(maximizer.begin(), max_it);
//
//   return index;
// }

int MCTSPlanner::getUCTBranchIndex(StateNode* node) {
  double det = log((double)node->numVisits_);

  vector<double> maximizer;
  int size = node->nodeVect_.size();
  for (int i = 0; i < size; ++i) {
    double val = node->nodeVect_[i]->avgReturn_;
    val += ucbScalar_ * sqrt(det / (double)node->nodeVect_[i]->numVisits_);
    maximizer.push_back(val);
  }
  vector<double>::iterator max_it = std::max_element(maximizer.begin(), maximizer.end());
  int index = std::distance(maximizer.begin(), max_it);

  return index;
}

void MCTSPlanner::updateValues(StateNode* node, double mcReturn) {
  if (mcReturn != 0) {
    cout << "[dbg] MCTS: non-zero MC Return: " << mcReturn << endl;
  }
  double totalReturn(mcReturn);
  node->numVisits_++;
  while (node->parentAct_ != NULL) { //back until root is reached
    ActionNode* parentAct = node->parentAct_;
    parentAct->numVisits_++;
    totalReturn *= gamma_;
    totalReturn += modifyReward(node->reward_);
    parentAct->avgReturn_ += (totalReturn - parentAct->avgReturn_) / parentAct->numVisits_;
    node = parentAct->parentState_;
    node->numVisits_++;
  }
}

double MCTSPlanner::MC_Sampling(StateNode* node, int depth) {
  double mcReturn(leafValue_);
  sim_->setHistory(node->stateHistory(HISTORY_SIZE));
  sim_->setState(node->state_);
  double discnt = 1;
  for (int i = 0; i < depth; i++) {
    if (sim_->isTerminal()) {
      mcReturn += endEpisodeValue_;
      break;
    }
    vector<SimAction*>& actions = sim_->getActions();
    int actID = rand() % actions.size();
    double r = sim_->act(actions[actID]);
    mcReturn += discnt * modifyReward(r);
    discnt *= gamma_;
  }
  return mcReturn;
}


double MCTSPlanner::MC_Sampling(StateNode* node) {
  double mcReturn(endEpisodeValue_);
  sim_->setHistory(node->stateHistory(HISTORY_SIZE));
  sim_->setState(node->state_);
  double discnt = 1;
  while (!sim_->isTerminal()) {
    vector<SimAction*>& actions = sim_->getActions();
    int actID = rand() % actions.size();
    double r = sim_->act(actions[actID]);
    mcReturn += discnt * modifyReward(r);
    discnt *= gamma_;
  }
  return mcReturn;
}

// memory management codes
void MCTSPlanner::pruneState(StateNode * state) {
  int sizeNode = state->nodeVect_.size();
  for (int i = 0; i < sizeNode; ++i) {
    ActionNode* tmp = state->nodeVect_[i];
    pruneAction(tmp);
  }
  state->nodeVect_.clear();
  delete state;
}

void MCTSPlanner::pruneAction(ActionNode * act) {
  int sizeNode = act->stateVect_.size();
  for (int i = 0; i < sizeNode; ++i) {
    StateNode* tmp = act->stateVect_[i];
    pruneState(tmp);
  }
  act->stateVect_.clear();
  delete act;
}

// Prune all ancestors, except keeps `historySize` of them to use with
// the state prediction model.
void MCTSPlanner::pruneAncestors(int historySize) {
  int cnt = 0;
  StateNode* current = root_;
  while (cnt < historySize && current->parentAct_ != NULL) {
    assert(current->parentAct_->parentState_ != NULL);
    cnt++;
    current = current->parentAct_->parentState_;
  }
  // XXX: Properly prune
  // pruneState(current);
}

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
  sim_->setHistory(root_->stateHistory(HISTORY_SIZE));
}

// XXX: Finish this
void MCTSPlanner::prune(SimAction* act, int historySize) {// check whether the root is terminal or not
  /*
  StateNode * nextRoot = NULL;
  int size = root_->nodeVect_.size();
  for (int i = 0 ; i < size; ++i) {
    if (act->equal(root_->actVect_[i])) {
      // XXX: What to do with this?
      assert(root_->nodeVect_[i]->stateVect_.size() == 1);
      nextRoot = root_->nodeVect_[i]->stateVect_[0];

      ActionNode* tmp = root_->nodeVect_[i];
      delete tmp;
    } else {
      ActionNode* tmp = root_->nodeVect_[i];
      pruneAction(tmp);
    }
  }

  if (historySize > 0) {
    pruneAncestors(historySize);
  }

  assert(nextRoot != NULL);

  delete root_;
  root_ = nextRoot;
  root_->parentAct_ = NULL;
  */
}

/*
bool MCTSPlanner::testRoot(State* _state, double _reward, bool _isTerminal) {
  return root_ != NULL && (root_->reward_ == _reward) && (root_->isTerminal_ == _isTerminal) && root_->state_->equal(_state);
}


void MCTSPlanner::testDeterministicProperty() {
  if (testDeterministicPropertyState(root_)) {
    cout << "Deterministic Property Test passed!" << endl;
  } else {
    cout << "Error in Deterministic Property  Test!" << endl;
    exit(0);
  }
}

bool MCTSPlanner::testDeterministicPropertyState(StateNode* state) {
  int actSize = state->nodeVect_.size();
  for (int i = 0; i < actSize; ++i) {
    if (!testTreeStructureAction(state->nodeVect_[i])) {
      return false;
    }
  }
  return true;
}

bool MCTSPlanner::testDeterministicPropertyAction(ActionNode* action) {

  int stateSize = action->stateVect_.size();
  if (stateSize != 1) {
    cout << "Error in Deterministic Property Test!" << endl;
    return false;
  }

  for (int i = 0; i < stateSize; ++i) {
    if (!testTreeStructureState(action->stateVect_[i])) {
      return false;
    }
  }
  return true;
}

// visit number checkings
// avg value checkings
void MCTSPlanner::testTreeStructure() {
  if (testTreeStructureState(root_)) {
    cout << "Tree Structure Test passed!" << endl;
  } else {
    cout << "Error in Tree Structure Test!" << endl;
    exit(0);
  }
}

bool MCTSPlanner::testTreeStructureState(StateNode* state) {
  // numVisits testing
  // n(s) = \sum_{a} n(s,a) + 1 (one offset due to first sample)
  int actVisitCounter = 0;
  int actSize = state->nodeVect_.size();
  for (int i = 0; i < actSize; ++i) {
    actVisitCounter += state->nodeVect_[i]->numVisits_;
  }
  if ((actVisitCounter + 1 != state->numVisits_) && (! state->isTerminal_) ) {
    cout << "n(s) = sum_{a} n(s,a) + 1 failed !" << "\nDiff: " << (actVisitCounter + 1 - state->numVisits_) << "\nact: " << (actVisitCounter + 1) << "\nState: " << state->numVisits_ << "\nTerm: " << (state->isTerminal_ ? "true" : "false" ) << "\nState: " << endl;
    state->state_->print();
    cout << endl;
    return false;
  }
  for (int i = 0; i < actSize; ++i) {
    if (!testTreeStructureAction(state->nodeVect_[i])) {
      return false;
    }
  }
  return true;
}

bool MCTSPlanner::testTreeStructureAction(ActionNode* action) {
  // numVisits testing
  // n(s,a) = \sum n(s')
  int stateVisitCounter = 0;
  int stateSize = action->stateVect_.size();
  for (int i = 0; i < stateSize; ++i) {
    stateVisitCounter += action->stateVect_[i]->numVisits_;
  }
  if (stateVisitCounter != action->numVisits_) {
    cout << "n(s,a) = sum n(s') failed !" << endl;
    return false;
  }

  // avg
  // Q(s,a) = E {r(s') + gamma * sum pi(a') Q(s',a')}
  // Q(s,a) = sum_{s'} n(s') / n(s,a) * ( r(s') + gamma * sum_{a'} (n (s',a') * Q(s',a') + first) / n(s'))
  double value = 0;
  for (int i = 0; i < stateSize; ++i) {
    StateNode* next = action->stateVect_[i];
    double w = next->numVisits_ / (double) action->numVisits_;
    double nextValue = next->firstMC_;
    int nextActSize = next->nodeVect_.size();
    for (int j = 0; j < nextActSize; ++j) {
      nextValue += next->nodeVect_[j]->numVisits_ * next->nodeVect_[j]->avgReturn_;
    }
    nextValue = (nextValue) / (double) next->numVisits_ * gamma_;
    nextValue += next->reward_;
    value += w * nextValue;
  }
  if ( (action->avgReturn_ - value) * (action->avgReturn_ - value) > 1e-10 ) {
    cout << "value constraint failed !" << "avgReturn=" << action->avgReturn_ << " value=" << value << endl;
    return false;
  }

  for (int i = 0; i < stateSize; ++i) {
    if (!testTreeStructureState(action->stateVect_[i])) {
      return false;
    }
  }
  return true;
}
*/
