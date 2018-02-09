#ifndef __TF_UTIL_H__
#define __TF_UTIL_H__

// #include "tensorflow/cc/ops/const_op.h"
// #include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace tensorflow;

class TFModel {
  TFModel() = delete;
  TFModel(const string&);
  // Disable copy constructor and assignment operator since they can mess things
  // up because of unique_ptr member variable.
  TFModel(const TFModel&) = delete;
  TFModel& operator= (const TFModel&) = delete;
  virtual ~TFModel();

  Status Run(const vector<pair<string, Tensor> >& inputs,
             const vector<string>& output_tensor_names,
             const vector<string>& target_node_names,
             vector<Tensor>* outputs);

  private:
    std::unique_ptr<Session> session;
};

#endif
