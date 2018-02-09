#ifndef __TF_UTIL_H__
#define __TF_UTIL_H__

// #include "tensorflow/cc/ops/const_op.h"
// #include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace tensorflow;

class TFModel {
  public:
    TFModel() = delete;
    TFModel(const string&);
    // Disable copy constructor and assignment operator since they can mess things
    // up because of unique_ptr member variable.
    TFModel(const TFModel&) = delete;
    TFModel& operator= (const TFModel&) = delete;
    virtual ~TFModel();

    // TODO
    // Status Load(const string&);

    // Status Run(const vector<pair<string, Tensor> >& inputs,
    //            const vector<string>& output_tensor_names,
    //            const vector<string>& target_node_names,
    //            vector<Tensor>* outputs);

    Status Run(const vector<float>& input,
               const string& output_name,
               Tensor* output);

    std::unique_ptr<Session> session_;
    std::string model_prefix_;
};

#endif
