#ifndef __TF_UTIL_H__
#define __TF_UTIL_H__

// #include "tensorflow/cc/ops/const_op.h"
// #include "tensorflow/core/protobuf/meta_graph.pb.h"
#include <ale_interface.hpp>
#include "tensorflow/core/public/session.h"
#include "constants.h"

using namespace std;
using namespace tensorflow;

void AppendOnehotAction(Vec* v, int chosen, int num_actions);

const Vec AleScreenToObjState(const ale::ALEScreen& screen);

class TFModel {
  public:
    TFModel() = delete;
    TFModel(const string&);
    // Disable copy constructor and assignment operator since they can mess
    // things up because of unique_ptr member variable.
    TFModel(const TFModel&) = delete;
    TFModel& operator= (const TFModel&) = delete;
    virtual ~TFModel();

    // TODO
    // Status Load(const string&);

    void RunVector(const Vec& input,
                   const string& placeholderName,
                   const string& outputName,
                   Tensor* output);

    void RunMatrix(const Vec& input,
                   const TensorShape& shape,
                   const string& placeholderName,
                   const string& outputName,
                   Tensor* output);

    std::unique_ptr<Session> session_;
    std::string model_prefix_;

  private:
    void RunHelper(const vector<pair<string, Tensor>>&, 
                   const string& outputName,
                   Tensor* output);
};

#endif
