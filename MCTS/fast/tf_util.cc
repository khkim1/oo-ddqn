#include "tf_util.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace tensorflow;

// `chosen` is 0-based
void AppendOnehotAction(vector<float>* v, int chosen, int num_actions) {
  for (int i = 0; i < num_actions; i++) {
    if (i == chosen) {
      v->push_back(1.);
    }
    else{
      v->push_back(0.);
    }
  }
}

// TODO: Converts the output of ALE game screen into object state vector.
Vec AleScreenToObjState(const vector<unsigned char>& output_rgb_buffer) {
  Vec out({ -0.77987421,   0.10062893,   1.,
            -0.02515723,   0.03773585,   1.,
             0.77987421,   0.01257862,   1. });
  return out;
}

TFModel::TFModel(const string& path_prefix) {
  model_prefix_ = path_prefix;
  const string meta_graph_path = path_prefix + ".meta";

  session_.reset(NewSession(SessionOptions()));
  if (session_ == nullptr) {
    throw runtime_error("Could not create Tensorflow session.");
  }

  Status status;

  // Load the protobuf graph def.
  MetaGraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), meta_graph_path, &graph_def);
  if (!status.ok()) {
      throw runtime_error("Error reading graph definition from "
          + meta_graph_path + ": " + status.ToString());
  }

  // Add the graph to the session
  status = session_->Create(graph_def.graph_def());
  if (!status.ok()) {
      throw runtime_error("Error creating graph: " + status.ToString());
  }

  // Read weights from the checkpoint
  Tensor path_tensor(DT_STRING, TensorShape());
  path_tensor.scalar<std::string>()() = path_prefix;
  status = session_->Run({
      {graph_def.saver_def().filename_tensor_name(), path_tensor}},
      {}, {graph_def.saver_def().restore_op_name()}, nullptr);
  if (!status.ok()) {
      throw runtime_error("Error loading checkpoint from "
          + path_prefix + ": " + status.ToString());
  }
}

TFModel::~TFModel() {
  // Release session.
  if (!session_->Close().ok()) {
    cout << "Couldn't close session properly.";
  }
  session_.reset();
}

void TFModel::RunHelper(const vector<pair<string, Tensor>>& feed, 
                        const string& outputName,
                        Tensor* output) {
  Status status;
  vector<Tensor> outputs;
  status = session_->Run(feed, {outputName}, {}, &outputs);
  if (status.ok() && outputs.size() == 1) {
    *output = outputs[0];
  }
  else {
    cout << "Error: " << status.ToString() << endl;
    throw runtime_error("Error: " + status.ToString());
  }
}


void TFModel::RunVector(const vector<float>& input,
                        const string& placeholderName,
                        const string& outputName,
                        Tensor* output) {
  int size = input.size();
  Tensor val(DT_FLOAT, TensorShape({size}));
  for (int i = 0; i < size; ++i) {
    val.vec<float>()(i) = input[i];
  }
  const vector<pair<string, Tensor>> feed = {
    { placeholderName, val }
  };

  RunHelper(feed, outputName, output);
}

void TFModel::RunMatrix(const vector<float>& input,
                          const TensorShape& shape,
                          const string& placeholderName,
                          const string& outputName,
                          Tensor* output) {
  const int d0 = shape.dim_size(0);
  const int d1 = shape.dim_size(1);
  
  Tensor val(DT_FLOAT, shape);
  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      val.matrix<float>()(i, j) = input[i*d1 + j];
    }
  }
  // TODO: placeholder names?
  const vector<pair<string, Tensor>> feed = {
    { placeholderName, val }
  };

  RunHelper(feed, outputName, output);
}

