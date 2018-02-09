#include "tf_util.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace tensorflow;

// Returns the session containing the loaded graph and weights for the model
// specified by the path prefix.

TFModel::TFModel(const string& path_prefix) {
  const string meta_graph_path = path_prefix + ".meta";

  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  if (session == nullptr) {
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
  status = session->Create(graph_def.graph_def());
  if (!status.ok()) {
      throw runtime_error("Error creating graph: " + status.ToString());
  }

  // Read weights from the checkpoint
  Tensor path_tensor(DT_STRING, TensorShape());
  path_tensor.scalar<std::string>()() = path_prefix;
  status = session->Run({
      {graph_def.saver_def().filename_tensor_name(), path_tensor}},
      {}, {graph_def.saver_def().restore_op_name()}, nullptr);
  if (!status.ok()) {
      throw runtime_error("Error loading checkpoint from "
          + path_prefix + ": " + status.ToString());
  }
}


Status Run(const vector<pair<string, Tensor> >& inputs,
           const vector<string>& output_tensor_names,
           const vector<string>& target_node_names,
           vector<Tensor>* outputs) {
  status = session->Run(inputs, output_tensor_names, target_node_names,
                        &outputs);
  return status;
}

TFModel::~TFModel() {
  // Release session.
  if (!session->Close().ok()) {
    cout << "Couldn't close session properly.";
  }
  session.reset();
}


/*
  Tensor x_val(DT_FLOAT, TensorShape({3}));
  x_val.vec<float>()(0) = 1.0f;
  x_val.vec<float>()(1) = 2.0f;
  x_val.vec<float>()(2) = 3.0f;
  const vector<pair<string, Tensor>> feed = { {"ph_x", x_val} };
  vector<string> outputOps = {"pred", "wb:0"};
  vector<Tensor> results;
  status = sess->Run(feed, outputOps, {}, &results);
  for (int i = 0; i < results.size(); ++i) {
    cout << results[i].DebugString() << endl;
  }
*/

