// #include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace tensorflow;

int main() {
  const string pathToGraph = "../test_model/test_model.meta";
  const string checkpointPath = "../test_model/test_model";

  // std::unique_ptr<Session> session(NewSession(SessionOptions()));
  auto session = NewSession(SessionOptions());
  if (session == nullptr) {
    throw runtime_error("Could not create Tensorflow session.");
  }

  Status status;

  // Read in the protobuf graph we exported
  MetaGraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
  if (!status.ok()) {
      throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
  }

  // Add the graph to the session
  status = session->Create(graph_def.graph_def());
  if (!status.ok()) {
      throw runtime_error("Error creating graph: " + status.ToString());
  }

  // Read weights from the saved checkpoint
  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = checkpointPath;
  status = session->Run(
          {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
          {},
          {graph_def.saver_def().restore_op_name()},
          nullptr);
  if (!status.ok()) {
      throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
  }

  // Run inference
  // auto root_scope = Scope::NewRootScope();
  // auto x_val = ops::Const(root_scope, { {1.f, 3.f, 5.f} });
  // ClientSession::FeedType feed;
  // feed.insert({"ph_x", x_val});

  Tensor x_val(DT_FLOAT, TensorShape({3}));
  x_val.vec<float>()(0) = 1.0f;
  x_val.vec<float>()(1) = 2.0f;
  x_val.vec<float>()(2) = 3.0f;
  const vector<pair<string, Tensor>> feed = { {"ph_x", x_val} };
  vector<string> outputOps = {"pred", "wb:0"};
  vector<Tensor> results;
  status = session->Run(feed, outputOps, {}, &results);
  if (!status.ok()) {
    throw runtime_error("Session Run() failed");
  }

  cout << "session->Run() returned " << results.size() << " items." << endl;

  for (int i = 0; i < results.size(); ++i) {
    cout << results[i].DebugString() << endl;
  }
  
  session->Close();

  return 0;
}


