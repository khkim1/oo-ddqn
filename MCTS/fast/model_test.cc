#include "tf_util.h"

using namespace std;
using namespace tensorflow;

int main() {
  Status status;
  TFModel model("../test_model/test_model");
  Tensor out;
  status = model.Run({1,2,3}, "pred", &out);

  // Tensor x_val(DT_FLOAT, TensorShape({3}));
  // x_val.vec<float>()(0) = 1.0f;
  // x_val.vec<float>()(1) = 2.0f;
  // x_val.vec<float>()(2) = 3.0f;
  // const vector<pair<string, Tensor>> feed = { {"ph_x", x_val} };
  // vector<string> outputOps = {"pred", "wb:0"};
  // vector<Tensor> results;
  // status = sess->Run(feed, outputOps, {}, &results);
  cout << out.DebugString() << endl;
  for (int i = 0; i < out.dim_size(0); ++i) {
    cout << out.vec<float>()(i) << ",";
  }
}

