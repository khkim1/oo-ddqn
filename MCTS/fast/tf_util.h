#ifndef __TF_UTIL_H__
#define __TF_UTIL_H__

#include <ale_interface.hpp>
#include "tensorflow/core/public/session.h"
#include "constants.h"

namespace oodqn {

void AppendOnehotAction(Vec* v, int chosen, int num_actions);

class TFModel {
  public:
    TFModel() = delete;
    TFModel(const std::string&);
    // Disable copy constructor and assignment operator since they can mess
    // things up because of unique_ptr member variable.
    TFModel(const TFModel&) = delete;
    TFModel& operator= (const TFModel&) = delete;
    inline std::string getModelPrefix() const {
      return model_prefix_;
    }
    virtual ~TFModel();

    // TODO
    // Status Load(const std::string&);

    void RunVector(const Vec& input,
                   const std::string& placeholderName,
                   const std::string& outputName,
                   tensorflow::Tensor* output);

    void RunMatrix(const Vec& input,
                   const tensorflow::TensorShape& shape,
                   const std::string& placeholderName,
                   const std::string& outputName,
                   tensorflow::Tensor* output);

    std::unique_ptr<tensorflow::Session> session_;
    std::string model_prefix_;

  private:
    void RunHelper(
        const std::vector<std::pair<std::string, tensorflow::Tensor>>&,
        const std::string& outputName, tensorflow::Tensor* output);
};

}  // namespace oodqn

#endif
