#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <string>

#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/backend/backend.hpp"

namespace auto_aim
{
class Classifier
{
public:
  explicit Classifier(const std::string & config_path);

  void classify(Armor & armor);

  void backend_classify(Armor & armor);

private:
  cv::dnn::Net net_;
  Backend backend_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP