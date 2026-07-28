#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include "armor.hpp"

namespace auto_aim
{

struct AimResult
{
  Eigen::Vector3d observed_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d predicted_position = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double pitch = 0.0;
  bool valid = false;
};

class Aimer
{
public:
  Aimer();
  AimResult update(const Armor & armor, double timestamp);

private:
  static constexpr double kPredictTime = 0.5;

  bool initialized_ = false;
  double last_timestamp_ = 0.0;
  ArmorName tracked_name_ = ArmorName::not_armor;

  // x = [px, py, pz, vx, vy, vz]
  Eigen::Matrix<double, 6, 1> state_ = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity();

  Eigen::Matrix<double, 6, 6> process_noise_ = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix3d measurement_noise_ = Eigen::Matrix3d::Identity();

  void reset(const Armor & armor, double timestamp);
  void predict(double dt);
  void correct(const Eigen::Vector3d & measurement);
  Eigen::Vector3d predictPosition(double dt) const;
};

}  // namespace auto_aim

#endif