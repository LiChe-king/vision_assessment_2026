#include "aimer.hpp"
#include <cmath>

namespace auto_aim
{

Aimer::Aimer()
{
  covariance_.setIdentity();
  covariance_ *= 0.1;

  process_noise_.setIdentity();
  process_noise_.topLeftCorner<3, 3>() *= 0.02;
  process_noise_.bottomRightCorner<3, 3>() *= 0.5;

  measurement_noise_.setIdentity();
  measurement_noise_ *= 0.03;
}

AimResult Aimer::update(const Armor & armor, double timestamp)
{
  const auto measurement = armor.xyz_in_gimbal;

  if (!initialized_ || armor.name != tracked_name_) {
    reset(armor, timestamp);
  } else {
    double dt = timestamp - last_timestamp_;
    if (dt <= 0.0 || dt > 0.5) {
      reset(armor, timestamp);
    } else {
      predict(dt);
      correct(measurement);
      last_timestamp_ = timestamp;
    }
  }

  const auto predicted = predictPosition(kPredictTime);
  const double xy_distance = std::hypot(predicted.x(), predicted.y());

  AimResult result;
  result.observed_position = measurement;
  result.predicted_position = predicted;
  result.yaw = std::atan2(predicted.y(), predicted.x());
  result.pitch = std::atan2(predicted.z(), xy_distance);
  result.valid = true;
  return result;
}

void Aimer::reset(const Armor & armor, double timestamp)
{
  state_.setZero();
  state_.head<3>() = armor.xyz_in_gimbal;

  covariance_.setIdentity();
  covariance_.topLeftCorner<3, 3>() *= 0.05;
  covariance_.bottomRightCorner<3, 3>() *= 1.0;

  tracked_name_ = armor.name;
  last_timestamp_ = timestamp;
  initialized_ = true;
}

void Aimer::predict(double dt)
{
  Eigen::Matrix<double, 6, 6> f = Eigen::Matrix<double, 6, 6>::Identity();
  f(0, 3) = dt;
  f(1, 4) = dt;
  f(2, 5) = dt;

  state_ = f * state_;
  covariance_ = f * covariance_ * f.transpose() + process_noise_ * dt;
}

void Aimer::correct(const Eigen::Vector3d & measurement)
{
  Eigen::Matrix<double, 3, 6> h = Eigen::Matrix<double, 3, 6>::Zero();
  h(0, 0) = 1.0;
  h(1, 1) = 1.0;
  h(2, 2) = 1.0;

  Eigen::Vector3d residual = measurement - h * state_;
  Eigen::Matrix3d s = h * covariance_ * h.transpose() + measurement_noise_;
  Eigen::Matrix<double, 6, 3> k = covariance_ * h.transpose() * s.inverse();

  state_ = state_ + k * residual;
  covariance_ = (Eigen::Matrix<double, 6, 6>::Identity() - k * h) * covariance_;
}

Eigen::Vector3d Aimer::predictPosition(double dt) const
{
  return state_.head<3>() + state_.tail<3>() * dt;
}

}  // namespace auto_aim