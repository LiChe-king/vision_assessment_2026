#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

namespace auto_aim
{

Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  auto camera_matrix = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs = yaml["distort_coeffs"].as<std::vector<double>>();

  camera_matrix_ = cv::Mat(3, 3, CV_64F, camera_matrix.data()).clone();
  distort_coeffs_ =
    cv::Mat(1, static_cast<int>(distort_coeffs.size()), CV_64F, distort_coeffs.data()).clone();

  covariance_.setIdentity();
  covariance_ *= 0.1;

  process_noise_.setIdentity();
  process_noise_.topLeftCorner<4, 4>() *= 0.02;
  process_noise_.bottomRightCorner<4, 4>() *= 0.6;

  measurement_noise_.setIdentity();
  measurement_noise_.diagonal() << 0.03, 0.03, 0.03, 0.08;
}

AimResult Aimer::update(const Armor & armor, double timestamp)
{
  const auto measurement = makeMeasurement(armor);

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

  const auto predicted_position = predictPosition(kPredictTime);
  const auto predicted_yaw = predictYaw(kPredictTime);
  const double xy_distance = std::hypot(predicted_position.x(), predicted_position.y());

  AimResult result;
  result.observed_position = measurement.head<3>();
  result.predicted_position = predicted_position;
  result.observed_yaw = measurement(3);
  result.predicted_yaw = predicted_yaw;
  result.yaw = std::atan2(predicted_position.y(), predicted_position.x());
  result.pitch = std::atan2(predicted_position.z(), xy_distance);
  result.valid = true;
  return result;
}

void Aimer::drawReprojection(cv::Mat & image, const AimResult & result) const
{
  if (!result.valid || image.empty()) {
    return;
  }

  for (int i = 0; i < 4; ++i) {
    const bool big = (i % 2 == 0);
    auto corners = makeArmorCorners(result.predicted_position, result.predicted_yaw, i, big);

    std::vector<cv::Point2f> image_points;
    cv::projectPoints(
      corners,
      cv::Vec3d(0.0, 0.0, 0.0),
      cv::Vec3d(0.0, 0.0, 0.0),
      camera_matrix_,
      distort_coeffs_,
      image_points
    );

    const cv::Scalar color = i == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

    for (int j = 0; j < 4; ++j) {
      cv::line(image, image_points[j], image_points[(j + 1) % 4], color, 2);
    }
  }
}

std::vector<cv::Point2f> Aimer::predictArmorPoints2D(
  const Armor & armor,
  const AimResult & result) const
{
  std::vector<cv::Point2f> predicted_center_2d;

  std::vector<cv::Point3f> point_3d = {
    cv::Point3f(
      static_cast<float>(result.predicted_position.x()),
      static_cast<float>(result.predicted_position.y()),
      static_cast<float>(result.predicted_position.z())
    )
  };

  cv::projectPoints(
    point_3d,
    cv::Vec3d(0.0, 0.0, 0.0),
    cv::Vec3d(0.0, 0.0, 0.0),
    camera_matrix_,
    distort_coeffs_,
    predicted_center_2d
  );

  if (predicted_center_2d.empty()) {
    return armor.points;
  }

  cv::Point2f old_center(0.0f, 0.0f);
  for (const auto & p : armor.points) {
    old_center += p;
  }
  old_center *= 0.25f;

  cv::Point2f offset = predicted_center_2d[0] - old_center;

  std::vector<cv::Point2f> predicted_points;
  predicted_points.reserve(armor.points.size());

  for (const auto & p : armor.points) {
    predicted_points.emplace_back(p + offset);
  }

  return predicted_points;
}

void Aimer::reset(const Armor & armor, double timestamp)
{
  const auto measurement = makeMeasurement(armor);

  state_.setZero();
  state_.head<3>() = measurement.head<3>();
  state_(3) = measurement(3);

  covariance_.setIdentity();
  covariance_.topLeftCorner<4, 4>() *= 0.05;
  covariance_.bottomRightCorner<4, 4>() *= 1.0;

  tracked_name_ = armor.name;
  last_timestamp_ = timestamp;
  initialized_ = true;
}

void Aimer::predict(double dt)
{
  Eigen::Matrix<double, 8, 8> f = Eigen::Matrix<double, 8, 8>::Identity();

  f(0, 4) = dt;
  f(1, 5) = dt;
  f(2, 6) = dt;
  f(3, 7) = dt;

  state_ = f * state_;
  state_(3) = normalizeAngle(state_(3));

  covariance_ = f * covariance_ * f.transpose() + process_noise_ * dt;
}

void Aimer::correct(const Eigen::Matrix<double, 4, 1> & measurement)
{
  Eigen::Matrix<double, 4, 8> h = Eigen::Matrix<double, 4, 8>::Zero();

  h(0, 0) = 1.0;
  h(1, 1) = 1.0;
  h(2, 2) = 1.0;
  h(3, 3) = 1.0;

  Eigen::Matrix<double, 4, 1> residual = measurement - h * state_;
  residual(3) = normalizeAngle(residual(3));

  Eigen::Matrix4d s = h * covariance_ * h.transpose() + measurement_noise_;
  Eigen::Matrix<double, 8, 4> k = covariance_ * h.transpose() * s.inverse();

  state_ = state_ + k * residual;
  state_(3) = normalizeAngle(state_(3));

  covariance_ = (Eigen::Matrix<double, 8, 8>::Identity() - k * h) * covariance_;
}

Eigen::Matrix<double, 4, 1> Aimer::makeMeasurement(const Armor & armor) const
{
  Eigen::Matrix<double, 4, 1> measurement;

  const double yaw = normalizeAngle(armor.ypr_in_gimbal.x());

  measurement.head<3>() = armorToVehicleCenter(armor.xyz_in_gimbal, yaw);
  measurement(3) = yaw;

  return measurement;
}

Eigen::Vector3d Aimer::armorToVehicleCenter(
  const Eigen::Vector3d & armor_position,
  double yaw) const
{
  Eigen::Vector3d normal(std::sin(yaw), 0.0, std::cos(yaw));
  return armor_position - normal * kVehicleRadius;
}

Eigen::Vector3d Aimer::predictPosition(double dt) const
{
  return state_.head<3>() + state_.segment<3>(4) * dt;
}

double Aimer::predictYaw(double dt) const
{
  return normalizeAngle(state_(3) + state_(7) * dt);
}

double Aimer::normalizeAngle(double angle)
{
  while (angle > CV_PI) {
    angle -= 2.0 * CV_PI;
  }

  while (angle < -CV_PI) {
    angle += 2.0 * CV_PI;
  }

  return angle;
}

std::vector<cv::Point3f> Aimer::makeArmorCorners(
  const Eigen::Vector3d & vehicle_center,
  double vehicle_yaw,
  int index,
  bool big) const
{
  const double yaw = vehicle_yaw + index * CV_PI / 2.0;
  const double width = big ? kBigArmorWidth : kSmallArmorWidth;

  const Eigen::Vector3d normal(std::sin(yaw), 0.0, std::cos(yaw));
  const Eigen::Vector3d right(std::cos(yaw), 0.0, -std::sin(yaw));
  const Eigen::Vector3d center = vehicle_center + normal * kVehicleRadius;

  std::vector<Eigen::Vector3d> corners = {
    center - right * (width / 2.0) - Eigen::Vector3d(0.0, kArmorHeight / 2.0, 0.0),
    center + right * (width / 2.0) - Eigen::Vector3d(0.0, kArmorHeight / 2.0, 0.0),
    center + right * (width / 2.0) + Eigen::Vector3d(0.0, kArmorHeight / 2.0, 0.0),
    center - right * (width / 2.0) + Eigen::Vector3d(0.0, kArmorHeight / 2.0, 0.0),
  };

  std::vector<cv::Point3f> cv_corners;
  cv_corners.reserve(corners.size());

  for (const auto & point : corners) {
    cv_corners.emplace_back(
      static_cast<float>(point.x()),
      static_cast<float>(point.y()),
      static_cast<float>(point.z())
    );
  }

  return cv_corners;
}

}  // namespace auto_aim