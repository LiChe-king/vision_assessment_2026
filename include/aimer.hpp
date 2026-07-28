#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "armor.hpp"

namespace auto_aim
{

struct AimResult
{
  Eigen::Vector3d observed_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d predicted_position = Eigen::Vector3d::Zero();
  double observed_yaw = 0.0;
  double predicted_yaw = 0.0;
  double yaw = 0.0;
  double pitch = 0.0;
  bool valid = false;
};

class Aimer
{
public:
  explicit Aimer(const std::string & config_path);

  AimResult update(const Armor & armor, double timestamp);
  void drawReprojection(cv::Mat & image, const AimResult & result) const;

  std::vector<cv::Point2f> predictArmorPoints2D(
  const Armor & armor,
  const AimResult & result) const;

private:
  static constexpr double kPredictTime = 0.5;
  static constexpr double kVehicleRadius = 0.28;
  static constexpr double kArmorHeight = 0.055;
  static constexpr double kSmallArmorWidth = 0.135;
  static constexpr double kBigArmorWidth = 0.225;

  bool initialized_ = false;
  double last_timestamp_ = 0.0;
  ArmorName tracked_name_ = ArmorName::not_armor;

  // x = [xc, yc, zc, yaw, vxc, vyc, vzc, vyaw]
  Eigen::Matrix<double, 8, 1> state_ = Eigen::Matrix<double, 8, 1>::Zero();
  Eigen::Matrix<double, 8, 8> covariance_ = Eigen::Matrix<double, 8, 8>::Identity();

  Eigen::Matrix<double, 8, 8> process_noise_ = Eigen::Matrix<double, 8, 8>::Identity();
  Eigen::Matrix4d measurement_noise_ = Eigen::Matrix4d::Identity();

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;

  void reset(const Armor & armor, double timestamp);
  void predict(double dt);
  void correct(const Eigen::Matrix<double, 4, 1> & measurement);

  Eigen::Matrix<double, 4, 1> makeMeasurement(const Armor & armor) const;
  Eigen::Vector3d armorToVehicleCenter(const Eigen::Vector3d & armor_position, double yaw) const;
  Eigen::Vector3d predictPosition(double dt) const;
  double predictYaw(double dt) const;

  static double normalizeAngle(double angle);

  std::vector<cv::Point3f> makeArmorCorners(
    const Eigen::Vector3d & vehicle_center,
    double vehicle_yaw,
    int index,
    bool big) const;

};

}  // namespace auto_aim

#endif