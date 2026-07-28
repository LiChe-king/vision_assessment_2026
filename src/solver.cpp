#include "solver.hpp"
#include <yaml-cpp/yaml.h>

namespace auto_aim
{

Solver::Solver(const std::string & config_path)
{
  // TODO: 从 yaml 文件(如 configs/infantry.yaml) 中读取相机内参 camera_matrix_ (通常为 3x3) 和畸变参数 distort_coeffs_
  auto yaml = YAML::LoadFile(config_path);

  auto camera_matrix = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs = yaml["distort_coeffs"].as<std::vector<double>>();

  camera_matrix_ = cv::Mat(3, 3, CV_64F, camera_matrix.data()).clone();
  distort_coeffs_ = cv::Mat(1, static_cast<int>(distort_coeffs.size()), CV_64F, distort_coeffs.data()).clone();
}
}

void Solver::solve(Armor & armor) const
{
  // TODO: 根据装甲板 2D 提取到的特征点 (armor.points) 和真实大小 (ArmorType)
  // 调用 cv::solvePnP 算法计算出 rvec 和 tvec。
  // 注意将计算出的 translation 距离记录到 armor.xyz_in_gimbal 或自定义变量中，方便后续读取。(这里只做简单距离解算，忽略各种坐标系变换)。
  if (armor.points.size() != 4) {
    return;
  }

  constexpr double small_armor_width = 0.135;
  constexpr double big_armor_width = 0.225;
  constexpr double armor_height = 0.055;

  double armor_width = armor.type == ArmorType::big ? big_armor_width : small_armor_width;

  std::vector<cv::Point3f> object_points = {
    {-static_cast<float>(armor_width / 2.0), -static_cast<float>(armor_height / 2.0), 0.0f},
    { static_cast<float>(armor_width / 2.0), -static_cast<float>(armor_height / 2.0), 0.0f},
    { static_cast<float>(armor_width / 2.0),  static_cast<float>(armor_height / 2.0), 0.0f},
    {-static_cast<float>(armor_width / 2.0),  static_cast<float>(armor_height / 2.0), 0.0f}
  };

  cv::Mat rvec;
  cv::Mat tvec;

  bool ok = cv::solvePnP(
    object_points,
    armor.points,
    camera_matrix_,
    distort_coeffs_,
    rvec,
    tvec,
    false,
    cv::SOLVEPNP_IPPE
  );

  if (!ok) {
    return;
  }

  armor.xyz_in_gimbal = Eigen::Vector3d(
    tvec.at<double>(0),
    tvec.at<double>(1),
    tvec.at<double>(2)
  );

}  // namespace auto_aim
