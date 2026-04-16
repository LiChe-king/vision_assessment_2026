#include "solver.hpp"

namespace auto_aim
{

Solver::Solver(const std::string & config_path)
{
  // TODO: 从 yaml 文件(如 configs/infantry.yaml) 中读取相机内参 camera_matrix_ (通常为 3x3) 和畸变参数 distort_coeffs_
}

void Solver::solve(Armor & armor) const
{
  // TODO: 根据装甲板 2D 提取到的特征点 (armor.points) 和真实大小 (ArmorType)
  // 调用 cv::solvePnP 算法计算出 rvec 和 tvec。
  // 注意将计算出的 translation 距离记录到 armor.xyz_in_gimbal 或自定义变量中，方便后续读取。(这里只做简单距离解算，忽略各种坐标系变换)。
}

}  // namespace auto_aim
