#include <iostream>
#include <opencv2/opencv.hpp>

#include "yolov5.hpp"
#include "solver.hpp"

using namespace auto_aim;
using namespace std;

int main(int argc, char** argv) {
  // 1. 初始化模型与求解器
  string model_path = "assets/0526.onnx";   // 或是0526.xml
  string config_path = "configs/infantry.yaml";
  string video_path = "assets/infantry.avi";

  // TODO: 初始化 YOLOv5 类

  // TODO: 初始化 Solver 类

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    cerr << "Failed to open video: " << video_path << endl;
    return -1;
  }

  cv::Mat frame;
  while (cap.read(frame)) {
    // 2. 调用模型识别
    std::vector<Armor> armors;
    // TODO: 使用 yolov5 提取 armors

    // 3. 将装甲板画出来并求解 PnP
    for (auto & armor : armors) {
      // TODO: 使用 solver.solve(armor) 进行位姿解算得到相对距离
      
      // TODO: 在图像上绘制装甲板（或者利用 yolo 类自带的画图函数）并可将距离通过 cv::putText 写在图像上
    }

    cv::imshow("Vision Assessment", frame);
    if (cv::waitKey(1) == 27) { // ESC 退出
      break;
    }
  }

  return 0;
}
