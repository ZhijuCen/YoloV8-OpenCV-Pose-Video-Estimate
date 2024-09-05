
#pragma once
#include <opencv2/opencv.hpp>
#include "pose.hpp"


class PoseNet {
public:
    PoseNet(const std::string &onnxModelPath, const cv::Size2f &modelInputShape = cv::Size2f(640.f, 640.f), const bool useCuda = true);
    std::vector<Pose> operator()(const cv::Mat &inputImg);

    PoseNet& setNmsThres(float value) { nmsThres = value; return *this; }
    PoseNet& setScoreThres(float value) { scoreThres = value; return *this; }
private:
    cv::dnn::Net net;
    std::string modelPath{};
    bool runWithCuda{false};
    cv::Size2f inputShape{640.f, 640.f};
    float nmsThres{0.8};
    float scoreThres{0.4};

    void loadOnnxNet();
    cv::Mat letterBox(const cv::Mat &source);
};
