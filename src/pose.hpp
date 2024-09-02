
#pragma once
#include <opencv2/opencv.hpp>


struct Pose
{
    float score{0.0};
    cv::Rect bbox{};
    std::vector<std::vector<float>> keypoints{};

    const std::vector<std::vector<int64_t>> keypointConns = {
        {1, 3}, {2, 4}, {1, 2},
        {3, 5}, {4, 6}, {5, 6},
        {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {5, 11}, {6, 12}, {11, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };

    const std::map<uint32_t, std::string> keypointBodyNames = {
        {0, "nose"},
        {1, "eye_l"}, {2, "eye_r"},
        {3, "ear_l"}, {4, "ear_r"},
        {5, "shoulder_l"}, {6, "shoulder_r"},
        {7, "elbow_l"}, {8, "elbow_r"},
        {9, "wrist_l"}, {10, "wrist_r"},
        {11, "hip_l"}, {12, "hip_r"},
        {13, "knee_l"}, {14, "knee_r"},
        {15, "ankle_l"}, {16, "ankle_r"},
    };
};
