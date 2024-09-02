
#include "imgprocess.hpp"

void drawBBox(cv::Mat &img, cv::Rect &bbox)
{
    cv::Scalar color{192., 23., 23.};
    cv::rectangle(
        img,
        bbox,
        color
        );
}

void drawKeypoints(cv::Mat &img, Pose &pose)
{
    cv::Scalar kpColor{23., 192., 192.};
    cv::Scalar lineColor{168., 168., 64.};
    for (auto kp : pose.keypoints)
    {
        cv::circle(img, cv::Point2f(kp[0], kp[1]), 3, kpColor, cv::LineTypes::FILLED);
    }
    for (auto conn : pose.keypointConns)
    {
        auto kpt1 = pose.keypoints[conn[0]];
        auto kpt2 = pose.keypoints[conn[1]];
        cv::Point2f pt1(kpt1[0], kpt1[1]);
        cv::Point2f pt2(kpt2[0], kpt2[1]);
        cv::line(img, pt1, pt2, lineColor, 1, cv::LineTypes::LINE_4);
    }
}
