
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
    cv::Scalar kpColor{23., 192., 192.}, kpColorLeft{192., 46., 46.}, kpColorRight{46., 46., 192.};
    cv::Scalar lineColor{168., 168., 64.};
    int radius = 5;
    int thicknessInvisible = 1;
    for (int i = 0; i < pose.keypoints.size(); i++)
    {
        float x, y, visibility;
        x = pose.keypoints.at(i).at(0);
        y = pose.keypoints.at(i).at(1);
        visibility = pose.keypoints.at(i).at(2);
        if (i % 2 == 1 && i > 0)
        {
            if (visibility >= 0.5)
                cv::circle(img, cv::Point2f(x, y), radius, kpColorLeft, cv::LineTypes::FILLED);
            else
                cv::circle(img, cv::Point2f(x, y), radius, kpColorLeft, thicknessInvisible, cv::LineTypes::LINE_4);
        }
        else if (i % 2 == 0 && i > 0)
        {
            if (visibility >= 0.5)
                cv::circle(img, cv::Point2f(x, y), radius, kpColorRight, cv::LineTypes::FILLED);
            else
                cv::circle(img, cv::Point2f(x, y), radius, kpColorRight, thicknessInvisible, cv::LineTypes::LINE_4);
        }
        else
        {
            if (visibility >= 0.5)
                cv::circle(img, cv::Point2f(x, y), radius, kpColor, cv::LineTypes::FILLED);
            else
                cv::circle(img, cv::Point2f(x, y), radius, kpColor, thicknessInvisible, cv::LineTypes::LINE_4);
        }
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
