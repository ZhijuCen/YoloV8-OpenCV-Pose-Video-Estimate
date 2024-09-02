
#pragma once
#include <opencv2/opencv.hpp>
#include "pose.hpp"

void drawBBox(cv::Mat &img, cv::Rect &bbox);

void drawKeypoints(cv::Mat &img, Pose &pose);
