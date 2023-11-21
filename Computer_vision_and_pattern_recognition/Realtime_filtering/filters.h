#pragma once
#include <opencv2/core.hpp>

using namespace cv;

int greyscale(Mat& src, Mat& dst);
int blur5x5(Mat& src, Mat& dst);
int sobelX3x3(Mat& src, Mat& dst);
int sobelY3x3(Mat& src, Mat& dst);
int magnitude(Mat& sx, Mat& sy, Mat& dst);
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);
int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);
int brightness(Mat& src, Mat& dst);
int invertIntensity(Mat& src, Mat& dst);
int EdgeDetection(Mat& src, Mat& dst);
int ColorReductionFilter(Mat& src, Mat& dst, int divisor);