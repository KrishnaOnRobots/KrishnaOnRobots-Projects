/*Created by Krishna Prasath Senthil Kumaran*/
#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

std::vector<float> matConvVector(cv::Mat& mat);
std::vector<float> baselineMatching(Mat image);
std::vector<float> histogram(cv::Mat& src);
std::vector<float> multiHistogram(cv::Mat& src);
std::vector<float> texture(cv::Mat& src);
std::vector<float> textureAndColor(cv::Mat& src);
std::vector<float> gaborTexture(cv::Mat& src);
cv::Mat sobelX(cv::Mat& src);
cv::Mat sobelY(cv::Mat& src);
cv::Mat magnitude(cv::Mat& src);
cv::Mat orientation(cv::Mat& src);
cv::Mat getMiddle(cv::Mat& src);
