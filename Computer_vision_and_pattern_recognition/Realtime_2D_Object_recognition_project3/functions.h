#pragma once
/*Code written by Krishna Prasath Senthil Kumaran*/
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat threshold(Mat& image);

Mat cleanImage(Mat& image);

Mat extractSegmentedRegions(Mat& image, Mat& labeledRegions, Mat& stats, Mat& centroids, vector<int>& topNLabels);

RotatedRect makeBoundingBox(Mat& region, double x, double y, double alpha);

void pointer(Mat& image, double x, double y, double alpha, Scalar color);

void drawBoundingBox(Mat& image, RotatedRect makeBoundingBox, Scalar color);

void HuMomentsCalculation(Moments moment, vector<double>& huMoments);

string nearestNeighbour(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, string distMetric);

string KNN_classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K, string distMetric);

string className_input(char c);