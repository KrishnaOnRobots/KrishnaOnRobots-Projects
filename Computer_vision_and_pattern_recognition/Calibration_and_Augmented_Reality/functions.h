//Coded by krishna Prasath Senthil Kumaran
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<Vec3f> findWorldCoordinates(Size patternSize);
bool extractChessboardCorners(Mat& frame, Size patternSize, vector<Point2f>& corners);
void printMatrix(Mat& m);
vector<Vec3f> constructObjectPoints();
void drawObjects(Mat& frame, vector<Point2f> p);
void highlightOutsideCorners(Mat& frame, vector<Vec3f> points, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs);
void drawVirtualObject(Mat& frame, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs);