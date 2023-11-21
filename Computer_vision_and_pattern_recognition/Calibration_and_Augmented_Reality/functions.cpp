//Coded by krishna Prasath Senthil Kumaran
#include <iostream>
#include <opencv2/opencv.hpp>
#include "functions.h"

using namespace std;
using namespace cv;

/*
* This function calculates the world coordinates
*/
vector<Vec3f> findWorldCoordinates(Size patternSize) {
    
    vector<Vec3f> points;
    
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            Vec3f coordinates = Vec3f(j, -i, 0);
            points.push_back(coordinates);
        }
    }
    return points;
}

/*
* This function extracts the corners from the chessboard.
*/
bool extractChessboardCorners(Mat& frame, Size patternSize, vector<Point2f>& corners) {
    bool foundCorners = findChessboardCorners(frame, patternSize, corners);
    if (foundCorners) {
        Mat grayscale;
        cvtColor(frame, grayscale, COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
        Size subPixWinSize(10, 10);
        TermCriteria termCrit(TermCriteria::COUNT | TermCriteria::EPS, 1, 0.1);
        cornerSubPix(grayscale, corners, subPixWinSize, Size(-1, -1), termCrit);
    }
    return foundCorners;
}

void printMatrix(Mat& m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            cout << m.at<double>(i, j) << ", ";
        }
        cout << "\n";
    }
}

/*
 * Assigning the vertices for the virtual object which will be constructed later.
 */
vector<Vec3f> constructObjectPoints() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(1, -1, 1));
    objectPoints.push_back(Vec3f(1, -4, 1));
    objectPoints.push_back(Vec3f(4, -1, 1));
    objectPoints.push_back(Vec3f(4, -4, 1));
    objectPoints.push_back(Vec3f(2, -1, 3));
    objectPoints.push_back(Vec3f(2, -4, 3));
    objectPoints.push_back(Vec3f(5, -1, 3));
    objectPoints.push_back(Vec3f(5, -4, 3));
    
    Vec3f center(0, 0, 0);
    for (const auto& p : objectPoints) {
        center += p;
    }
    center /= static_cast<float>(objectPoints.size());


    // Second cube
    float x_offset = center[0] - 2.5; // offset to center second cube on top of first cube
    objectPoints.push_back(Vec3f(2 + x_offset, -2, 3));
    objectPoints.push_back(Vec3f(2 + x_offset, -3, 3));
    objectPoints.push_back(Vec3f(3 + x_offset, -2, 3));
    objectPoints.push_back(Vec3f(3 + x_offset, -3, 3));
    objectPoints.push_back(Vec3f(3 + x_offset, -2, 5));
    objectPoints.push_back(Vec3f(3 + x_offset, -3, 5));
    objectPoints.push_back(Vec3f(4 + x_offset, -2, 5));
    objectPoints.push_back(Vec3f(4 + x_offset, -3, 5));
    return objectPoints;
}

/*
 * The function draw lines to connect the given coordinate point and the faces of the shape is filled.
 */
void drawObjects(Mat& frame, vector<Point2f> p) {
    // Draw lines
    line(frame, p[0], p[1], Scalar(51, 255, 51), 2);
    line(frame, p[0], p[2], Scalar(51, 255, 51), 2);
    line(frame, p[1], p[3], Scalar(51, 255, 51), 2);
    line(frame, p[2], p[3], Scalar(51, 255, 51), 2);
    line(frame, p[4], p[6], Scalar(51, 255, 51), 2);
    line(frame, p[4], p[5], Scalar(51, 255, 51), 2);
    line(frame, p[5], p[7], Scalar(51, 255, 51), 2);
    line(frame, p[6], p[7], Scalar(51, 255, 51), 2);
    line(frame, p[0], p[4], Scalar(51, 255, 51), 2);
    line(frame, p[1], p[5], Scalar(51, 255, 51), 2);
    line(frame, p[2], p[6], Scalar(51, 255, 51), 2);
    line(frame, p[3], p[7], Scalar(51, 255, 51), 2);

    vector<Point> object1Points = { p[0], p[1], p[3], p[2] };
    fillConvexPoly(frame, object1Points, Scalar(255, 0, 0));

    vector<Point> object2Points = { p[8], p[9], p[11], p[10] };
    fillConvexPoly(frame, object2Points, Scalar(0, 0, 255));

    vector<Point> object3Points = { p[0], p[4], p[6], p[2] };
    fillConvexPoly(frame, object3Points, Scalar(255, 0, 0));

    vector<Point> object4Points = { p[4], p[5], p[7], p[6] };
    fillConvexPoly(frame, object4Points, Scalar(255, 0, 0));

    vector<Point> object5Points = { p[1], p[5], p[7], p[3] };
    fillConvexPoly(frame, object5Points, Scalar(255, 0, 0));

    vector<Point> object6Points = { p[8], p[12], p[14], p[10] };
    fillConvexPoly(frame, object6Points, Scalar(0, 0, 255));

    vector<Point> object7Points = { p[12], p[13], p[15], p[14] };
    fillConvexPoly(frame, object7Points, Scalar(0, 0, 255));

    vector<Point> object8Points = { p[9], p[13], p[15], p[11] };
    fillConvexPoly(frame, object8Points, Scalar(0, 0, 255));

    // Draw lines
    line(frame, p[8], p[9], Scalar(51, 255, 51), 2);
    line(frame, p[8], p[10], Scalar(51, 255, 51), 2);
    line(frame, p[9], p[11], Scalar(51, 255, 51), 2);
    line(frame, p[10],p[11], Scalar(51, 255, 51), 2);
    line(frame, p[12], p[14], Scalar(51, 255, 51), 2);
    line(frame, p[12], p[13], Scalar(51, 255, 51), 2);
    line(frame, p[13], p[15], Scalar(51, 255, 51), 2);
    line(frame, p[14], p[15], Scalar(51, 255, 51), 2);
    line(frame, p[8], p[12], Scalar(51, 255, 51), 2);
    line(frame, p[9], p[13], Scalar(51, 255, 51), 2);
    line(frame, p[10], p[14], Scalar(51, 255, 51), 2);
    line(frame, p[11], p[15], Scalar(51, 255, 51), 2);
}


/*
 * The function projects the 3D points to a 2D image frame, and draws the points on the image frame
 * The points are projected according to the results(camera matrix and intrinsic features) of camera calibration
 */
void highlightOutsideCorners(Mat& frame, vector<Vec3f> points, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs) {
    vector<Point2f> imagePts;
    projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, imagePts);
    int index[] = { 0, 8, 45, 53 };
    for (int i : index) {
        circle(frame, imagePts[i], 5, Scalar(51, 255, 51), 4);
    }
}

/*
 * The function constructs a set of points which are the vertices of a set of cubes using objectPoints()
 * The function then projects the 3D points to a 2D image frame and draws the points on the image frame
 * The function also draws lines between the points to form the cubes using drawObjects()
 * The points are projected according to the results(camera matrix and intrinsic features) of camera calibration
 */
void drawVirtualObject(Mat& frame, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs) {
    vector<Vec3f> objectPoints = constructObjectPoints();
    vector<Point2f> projectedPoints;
    projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
    for (int i = 0; i < projectedPoints.size(); i++) {
        circle(frame, projectedPoints[i], 1, Scalar(51, 255, 51), 4);
    }
    drawObjects(frame, projectedPoints);
}

