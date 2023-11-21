//Coded by krishna Prasath Senthil Kumaran
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "functions.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	//Initializin parameters
	Size chessboardPatternSize(9, 6);
	Mat cameraMatrix, chessboardDistCoeff;
	vector<Mat> chessboardR, chessboardT;

	vector<Vec3f> chessBoardPoints;
	vector<vector<Point2f> > chessboardCornerList;
	vector<vector<Vec3f> > chessboardPointList;
	int min_calibration_frames = 5;

	//Enabling video feed
	VideoCapture* capdev;
	capdev = new VideoCapture(1);
	if (!capdev->isOpened()) {
		cout << "Problem with video device\n";
		return -1;
	}

	namedWindow("Video", 1);

	Mat frame;

	chessBoardPoints = findWorldCoordinates(chessboardPatternSize);

	while (1) {

		*capdev >> frame;
		if (frame.empty()) {
			cout << "No video input...\n";
			break;
		}

		resize(frame, frame, Size(), 0.75, 0.75);

		Mat outputFram = frame.clone(); // the frame displayed

		char key = waitKey(10); // see if there is a waiting keystroke for the video

		// extract chessboard corners
		vector<Point2f> chessboardCorners; // the image points found by extractChessboardCorners()
		bool foundChessboardCorners = extractChessboardCorners(frame, chessboardPatternSize, chessboardCorners);
		if (foundChessboardCorners) { // display the chessboard corners
			drawChessboardCorners(outputFram, chessboardPatternSize, chessboardCorners, foundChessboardCorners);
		}


		if (key == 's') { // select calibration images for chessboard
			if (foundChessboardCorners) {
				cout << "Capturing Calibration images..." << endl;
				// add the vector of corners found by findChessCorners() into a cornerList
				chessboardCornerList.push_back(chessboardCorners);
				// add the vector of real-world points into a pointList
				chessboardPointList.push_back(chessBoardPoints);
			}
			else {
				cout << "No chessboard corners found" << endl;
			}
		}
		else if (key == 'c') { // calibrate the camera for chessboard
			if (chessboardPointList.size() < min_calibration_frames) { // not enough calibration frames
				cout << "Not enough calibration frames. 5 or more needed." << endl;
			}
			else {
				cout << "Camera Calibrated" << endl;
				// calibrate the camera
				double chessboardError = calibrateCamera(chessboardPointList, chessboardCornerList, Size(frame.rows, frame.cols),
					cameraMatrix, chessboardDistCoeff, chessboardR, chessboardT);

				// print out the intrinsic parameters and the final re-projection error
				cout << "Chessboard Camera Matrix: " << endl;
				printMatrix(cameraMatrix);
				cout << "Chessboard Distortion Coefficients: " << endl;
				printMatrix(chessboardDistCoeff);
				cout << "Chessboard Re-projection Error: " << chessboardError << endl;
			}
		}
		else if (key == 'r') { // reset the calibration for chessboard
			cout << "reset calibration" << endl;
			chessboardCornerList.clear();
			chessboardPointList.clear();
		}

		if (chessboardDistCoeff.rows != 0) {
			// extractChessboardCorners of current frame
			vector<Point2f> currCorners; // the image points found by findChessboardCorners
			bool foundCurrCorners = extractChessboardCorners(frame, chessboardPatternSize, currCorners);

			if (foundCurrCorners) {
				Mat rvec, tvec; // output arrays for solvePnP()
				bool status = solvePnP(chessBoardPoints, currCorners, cameraMatrix, chessboardDistCoeff, rvec, tvec);

				if (status) { 
					// project outside corners
					highlightOutsideCorners(outputFram, chessBoardPoints, rvec, tvec, cameraMatrix, chessboardDistCoeff);

					//printing the Rotational and translational data
					cout << "Rotational data: \n" << rvec << endl;
					cout << "Translational data: \n" << tvec << endl;

					// project a virtual object
					drawVirtualObject(outputFram, rvec, tvec, cameraMatrix, chessboardDistCoeff);
				}
			}
		}

		imshow("Video", outputFram);

		if (key == 'q') { // press 'q' to quit the system
			break;
		}
	}
	return 0;
}