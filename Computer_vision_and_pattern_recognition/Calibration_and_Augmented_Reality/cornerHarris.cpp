//Coded by krishna Prasath Senthil Kumaran
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "functions.h"

using namespace std;
using namespace cv;

/*
 * This program will read a video stream
 * detect corners in the video using cornerHarris(),
 * and then display the corners on the video
 */
int main() {
    int blockSize = 2;
    int kSize = 3;
    double k = 0.04;

    // open the video device
    VideoCapture* capdev;
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }

    // identify window
    namedWindow("Video", 1);

    Mat frame;

    while (true) {
        *capdev >> frame; // get a new frame from the camera
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }
        // resize the frame to 1/2 of the original size
        resize(frame, frame, Size(), 0.5, 0.5);

        char key = waitKey(10);

        // run Harris corner detector
        Mat grayscale;
        cvtColor(frame, grayscale, COLOR_BGR2GRAY);
        Mat dst = Mat::zeros(grayscale.size(), CV_32FC1);
        cornerHarris(grayscale, dst, blockSize, kSize, k);

        double min, max;
        cv::minMaxLoc(dst, &min, &max);
        float thresh = 0.1 * max;
        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                if (dst.at<float>(i, j) > thresh) {
                    circle(frame, Point(j, i), 1, Scalar(147, 20, 255), 2);
                }
            }
        }

        imshow("Video", frame);

        if (key == 'q') { // press 'q' to quit the system
            break;
        }
    }

    destroyAllWindows();

    return 0;
}