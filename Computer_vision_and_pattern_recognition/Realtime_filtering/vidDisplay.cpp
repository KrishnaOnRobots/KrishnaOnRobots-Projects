#include <opencv2/opencv.hpp>
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include"opencv2/core/core.hpp"
#include"filters.h"
#include<iostream>
#include<string>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    VideoCapture* capdev;

    Mat grayscale;
    Mat Blur;
    Mat xsobel;
    Mat xsobel_conv;
    Mat ysobel;
    Mat ysobel_conv;
    Mat sobel_mag;
    Mat sobel_conv;
    Mat blurquant;
    Mat cartoonize;

    // open the video device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    namedWindow("Video", 1); // identifies a window
    Mat frame;

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        imshow("Video",frame);
        // see if there is a waiting keystroke
        char key = waitKey(10);
        if (key == 'q') {
            break;
        }
        else if (key == 's') {
            imwrite("C:/Users/senth/OneDrive/Desktop/Real_Time_Filtering/Real_Time_Filtering/Real_Time_Filtering/new.jpg", frame);
            namedWindow("image", 0);
            imshow("image", frame);
            resizeWindow("image", 600, 400);
        }
        else if (key == 'g') {
            cvtColor(frame, grayscale, COLOR_RGB2GRAY);
            imshow("Image1", grayscale);
        }
        else if (key == 'h') {
            greyscale(frame, grayscale);
            imshow("Image2", grayscale);
        }
        else if (key == 'b') {
            blur5x5(frame, Blur);
            imshow("Image3", Blur);
        }
        else if (key == 'x') {
            sobelX3x3(frame, xsobel);
            convertScaleAbs(xsobel, xsobel_conv, 1, 0);
            imshow("Image4", xsobel_conv);
        }
        else if (key == 'y') {
            sobelY3x3(frame, ysobel);
            convertScaleAbs(ysobel, ysobel_conv, 1, 0);
            imshow("Image5", ysobel_conv);
        }
        else if (key == 'm') {
            sobelX3x3(frame, xsobel);
            sobelY3x3(frame, ysobel);
            magnitude(xsobel, ysobel, sobel_mag);
            convertScaleAbs(sobel_mag, sobel_conv, 1, 0);
            imshow("Image6", sobel_conv);
        }
        else if (key == 'i') {
            blurQuantize(frame, blurquant, 15);
            imshow("Image7", blurquant);
        }
        else if (key == 'c') {
            cartoon(frame, cartoonize, 5, 20);
            imshow("Image8", cartoonize);
        }
        else if (key == 'z') {
            Mat brightImg;
            brightness(frame, brightImg);
            imshow("Image9", brightImg);
        }
        else if (key == 'v') {
            Mat Invertcolor;
            invertIntensity(frame, Invertcolor);
            imshow("Ext.Image10", Invertcolor);
        }
        else if (key == 'o') {
            Mat edgedetect;
            EdgeDetection(frame, edgedetect);
            imshow("Ext.Image11", edgedetect);
        }
        else if (key == 'f') {
            Mat colorReduction;
            ColorReductionFilter(frame, colorReduction, 60);
            imshow("Ext.Image12", colorReduction);
        }
    }
    delete capdev;
    return(0);
}