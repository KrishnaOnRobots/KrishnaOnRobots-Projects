#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main_1(int argc, char* argv[]) {
    Mat img = imread("C:/Users/senth/OneDrive/Desktop/Real_Time_Filtering/Real_Time_Filtering/Real_Time_Filtering/one.jpg", IMREAD_COLOR);
    namedWindow("image", 0);
    imshow("image", img);
    resizeWindow("image", 1920, 1080);

    while (true) {
        char key = (char)waitKey(20);
        if (key == 'q') {
            cout << "Ending the program...";
            break;
        }
    }

    return 0;
}