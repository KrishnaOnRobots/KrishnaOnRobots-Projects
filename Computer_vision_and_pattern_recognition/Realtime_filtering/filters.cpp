#include"filters.h"
#include<opencv2/opencv.hpp>
#include<math.h>
#include<iostream>

using namespace cv;
using namespace std;

int greyscale(Mat &src, Mat &dst) {
    dst.create(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                dst.at<uchar>(i, j) = src.at<Vec3b>(i, j)[c] / 3;
            }
        }
    }
    return 0;
}
int blur5x5(Mat& src, Mat& dst) {

    dst = src.clone();
    Mat temp = src.clone();

    // Implementing it as two linear filters
    int hor_filter[5] = { 1, 2, 4, 2, 1 };
    int ver_filter[5] = { 1, 2, 4, 2, 1 };


    //Normalization factor to bring the values between 0 to 255
    int norm_factor = 10;

    // Horizontal convolution
    for (int i = 0; i < src.rows; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            int b = 0, g = 0, r = 0, sum = 0;
            for (int k = -2; k <= 2; k++) {
                b = b + src.at<Vec3b>(i, j + k)[0] * hor_filter[k + 2];
                g = g + src.at<Vec3b>(i, j + k)[1] * hor_filter[k + 2];
                r = r + src.at<Vec3b>(i, j + k)[2] * hor_filter[k + 2];
            }
            temp.at<Vec3b>(i, j)[0] = b / norm_factor;
            temp.at<Vec3b>(i, j)[1] = g / norm_factor;
            temp.at<Vec3b>(i, j)[2] = r / norm_factor;
        }
    }

    // Vertical convolution
    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 0; j < src.cols; j++) {
            int b = 0, g = 0, r = 0;
            for (int k = -2; k <= 2; k++) {
                b = b + temp.at<Vec3b>(i + k, j)[0] * ver_filter[k + 2];
                g = g + temp.at<Vec3b>(i + k, j)[1] * ver_filter[k + 2];
                r = r + temp.at<Vec3b>(i + k, j)[2] * ver_filter[k + 2];
            }
            dst.at<Vec3b>(i, j)[0] = b / norm_factor;
            dst.at<Vec3b>(i, j)[1] = g / norm_factor;
            dst.at<Vec3b>(i, j)[2] = r / norm_factor;
        }
    }
    return 0;
}
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.rows,src.cols, CV_16SC3); 
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                // apply the Sobel kernel for the X direction
                int mat_x = src.at<cv::Vec3b>(i - 1, j - 1)[k] * -1 + src.at<cv::Vec3b>(i - 1, j + 1)[k] +
                          src.at<cv::Vec3b>(i, j - 1)[k] * -2 + src.at<cv::Vec3b>(i, j + 1)[k] * 2 +
                          src.at<cv::Vec3b>(i + 1, j - 1)[k] * -1 + src.at<cv::Vec3b>(i + 1, j + 1)[k];
                    
                dst.at<cv::Vec3s>(i, j)[k] = mat_x;
            }
        }
    }
    return 0;
}
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    dst = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                int mat_y = src.at<cv::Vec3b>(i - 1, j - 1)[k] * 1 + src.at<cv::Vec3b>(i - 1, j)[k] * 2 +
                    src.at<cv::Vec3b>(i - 1, j + 1)[k] * 1 + src.at<cv::Vec3b>(i + 1, j - 1)[k] * -1 +
                    src.at<cv::Vec3b>(i + 1, j)[k] * -2 + src.at<cv::Vec3b>(i + 1, j + 1)[k] * -1;

                dst.at<cv::Vec3s>(i, j)[k] = mat_y;
            }
        }
    }
    return 0;
}
int magnitude(Mat& sx, Mat& sy, Mat& dst){
    
    Mat mag(sx.size(), CV_32FC3);

    for (int y = 0; y < sx.rows; y++){
        for (int x = 0; x < sx.cols; x++){
            for (int c = 0; c < 3; c++){
                float dx = sx.at<Vec3s>(y, x)[c];
                float dy = sy.at<Vec3s>(y, x)[c];
                mag.at<Vec3f>(y, x)[c] = sqrt(dx * dx + dy * dy);
            }
        }
    }
    normalize(mag, mag, 0, 255, cv::NORM_MINMAX, CV_8UC3);
    mag.convertTo(dst, CV_8UC3);
    
    return 0;
}
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
    dst.create(src.size(), CV_8UC3);

    blur5x5(src, dst);
    //To figure out the bucket size
    int b = 255 / levels;

    //Extracting the color channels
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                int x = dst.at<Vec3b>(i, j)[k];
                int xt = x / b;
                int xf = xt * b;
                xf = dst.at<Vec3b>(i, j)[k];
            }
        }
    }
    return 0;
}
int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {
    Mat sobelx, sobely, mag, mag_conv, blurQuant;
    sobelX3x3(src, sobelx);
    sobelY3x3(src, sobely);
    
    magnitude(sobelx, sobely, mag);
    convertScaleAbs(mag, mag_conv, 1,0);

    blurQuantize(src, blurQuant, levels);
    
    dst.create(blurQuant.rows, blurQuant.cols - 1, CV_8UC3);

        for (int i = 0; i <= blurQuant.rows - 1; i++) {
            Vec3b *rowptr = dst.ptr<Vec3b>(i);
            Vec3b *magptr = mag.ptr<Vec3b>(i);
            Vec3b *blurquantptr = blurQuant.ptr<Vec3b>(i);

            for (int j = 0; j <= blurQuant.cols - 1; j++) {
                for (int k = 0; k < 3; k++) {
                    if (magptr[j][k] >= magThreshold) {
                        rowptr[j][k] = 0;
                    }
                    else
                        rowptr[j][k] = blurquantptr[j][k];
                }
            }
    }
        return 0;
}

// Increase and reduce brightness of the image
int brightness(Mat& src, Mat& dst) {
    int level;
    cout << "Enter a value between (50 - 200 or -50 to -200) to increase/decrease the brightness of the image: " << endl;
    cin >> level;

    Mat adjustbrightnessImg;
    if (level>=50 && level<=200 || level>=-200 && level<=-50){
        src.convertTo(adjustbrightnessImg, -1, 1, level);
        dst = adjustbrightnessImg;
    }
    else {
        cout << "Enter an another value that is within the range."<<endl;
    }
    return 0;
}
//Function to invert the intensity of the image
int invertIntensity(Mat& src, Mat& dst) {
    
    dst.create(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec3b intensity = src.at<Vec3b>(i, j);
            for (int k = 0; k < 3; k++) {
                intensity[k] = 255 - intensity[k];
                dst.at<Vec3b>(i, j) = intensity;
            }
        }
    }
    return 0;
}
//Function or edge detection
int EdgeDetection(Mat& src, Mat& dst) {
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gradient_x, gradient_y;
    cv::Sobel(gray, gradient_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradient_y, CV_32F, 0, 1, 3);
    cv::Mat gradient;
    cv::magnitude(gradient_x, gradient_y, gradient);
    cv::normalize(gradient, gradient, 0, 255, cv::NORM_MINMAX);
    gradient.convertTo(gradient, CV_8U);
    gradient.copyTo(dst);

    return 0;
}

//Function for color reduction
int ColorReductionFilter(Mat& src, Mat& dst, int divisor) {
    
    src.copyTo(dst);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b& p = dst.at<cv::Vec3b>(y, x);
            p[0] = cv::saturate_cast<uchar>(p[0] / divisor * divisor + divisor / 2);
            p[1] = cv::saturate_cast<uchar>(p[1] / divisor * divisor + divisor / 2);
            p[2] = cv::saturate_cast<uchar>(p[2] / divisor * divisor + divisor / 2);
        }
    }
    return 0;
}

