/*Created by Krishna Prasath Senthil Kumaran*/
#include <math.h>
#include<vector>
#include <opencv2/opencv.hpp>
#include "img_features.h"



//Helper function to flatten the Matrix
std::vector<float> matConvVector(cv::Mat& mat) {
    cv::Mat flat = mat.reshape(1, mat.total() * mat.channels());
    flat.convertTo(flat, CV_32F);
    return mat.isContinuous() ? flat : flat.clone();
}


std::vector<float> baselineMatching(Mat image)
{
    Mat img = image;
    Size sz = img.size();
    int wid = sz.width;
    int ht = sz.height;

    int count = 0;

    Mat final;
    final.create(9, 9, CV_8UC3);
    int row = 0;
    int col = 0;
    for (int i = ((ht / 2) - 4); i < (ht / 2) + 5; i++)
    {

        for (int j = ((wid / 2) - 4); j < (wid / 2) + 5; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                row = ((ht / 2) - 4);
                row = i - row;
                col = ((wid / 2) - 4);
                col = j - col;
                count++;
                final.at<Vec3b>(row, col)[k] = img.at<Vec3b>(i, j)[k];
            }
        }
    }
    return matConvVector(final);
}

std::vector<float> histogram(cv::Mat& src) {
    int channels[] = { 0, 1, 2 };
    int numChannels = 3;

    // Number of bins for each channel
    int histSize[] = { 256, 256, 256 };

    // Range of values for each channel
    float range[][2] = { {0, 256}, {0, 256}, {0, 256} };

    // Calculate the 2-D histogram
    Mat hist;

    //assigning rows and columns of the image to separate variables
    int rows = src.rows;
    int cols = src.cols;

    // Create a 2D histogram with the specified number of bins
    hist = Mat::zeros(histSize[0], histSize[1], CV_32F);

    // Loop through each pixel in the image
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float binValue_1 = 0;
            float binValue_2 = 0;

            // Compute the bin values for each channel
            for (int i = 0; i < numChannels; i++) {
                int channel = channels[i];
                float value = src.at<Vec3b>(y, x)[channel];
                float binWidth = (range[i][1] - range[i][0]) / histSize[i];
                float binValue = (value - range[i][0]) / binWidth;
                int binIndex = (int)binValue;

                // Check if the bin index is within range
                if (binIndex >= 0 && binIndex < histSize[i]) {
                    if (i == 0) {
                        binValue_1 = binIndex;
                    }
                    else if (i == 1) {
                        binValue_2 = binIndex;
                    }
                }
            }

            // Increment the histogram bin that corresponds to the bin values
            hist.at<float>(binValue_1, binValue_2) += 1;
        }
    }
    //return (0);
    // Normalize the histogram
    normalize(hist, hist, 1, 0, NORM_L1);
    return matConvVector(hist);
}

std::vector<float> multiHistogram(cv::Mat& src) {
    
    std::vector<float> multiHist;
    
    int x = src.cols / 2, y = src.rows / 2;
    int XofROI[] = { 0, x };
    int YofROI[] = { 0, y };
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cv::Mat m = src(cv::Rect(XofROI[i], YofROI[j], x, y)).clone(); // get ROI
            std::vector<float> v = histogram(m); // calculate feature vector
            multiHist.insert(multiHist.end(), v.begin(), v.end()); // concatenate
        }
    }
    return multiHist;
}

// Computes a texture feature vector for an image using the gradient magnitude and orientation
std::vector<float> texture(cv::Mat& src) {
    // Convert Image to Grayscale
    cv::Mat grayscale;
    cv::cvtColor(src, grayscale, cv::COLOR_BGR2GRAY);
    // Gradient Magnitude on Grayscale
    cv::Mat imgMag = magnitude(grayscale);
    // Gradient Orientation on Grayscale
    cv::Mat imgOri = orientation(grayscale);
    int histSize[] = { 8, 8 };
    cv::Mat texture = cv::Mat::zeros(2, histSize, CV_32F);

    float rangeMag = 400 / 8.0;
    float rangeOri = 2 * CV_PI / 8.0;

  
    for (int i = 0; i < imgMag.rows; i++) {
        for (int j = 0; j < imgMag.cols; j++) {
            int m = imgMag.at<float>(i, j) / rangeMag;
            int o = (imgOri.at<float>(i, j) + CV_PI) / rangeOri;
            texture.at<float>(m, o)++;
        }
    }
    // Normalization
    normalize(texture, texture, 1, 0, cv::NORM_L2, -1, cv::Mat());
    return matConvVector(texture);
}

// Computes a combined feature vector for an input color image, 
// by concatenating a texture feature vector and a color feature vector.
std::vector<float> textureAndColor(cv::Mat& src) {
    std::vector<float> feature = texture(src);
    std::vector<float> color = histogram(src);
    feature.insert(feature.end(), color.begin(), color.end());
    return feature;
}

// Different scales and orientation used to output 
std::vector<float> gaborTexture(cv::Mat& src) {
    std::vector<float> gaborTexture;

    // Convert Image to Grayscale
    cv::Mat grayscale;
    cv::cvtColor(src, grayscale, cv::COLOR_BGR2GRAY);

    float sigmaValue[] = { 1.0, 2.0, 4.0 };
    for (auto s : sigmaValue) {
        for (int k = 0; k < 16; k++) {
            float t = k * CV_PI / 8;
            cv::Mat gaborKernel = getGaborKernel(cv::Size(31, 31), s, t, 10.0, 0.5, 0, CV_32F);
            cv::Mat gaborImg;
            std::vector<float> hist(9, 0);
            filter2D(grayscale, gaborImg, CV_32F, gaborKernel);

            // Calculate the Mean and Standard Deviation of each filtered image
            cv::Scalar mean, stddev;
            meanStdDev(gaborImg, mean, stddev);
            gaborTexture.push_back(mean[0]);
            gaborTexture.push_back(stddev[0]);
        }
    }
    // Normalization
    normalize(gaborTexture, gaborTexture, 1, 0, cv::NORM_L2, -1, cv::Mat());
    return gaborTexture;
}

//SobelX Filter
cv::Mat sobelX(cv::Mat& src) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32F);
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_32F);
    // Horizontal Filter
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (j > 0 && j < src.cols - 1) {
                temp.at<float>(i, j) = -src.at<uchar>(i, j - 1) + src.at<uchar>(i, j + 1);
            }
        }
    }
    // Vertical Filter
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (i == 0) {
                dst.at<float>(i, j) = (temp.at<float>(i + 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j)) / 4;
            }
            else if (i == temp.rows - 1) {
                dst.at<float>(i, j) = (temp.at<float>(i - 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i - 1, j)) / 4;
            }
            else {
                dst.at<float>(i, j) = (temp.at<float>(i - 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j)) / 4;
            }
        }
    }
    return dst;
}

//SobelY Filter
cv::Mat sobelY(cv::Mat& src) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32F);
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_32F);

    // Horizontal Filter
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (j == 0) {
                temp.at<float>(i, j) = (src.at<uchar>(i, j + 1) + 2 * src.at<uchar>(i, j) + src.at<uchar>(i, j + 1)) / 4;
            }
            else if (j == src.cols - 1) {
                temp.at<float>(i, j) = (src.at<uchar>(i, j - 1) + 2 * src.at<uchar>(i, j) + src.at<uchar>(i, j - 1)) / 4;
            }
            else {
                temp.at<float>(i, j) = (src.at<uchar>(i, j - 1) + 2 * src.at<uchar>(i, j) + src.at<uchar>(i, j + 1)) / 4;
            }
        }
    }
    // Vertical Filter
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (i > 0 && i < temp.rows - 1) {
                dst.at<float>(i, j) = -temp.at<float>(i - 1, j) + temp.at<float>(i + 1, j);
            }
        }
    }
    return dst;
}

// Magnitude
cv::Mat magnitude(cv::Mat& src) {
    // Calculate SobelX and SobelY
    cv::Mat sx = sobelX(src);
    cv::Mat sy = sobelY(src);

    // Calculate Gradient Magnitude
    cv::Mat dst;
    sqrt(sx.mul(sx) + sy.mul(sy), dst);

    return dst;
}

// Orientation
cv::Mat orientation(cv::Mat& src) {
    // Calculate SobelX and SobelY
    cv::Mat sx = sobelX(src);
    cv::Mat sy = sobelY(src);

    // Calculate Orientation
    cv::Mat dst(src.size(), CV_32F);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<float>(i, j) = atan2(sy.at<float>(i, j), sx.at<float>(i, j));
        }
    }
    return dst;
}

// Select the middle section of a 3x3 Grid
cv::Mat getMiddle(cv::Mat& src) {
    int x = src.cols / 3, y = src.rows / 3;
    cv::Mat middle = src(cv::Rect(x, y, x, y)).clone();
    return middle;
}

