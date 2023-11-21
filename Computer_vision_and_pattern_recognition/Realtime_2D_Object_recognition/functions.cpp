/*Code written by Krishna Prasath Senthil Kumaran*/
#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "functions.h"

using namespace std;
using namespace cv;

/*
* This function takes in an image and returns a binary image.
*/
Mat threshold(Mat& image) {
    int thresHold = 130;
    Mat binaryImage, grayscale;
    binaryImage = Mat(image.size(), CV_8UC1);

    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    for (int i = 0; i < grayscale.rows; i++) {
        for (int j = 0; j < grayscale.cols; j++) {
            if (grayscale.at<uchar>(i, j) <= thresHold) {
                binaryImage.at<uchar>(i, j) = 255;
            }
            else {
                binaryImage.at<uchar>(i, j) = 0;
            }
        }
    }
    return binaryImage;
}

/*
* This function take in a binary image and returns a cleaned binary image through morphological closing.
*/
Mat cleanImage(Mat& image) {
    
    Mat binaryImage;
    
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(25, 25));
    morphologyEx(image, binaryImage, MORPH_CLOSE, kernel);
    
    return binaryImage;
}

/*
* This function takes in a binary image, the labeled regions, the stats 
  and centroids of the regions and the top N labels and returns a segmented image.
*/
Mat extractSegmentedRegions(Mat& inputImage, Mat& labeledRegions, Mat& stats, Mat& centroids, vector<int>& topNLabels) {

    // Initialize the output segmented image.
    Mat segmentedOutputImage;

    // Determine the number of connected components in the input image and their statistics.
    int numberOfLabels = connectedComponentsWithStats(inputImage, labeledRegions, stats, centroids);

    // Store the areas of each region in a matrix for sorting.
    Mat regionAreas = Mat::zeros(1, numberOfLabels - 1, CV_32S);
    Mat sortedAreas;

    for (int i = 1; i < numberOfLabels; i++) {
        int regionArea = stats.at<int>(i, CC_STAT_AREA);
        regionAreas.at<int>(i - 1) = regionArea;
    }

    // Sort the regions by area size in descending order.
    if (regionAreas.cols > 0) {
        sortIdx(regionAreas, sortedAreas, SORT_EVERY_ROW + SORT_DESCENDING);
    }

    // Initialize the colors vector with black for background and random colors for each region.
    vector<Vec3b> colors(numberOfLabels, Vec3b(0, 0, 0));
    for (int i = 1; i < numberOfLabels; i++) {
        //colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
        colors[i] = Vec3b(180, 150, 60);
    }

    // Select the N largest regions for segmentation, and ignore any region below a certain area threshold.
    int nLargestRegions = 3;
    int areaThreshold = 1000;
    nLargestRegions = (nLargestRegions < sortedAreas.cols) ? nLargestRegions : sortedAreas.cols;
    for (int i = 0; i < nLargestRegions; i++) {
        int regionIndex = sortedAreas.at<int>(i) + 1;
        if (stats.at<int>(regionIndex, CC_STAT_AREA) > areaThreshold) {
            topNLabels.push_back(regionIndex);
        }
    }

    // Apply the colors to each pixel in the labeled image.
    segmentedOutputImage = Mat::zeros(labeledRegions.size(), CV_8UC3);
    for (int i = 0; i < segmentedOutputImage.rows; i++) {
        for (int j = 0; j < segmentedOutputImage.cols; j++) {
            int label = labeledRegions.at<int>(i, j);
            segmentedOutputImage.at<Vec3b>(i, j) = colors[label];
        }
    }

    return segmentedOutputImage;
}
/*
* The function takes a binary image region and a set of parameters (centroid x and y, and rotation angle alpha) as input.
* It then rotates the region by the specified angle around the centroid, 
  and computes the minimum bounding box that encloses the rotated region.
* The function returns the rotated bounding box.
*/
RotatedRect makeBoundingBox(Mat& region, double x, double y, double alpha) {
    int maximum_x = INT_MIN; 
    int maximum_y = INT_MIN;
    int minimum_x = INT_MAX; 
    int minimum_y = INT_MAX;
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            if (region.at<uchar>(i, j) == 255) {
                int new_coord_x = (i - x) * cos(alpha) + (j - y) * sin(alpha);
                int new_coord_y = -(i - x) * sin(alpha) + (j - y) * cos(alpha);
                maximum_x = max(maximum_x, new_coord_x);
                minimum_x = min(minimum_x, new_coord_x);
                maximum_y = max(maximum_y, new_coord_y);
                minimum_y = min(minimum_y, new_coord_y);
            }
        }
    }
    int side_x = maximum_x - minimum_x;
    int side_y = maximum_y - minimum_y;

    Point centroid = Point(x, y);
    Size size = Size(side_x, side_y);

    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI);
}
/*
 * This function draws a line of 75 pixels given a starting point and an angle
 */
void pointer(Mat& image, double x, double y, double alpha, Scalar color) {
    double length = 75.0;
    double cosAlpha = cos(alpha);
    double sinAlpha = sin(alpha);
    double xPrime = x + length * cosAlpha;
    double yPrime = y + length * sinAlpha;
    arrowedLine(image, Point(x, y), Point(xPrime, yPrime), color, 3);
}

/*
 * This function draws a bounding box around the specified region.
 */
void drawBoundingBox(Mat& image, RotatedRect makeBoundingBox, Scalar color) {
    Point2f rect_points[4];
    makeBoundingBox.points(rect_points);
    for (int i = 0; i < 4; i++) {
        line(image, rect_points[i], rect_points[(i + 1) % 4], color, 3);
    }
}

/*
 * This function calculates the HU Moments according to the given central moments
 */
void HuMomentsCalculation(Moments moment, vector<double>& huMoments) {
    double huMoment[7]; // HuMoments require the parameter type to be double[]
    HuMoments(moment, huMoment);

    // covert array to vector
    for (double d : huMoment) {
        huMoments.push_back(d);
    }
    return;
}
double euclideanDistance(vector<double> point1, vector<double> features2) {
    double sum1 = 0.0;
    double sum2 = 0.0; 
    double sumDifference = 0.0;
    for (int i = 0; i < point1.size(); i++) {
        sumDifference += (point1[i] - features2[i]) * (point1[i] - features2[i]);
        sum1 += point1[i] * point1[i];
        sum2 += features2[i] * features2[i];
    }
    return sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2));
}

double manhattanDistance(vector<double> point1, vector<double> features2) {
    double x_diff = 0.0, y_diff = 0.0, distance = 0.0;
    for (int i = 0; i < point1.size(); i++) {
        x_diff = point1[i] - features2[i];
        y_diff = point1[i] - features2[i];
        distance += abs(x_diff) + abs(y_diff);
    }
    return distance;
}



/*
 * The function takes a set of known feature vectors, class names, 
   and a current feature vector as input, along with a distance metric to use.
 * It then finds the closest feature vector in the set of known feature vectors, using the specified distance metric.
 */
string nearestNeighbour(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, string distMetric) {
    double thresHold = 0.15;
    double distance = DBL_MAX;
    string className = " ";
    double curDistance = 0.0;
    for (int i = 0; i < featureVectors.size(); i++) { // loop the known features to get the closed one
        vector<double> dbFeature = featureVectors[i];
        string dbClassName = classNames[i];
        if (distMetric == "e") {
            curDistance = euclideanDistance(dbFeature, currentFeature);
        }
        else if (distMetric == "m") {
            curDistance = manhattanDistance(dbFeature, currentFeature);
        }
        else {
            cout << "Invalid distance metric" << "\n";
        }

        if (curDistance < distance && curDistance < thresHold) {
            className = dbClassName;
            distance = curDistance;
        }
    }
    return className;
}

/*
 * Given some data and a feature vector, this function gets the name of the given feature vector
 * Infers based on K-Nearest-Neighbor, and use normalized euclidean distance as distance metric
 */
string KNN_classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K, string distMetric) {
    double thresHold = 0.15;
    // compute the distances of current feature vector with all the feature vectors in DB
    vector<double> distances;
    double distance = 0.0;
    for (int i = 0; i < featureVectors.size(); i++) {
        vector<double> dbFeature = featureVectors[i];
        if (distMetric == "e"){
			distance = euclideanDistance(dbFeature, currentFeature);
        }
		else if (distMetric == "m"){
			distance = manhattanDistance(dbFeature, currentFeature);
        }
        else { cout<<"Invalid distance metric"; 
        }

        if (distance < thresHold) {
            distances.push_back(distance);
        }
    }

    string className = " ";
    if (distances.size() > 0) {
        // sort the distances in ascending order
        vector<int> sortedIdx;
        sortIdx(distances, sortedIdx, SORT_EVERY_ROW + SORT_ASCENDING);

        // get the first K class name, and count the number of each name
        vector<string> nNames;
        int sort_size = sortedIdx.size();
        map<string, int> noOfNames;
        int range = min(sort_size, K); 
        for (int i = 0; i < range; i++) {
            string name = classNames[sortedIdx[i]];
            if (noOfNames.find(name) != noOfNames.end()) {
                noOfNames[name]++;
            }
            else {
                noOfNames[name] = 1;
            }
        }

        // get the class name that appear the most times in the K nearest neighbors
        int count = 0;
        for (map<string, int>::iterator it = noOfNames.begin(); it != noOfNames.end(); it++) {
            if (it->second > count) {
                className = it->first;
                count = it->second;
            }
        }
    }
    return className;
}

/*
 * This function returns the corresponding class name given a code
 */
string className_input(char c) {
    std::map<char, string> myMap{
            {'p', "pen"}, {'h', "headphone"}, {'g', "eyeglass"},
            {'r', "wrench"}, {'c', "calculator"}, {'b', "box"}, {'k', "key"},
            {'n', "notebook"}, {'x', "belt"},{'s', "credit card"}, {'t', "tape"} , {'y', "bottle"}
    };
    return myMap[c];
}
