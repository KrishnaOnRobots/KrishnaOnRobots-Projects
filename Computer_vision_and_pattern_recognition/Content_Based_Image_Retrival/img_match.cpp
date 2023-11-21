/*Created by Krishna Prasath Senthil Kumaran*/
#include <dirent.h>
#include <string.h>
#include <utility>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "csv_util.h"
#include "img_features.h"
#include "dis_metrics.h"


int main(int argc, char *argv[]) {
    cv::Mat target;
    char targetImage[256];
    char featureParam[30];
    char imgFeatureCSV[256];
    char disMetric[256];
    char iterations[3];
    std::vector<float> targetFeature;

    strcpy(targetImage, argv[1]);
    strcpy(featureParam, argv[2]);
    strcpy(imgFeatureCSV, argv[3]);
    strcpy(disMetric, argv[4]);
    strcpy(iterations, argv[5]);

    if (argc < 6) {
        printf("Wrong number of input commands... \n");
        exit(-1);
    }

    target = cv::imread(targetImage);
    if (target.empty()) {
        printf("Image not found... \n");
        exit(-1);
    }


    if (!strcmp(featureParam, "baselineMatching")) { // baselineMatching
        targetFeature = baselineMatching(target);
    }
    // Color Histogram
    else if (!strcmp(featureParam, "histogram")) {
        targetFeature = histogram(target);
    }
    else if (!strcmp(featureParam, "multiHistogram")) {
        targetFeature = multiHistogram(target);
    }
    // Texture
    else if (!strcmp(featureParam, "texture")) {
        targetFeature = texture(target);
    }
    else if (!strcmp(featureParam, "textureColor")) { // texture and color
        targetFeature = textureAndColor(target);
    }
    else if (!strcmp(featureParam, "gaborTexture")) { // Gabor texture
        targetFeature = gaborTexture(target);
    }
    else {
        printf("Feature not found... Check the entered feature type... \n");
        exit(-1);
    }

    std::vector<char*> imageNames;
    std::vector<std::vector<float>> imageFeature;
    FILE* fp = fopen(imgFeatureCSV, "r");
    if (fp) {
        read_image_data_csv(imgFeatureCSV, imageNames, imageFeature);
    }
    

    // Compute the distances between the target and the images in olympus
    std::vector<std::pair<std::string, float>> distances;
    float d;
    std::pair<std::string, float> imgDistance;
    for (int i = 0; i < imageNames.size(); i++) {
        if (!strcmp(disMetric, "var")) {
            // sum of square difference
            d = variation(targetFeature, imageFeature[i]);
            imgDistance = std::make_pair(imageNames[i], d);
            distances.push_back(imgDistance);

            // Sort Distance Vectors in Ascending order
            sort(distances.begin(), distances.end(), [](auto &left, auto &right) {
                return left.second < right.second;
                });
        }
        else if (!strcmp(disMetric, "inter")) {
            // histogram intersection
            d = histogramIntersection(targetFeature, imageFeature[i]);
            imgDistance = std::make_pair(imageNames[i], d);
            distances.push_back(imgDistance);
            // sort the vector of distances in descending order
            sort(distances.begin(), distances.end(), [](auto &left, auto &right) {
                return left.second > right.second;
                });
        }
        else {
            printf("Check if the Distance metric has been specified... \n");
            exit(-1);
        }
    }
    std::ofstream matchedImagesFile("C:/Users/senth/OneDrive/Desktop/Project_2_Image_classification/olympus/matched_images.txt");

    // get the first N matches, exclude the target itself
    //The first N matched images along with their name and path are added to an text file called matched_images
    int N = 0, i = 0;
    while (N < std::stoi(iterations)) {
        cv::Mat image = cv::imread(distances[i].first);
        if (image.size() != target.size() || (sum(image != target) != cv::Scalar(0,0,0,0))) {
            std::string matchedImagePath = std::string(distances[i].first);
            std::cout << matchedImagePath << std::endl;
            // Write the matched image file path to the output file
            matchedImagesFile << matchedImagePath << std::endl;
            i++; N++;
        }
        else {
            i++;
        }
    }
    return 0;
}