///*Created by Krishna Prasath Senthil Kumaran*/
//#include <iostream>
//#include <string.h>
//#include <dirent.h>
//#include <opencv2/opencv.hpp>
//#include "img_features.h"
//#include "dis_metrics.h"
//#include "csv_util.h"
//
//int main(int argc, const char* argv[]) {
//    char imgDBLocation[256];
//    char featureParam[256];
//    char DB_Path[256];
//    DIR* dir;
//    struct dirent* d;
//
//    //Appending each argument to the variables
//    strcpy(imgDBLocation, argv[1]);
//    strcpy(featureParam, argv[2]);
//    strcpy(DB_Path, argv[3]);
//    
//    printf("Image directory: %s\n", imgDBLocation);
//    printf("Image feature parameter: %s\n", featureParam);
//    printf("DB Path: %s\n", DB_Path);
//
//    //Error prompt if we enter more than 3 arguments
//    if (argc < 4) {
//        printf("Check the arguments entered... \n");
//        exit(-1);
//    }
//    //Promt is displayed if there exists a CSV for the appropriate features
//    FILE* f = fopen(DB_Path, "r");
//    if (f) {
//        printf("The CSV has been already generated for the feature... \n");
//        return 0;
//    }
//
//    //std::string imgDBLocation = "C:\\Users\\senth\\Desktop\\Project_2\\olympus";
//   /* printf(imgDBLocation);*/
//    dir = opendir(imgDBLocation);
//    if (dir == NULL) {
//        printf("Error in opening image directory... \n");
//        exit(-1);
//    }
//    // loop over all the files in the image file listing
//    while ((d = readdir(dir)) != NULL) {
//        // check if the file is an image
//        if (strstr(d->d_name, ".jpg") || strstr(d->d_name, ".png") || strstr(d->d_name, ".ppm") || strstr(d->d_name, ".tif")) {
//            // build the overall filename
//            char featureName[256];
//            strcpy(featureName, imgDBLocation);
//            strcat(featureName, "/");
//            strcat(featureName, d->d_name);
//
//            cv::Mat image;
//            image = cv::imread(featureName);
//            std::vector<float> img_feature;
//            // Baseline Matching
//            if (!strcmp(featureParam, "baselineMatching")) {
//                img_feature = baselineMatching(image);
//                //std::cout<<"\nFeature size: "<<img_feature.size()<<'\n';
//            }
//            // Color Histogram
//            else if (!strcmp(featureParam, "histogram")) {
//                img_feature = histogram(image);
//            }
//            //Multi-Histogram
//            else if (!strcmp(featureParam, "multiHistogram")) {
//                img_feature = multiHistogram(image);
//            }
//            // Texture
//            else if (!strcmp(featureParam, "texture")) {
//                img_feature = texture(image);
//            }
//            // Texture and color
//            else if (!strcmp(featureParam, "textureColor")) {
//                img_feature = textureAndColor(image);
//            }
//            // Gabor Texture
//            else if (!strcmp(featureParam, "gaborTexture")) {
//                img_feature = gaborTexture(image);
//            }
//            else {
//                printf("Unsupported feature parameter %s.\n", featureParam);
//                continue;
//            }
//            //Creating and appending the CSV with feature data
//            // To generate the CSV file run the program and enter the path of the Olympus-
//            // and the feature that you want to extract from the image and with the path give a name for your CSV file.
//            //append_image_data_csv(DB_Path, featureName, img_feature);
//            if (append_image_data_csv(DB_Path, featureName, img_feature)) {
//                printf("Error in writing image feature to CSV file.\n");
//                return -1;
//            }
//        }
//        
//    }
//    closedir(dir);
//    return 0;
//}
