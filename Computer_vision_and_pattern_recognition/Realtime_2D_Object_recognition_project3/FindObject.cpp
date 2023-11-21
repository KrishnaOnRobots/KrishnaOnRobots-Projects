/*Code written by Krishna Prasath Senthil Kumaran*/
#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include "functions.h"
#include "csv_util.h"
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Wrong input." << endl;
        exit(-1);
    }

    // featuresDataBase and classNameDataBase are used to save the feature vectors of known objects
    vector<string> classNameDataBase;
    vector<vector<double>> featuresDataBase;
  

    // load existing data from csv file to featuresDataBase and classNameDB
    readFromCSV(argv[1], classNameDataBase, featuresDataBase);

    cv::VideoCapture* capdev;
    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;
    bool training = false;

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
       cv::imshow("Video", frame);

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 't') {
            training = !training;
            if (training) {
                cout << "Training Mode" << endl;
            }
            else {
                cout << "Display Mode" << endl;
            }
        }

        //compute the threshold of the current frame
        Mat thresh = threshold(frame);
        imshow("Threshold", thresh);

        //Cleaning the threshold image
        Mat cleanedFrame = cleanImage(thresh);
        imshow("CleanedFrame", cleanedFrame);

        // Extract the largest 3 regions from the frame
        Mat labeledRegions, stats, centroids;
        vector<int> topNLabels;
        Mat regionFrame = extractSegmentedRegions(cleanedFrame, labeledRegions, stats, centroids, topNLabels);
        imshow("RegionFrame", regionFrame);

        for (int n = 0; n < topNLabels.size(); n++) {
            int label = topNLabels[n];
            Mat region;
            region = (labeledRegions == label);

            // calculate central moments, centroids, and alpha
            Moments m = moments(region, true);
            double centroidX = centroids.at<double>(label, 0);
            double centroidY = centroids.at<double>(label, 1);
            double alpha = 1.0 / 2.0 * atan2(2 * m.mu11, m.mu20 - m.mu02);

            // get the least central axis and bounding box of this region
            RotatedRect BoundingBox = makeBoundingBox(region, centroidX, centroidY, alpha);
            pointer(frame, centroidX, centroidY, alpha, Scalar(0, 0, 255));
            drawBoundingBox(frame, BoundingBox, Scalar(0, 255, 0));

            //Calculate Hu Moments
            vector<double> huMoments;
            HuMomentsCalculation(m, huMoments);

            if (training) {
                // in training mode
                // display current region in binary form
                namedWindow("Current Region", WINDOW_AUTOSIZE);
                imshow("Current Region", region);

                // ask the user for a class name
                cout << "Give a name for this class: " << endl;
                // get the code for each class name
                char k = waitKey(0);
                string className = className_input(k); //see the function for a detailed mapping

                // update the DB
                featuresDataBase.push_back(huMoments);
                classNameDataBase.push_back(className);

                // after labeling all the objects,
                // switch back to inference mode and destroy all the windows created in training mode
                if (n == topNLabels.size() - 1) {
                    training = false;
                    cout << "Display Mode" << endl;
                    destroyWindow("Current Region");
                }
            }
            else {
                // in inference mode
                // classify the object
                string className;
                if (!strcmp(argv[2], "n")) { // nearest neighbor
                    className = nearestNeighbour(featuresDataBase, classNameDataBase, huMoments, argv[3]);
                }
                else if (!strcmp(argv[2], "k")) { // KNN
                    className = KNN_classifier(featuresDataBase, classNameDataBase, huMoments, 8, argv[3]);
                }
                // overlay classname to the video
                putText(frame, className, Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
            }
        }

        imshow("Labelled Output", frame); // display the video

        // if user types 'q', quit.
        if (key == 'q') {
            // when quit, add data in classNameDataBase and featuresDataBase to csv file
            writeToCSV(argv[1], classNameDataBase, featuresDataBase);
            break;
       
        }
    }

    delete capdev;
    return(0);
}
