/*Created by Krishna Prasath Senthil Kumaran*/

#include "dis_metrics.h"
#include <opencv2/opencv.hpp>
#include<vector>
#include<iostream>

// Distance Metric: sum of square difference
float variation(std::vector<float>& target, std::vector<float>& src) {
    CV_Assert(target.size() == src.size());
    float sum = 0;
    for (int i = 0; i < target.size(); i++) {
        sum += (target[i] - src[i]) * (target[i] - src[i]);
    }
    return sum;
}

// Distance Metric: histogram intersection
float histogramIntersection(std::vector<float>& target, std::vector<float>& src) {
    CV_Assert(target.size() == src.size());
    float intersection = 0;
    for (int i = 0; i < target.size(); i++) {
        intersection += (std::min(target[i], src[i]));
    }
    return intersection;
}