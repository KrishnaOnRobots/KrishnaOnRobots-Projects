/*Created by Krishna Prasath Senthil Kumaran*/
#pragma once
#include <stdio.h>
#include<vector>
#include<iostream>
#include <opencv2/opencv.hpp>

float variation(std::vector<float>& target, std::vector<float>& src);
float histogramIntersection(std::vector<float>& target, std::vector<float>& src);