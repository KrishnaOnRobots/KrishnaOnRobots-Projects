#pragma once
/*Code written by Krishna Prasath Senthil Kumaran*/
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>

using namespace std;

void writeToCSV(string filename, vector<string> classNameDataBase, vector<vector<double>> featuresDataBase);

void readFromCSV(string filename, vector<string>& classNameDataBase, vector<vector<double>>& featuresDataBase);