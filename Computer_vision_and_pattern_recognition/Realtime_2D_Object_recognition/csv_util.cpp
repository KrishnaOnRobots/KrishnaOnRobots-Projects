/*Code written by Krishna Prasath Senthil Kumaran*/
#include <fstream>
#include <string>
#include <vector>
#include "csv_util.h"

using namespace std;

/*
 * Writting the features of the objects to a CSV file.
 */
void writeToCSV(string filename, vector<string> classNameDataBase, vector<vector<double>> featuresDataBase) {
    // create an output filestream object
    ofstream csvFile;
    csvFile.open(filename, ofstream::trunc);

    // send data to the stream
    for (int i = 0; i < classNameDataBase.size(); i++) {
        // add class name
        csvFile << classNameDataBase[i] << ",";
        // add features
        for (int j = 0; j < featuresDataBase[i].size(); j++) {
            csvFile << featuresDataBase[i][j];
            if (j != featuresDataBase[i].size() - 1) {
                csvFile << ","; // no comma at the end of line
            }
        }
        csvFile << "\n";
    }
    csvFile.close();
}

/*
 * Reading a CSV file which has the features of the objects.
 */
void readFromCSV(string filename, vector<string>& classNameDataBase, vector<vector<double>>& featuresDataBase) {
    // create an input filestream object
    ifstream csvFile(filename);
    if (csvFile.is_open()) {
        // read data line by line
        string line;
        while (getline(csvFile, line)) {
            vector<string> readLine; 
            int pos = 0;
            string token;
            while ((pos = line.find(",")) != string::npos) {
                token = line.substr(0, pos);
                readLine.push_back(token);
                line.erase(0, pos + 1);
            }
            readLine.push_back(line);

            vector<double> firtstFeature; // all the values except the first one from current line
            if (readLine.size() != 0) {
                classNameDataBase.push_back(readLine[0]);
                for (int i = 1; i < readLine.size(); i++) {
                    firtstFeature.push_back(stod(readLine[i]));
                }
                featuresDataBase.push_back(firtstFeature);
            }
            csvFile.close();
        }
    }
}