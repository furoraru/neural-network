//
// Created by Константинова Вера Валентиновна on 08.11.2022.
//

#include "TrainingData.h"
#include <sstream>

void TrainingData::getTopology(vector<unsigned> &topology) {
    string line;
    string label;

    getline(trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename) {
    trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputValues) {
    inputValues.clear();

    string line;
    getline(trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputValues.push_back(oneValue);
        }
    }

    return inputValues.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputValues) {
    targetOutputValues.clear();

    string line;
    getline(trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputValues.push_back(oneValue);
        }
    }

    return targetOutputValues.size();
}