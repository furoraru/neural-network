//
// Created by Константинова Вера Валентиновна
//

#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <fstream>
#include "../neuron/Neuron.h"

class TrainingData {
public:
    TrainingData(const string filename);

    bool isEof(void) {
        return trainingDataFile.eof();
    }

    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputValues);

    unsigned getTargetOutputs(vector<double> &targetOutputValues);

private:
    ifstream trainingDataFile;
};


#endif
