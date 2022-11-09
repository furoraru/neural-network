//
// Created by Константинова Вера Валентиновна
//

#ifndef NETWORK_H
#define NETWORK_H
#include "../neuron/Neuron.h"

class Network {
public:
    Network(const vector<unsigned> &topology);

    // function for forward propagation of data
    void propagateForward(const vector<double> &inputValues);

    // function for backward propagation of errors made by neurons
    void propagateBackward(const vector<double> &targetValues);

    void getResults(vector<double> &resultValues) const;

    double getRecentAverageError(void) const { return recentAverageError; }
    double getError(void) const { return error; }

private:
    vector<Layer> layers; //layers[layerNum][neuronNum]
    double error;
    double recentAverageError;
    static double recentAverageSmoothingFactor;
};


#endif
