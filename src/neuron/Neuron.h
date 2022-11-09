//
// Created by Константинова Вера Валентиновна
//

#ifndef NEURON_H
#define NEURON_H

#include <vector>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;
struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron {
public:
    Neuron(unsigned numberOfOutputs, unsigned index);

    void setOutputVal(double value) { neuronOutputValue = value; }

    double getOutputVal(void) const { return neuronOutputValue; }

    void propagateForward(const Layer &previousLayer);

    void calcOutputGradients(double targetValue);

    void calcHiddenGradients(const Layer &nextLayer);

    void updateInputWeights(Layer &previousLayer);

private:
    static double activationFunction(double x);

    static double activationFunctionDerivative(double x);

    // randomWeight: 0 - 1
    static double randomWeight(void) { return rand() / double(RAND_MAX); }

    double gradientWeightSum(const Layer &nextLayer) const;

    double neuronOutputValue;
    vector<Connection> neuronOutputWeights;
    unsigned neuronIndex;
    double neuronGradient;
    static double eta;
};

#endif //NEURON_H