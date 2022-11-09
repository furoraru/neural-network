//
// Created by Константинова Вера Валентиновна
//

#include <cmath>
#include "Neuron.h"

double Neuron::eta = 0.15; // overall net learning rate

void Neuron::updateInputWeights(Layer &previousLayer) {
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (unsigned n = 0; n < previousLayer.size(); n++) {
        Neuron &neuron = previousLayer[n];
        double oldDeltaWeight = neuron.neuronOutputWeights[neuronIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * neuronGradient + oldDeltaWeight;

        neuron.neuronOutputWeights[neuronIndex].deltaWeight = newDeltaWeight;
        neuron.neuronOutputWeights[neuronIndex].weight += newDeltaWeight;
    }
}

double Neuron::gradientWeightSum(const Layer &nextLayer) const {
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
        sum += neuronOutputWeights[n].weight * nextLayer[n].neuronGradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double sum = gradientWeightSum(nextLayer);
    neuronGradient = sum * Neuron::activationFunctionDerivative(neuronOutputValue);
}

void Neuron::calcOutputGradients(double targetValue) {
    double delta = targetValue - neuronOutputValue;
    neuronGradient = delta * Neuron::activationFunctionDerivative(neuronOutputValue);
}

double Neuron::activationFunction(double x) {
    // sigmoid output range [0.0..1.0]
    return 1 / (1 + pow(M_E, -x));
    // tanh - output range [-1.0..1.0]
//    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    // sigmoid derivative
    return x * (1 - x);
// tanh derivative
//    return 1.0 - x * x;
}

// calculate output for every neuron
void Neuron::propagateForward(const Layer &previousLayer) {
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (unsigned n = 0; n < previousLayer.size(); n++) {
        sum += previousLayer[n].getOutputVal() * previousLayer[n].neuronOutputWeights[neuronIndex].weight;
    }

    // then use activation function on result
    neuronOutputValue = Neuron::activationFunction(sum);
}

Neuron::Neuron(unsigned numberOfOutputs, unsigned index) {
    for (unsigned c = 0; c < numberOfOutputs; c++) {
        neuronOutputWeights.push_back(Connection());
        neuronOutputWeights.back().weight = randomWeight();
    }

    neuronIndex = index;
}