//
// Created by Константинова Вера Валентиновна
//

#include "Network.h"
#include <cmath>

double Network::recentAverageSmoothingFactor = 30.0; // Number of training samples to average over

void Network::getResults(vector<double> &resultValues) const {
    resultValues.clear();

    for (unsigned n = 0; n < layers.back().size() - 1; ++n) {
        resultValues.push_back(layers.back()[n].getOutputVal());
    }
}

void Network::propagateBackward(const std::vector<double> &targetValues) {
    // Calculate overall network error (output neuron errors)
    Layer &outputLayer = layers.back();
    error = 0.0;

    for (unsigned n = 0; n < outputLayer.size(); n++) {
        double delta = targetValues[n] - outputLayer[n].getOutputVal();
        error += delta;
    }
    error /= outputLayer.size() - 1; // get average error squared
//    error = sqrt(error);

    // Implement a recent average measurement:
    recentAverageError =
            (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
        outputLayer[n].calcOutputGradients(targetValues[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned layerNum = layers.size() - 2; layerNum > 0; layerNum--) {
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); n++) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights
    for (unsigned layerNum = layers.size() - 1; layerNum > 0; layerNum--) {
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Network::propagateForward(const vector<double> &inputValues) {
    // Check the num of inputValues equal to neuron num expect bias
    assert(inputValues.size() == layers[0].size() - 1);

    // Assign {latch} the input values into the input neurons
    for (unsigned i = 0; i < inputValues.size(); i++) {
        layers[0][i].setOutputVal(inputValues[i]);
    }

    // Forward propagate
    for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++) {
        Layer &prevLayer = layers[layerNum - 1];
        for (unsigned n = 0; n < layers[layerNum].size() - 1; n++) {
            layers[layerNum][n].propagateForward(prevLayer);
        }
    }
}

Network::Network(const vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
        layers.push_back(Layer());
        // numOutputs of layer[i] is the numInputs of layer[i+1]
        // numOutputs of last layer is 0
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We have made a new Layer, now fill it ith neurons, and
        // add a bias neuron to the layer:
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
            layers.back().push_back(Neuron(numOutputs, neuronNum));
        }

        // Force the bias node's output value to 1.0. It's the last neuron created above
        layers.back().back().setOutputVal(1.0);
    }
}