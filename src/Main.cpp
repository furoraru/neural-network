#include <iostream>
#include <cassert>
#include "network/Network.h"
#include "training/TrainingData.h"

void showVectorVals(string label, vector<double> &v) {
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}

int main() {
    TrainingData trainData("trainingData.txt");
    vector<unsigned> topology;

    trainData.getTopology(topology);
    Network myNet(topology);

    vector<double> inputValues, targetValues, resultValues;
    int trainingPass = 0;
    while (!trainData.isEof()) {
        trainingPass++;
        cout << endl << "Pass" << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputValues) != topology[0])
            break;
        showVectorVals(": Inputs :", inputValues);
        myNet.propagateForward(inputValues);

        // Collect the network's actual results:
        myNet.getResults(resultValues);
        showVectorVals("Outputs:", resultValues);

        // Train the network what the outputs should have been:
        trainData.getTargetOutputs(targetValues);
        showVectorVals("Targets:", targetValues);
//        assert(targetValues.size() == topology.back());

        myNet.propagateBackward(targetValues);

        // Report how well the training is working, average over recent
        cout << "Network recent average error: " << myNet.getError() << endl;
    }

    cout << endl << "Done" << endl;

}