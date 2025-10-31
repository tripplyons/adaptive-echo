#define _USE_MATH_DEFINES
#include <cmath>

#include "Normalization.hpp"
#include "Parameters.hpp"
#include "Synth.hpp"
#include "WavHandler.hpp"

#include <Eigen/Dense>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <iostream>
#include <vector>

using namespace autodiff;

int main() {
    SynthesizerParameters params;
    // Set some non-zero values for the parameters we're using
    params.oscillatorA.lowModulation.frequency = 0.0;
    params.oscillatorA.lowModulation.phaseShift = 2.0;
    params.oscillatorA.lowModulation.noiseLevel = 0.0;
    params.oscillatorA.lowModulation.warmth = 0.0;
    params.oscillatorA.lowModulation.harshness = 0.0;
    params.oscillatorA.lowModulation.amplitude = 3.0;

    double frequencyHertz = sqrt(10.0 * 10000.0);
    unsigned int numSamples = 48000;
    unsigned int batchSize = 16;
    double sampleRate = 48000;

    vector<double> time(numSamples);
    for (unsigned int i = 0; i < numSamples; i++) {
        time[i] = i / sampleRate;
    }
    vector<double> target(time.size());
    for (unsigned int i = 0; i < time.size(); i++) {
        target[i] = sin(2.0 * M_PI * frequencyHertz * time[i]);
    }

    writeData("target.wav", normalize(target), sampleRate);

    Synth synth(params);

    vector<double> initialOutput = synth.synthesize(time);
    writeData("initial_output.wav", normalize(initialOutput), sampleRate);

    for (unsigned int i = 0; i < 100; i++) {
        bool printLoss = i % 100 == 0;
        vector<unsigned int> batchIndices = vector<unsigned int>(numSamples);
        for (unsigned int i = 0; i < numSamples; i++) {
            batchIndices[i] = rand() % numSamples;
        }
        vector<double> timeBatch(batchSize);
        vector<double> targetBatch(batchSize);
        for (unsigned int j = 0; j < batchSize; j++) {
            timeBatch[j] = time[batchIndices[j]];
            targetBatch[j] = target[batchIndices[j]];
        }
        synth.simpleTraining(timeBatch, targetBatch, 0.0003, printLoss, 1);
        cout << "Iteration " << i
             << ", frequency: " << params.oscillatorA.lowModulation.frequency
             << ", phase shift: " << params.oscillatorA.lowModulation.phaseShift
             << endl;
    }

    vector<double> output = synth.synthesize(time);

    writeData("output.wav", normalize(output), sampleRate);

    return 0;
}
