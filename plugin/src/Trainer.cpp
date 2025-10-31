#define _USE_MATH_DEFINES
#include <cmath>

#include "Parameters.hpp"
#include "Synth.hpp"

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
    params.oscillatorA.lowModulation.phaseShift = 0.0;
    params.oscillatorA.lowModulation.noiseLevel = 0.0;
    params.oscillatorA.lowModulation.warmth = 0.0;
    params.oscillatorA.lowModulation.harshness = 0.0;
    params.oscillatorA.lowModulation.amplitude = 3.0;

    double frequencyHertz = sqrt(10.0 * 10000.0);
    double numSamples = 10;
    double sampleRate = numSamples * frequencyHertz;

    std::vector<float> time(numSamples);
    for (unsigned int i = 0; i < numSamples; i++) {
        time[i] = i / sampleRate;
    }
    std::vector<float> target(time.size());
    for (unsigned int i = 0; i < time.size(); i++) {
        target[i] = sin(2.0 * M_PI * frequencyHertz * time[i]);
    }

    Synth synth(params);
    for (unsigned int i = 0; i < 10000; i++) {
        bool printLoss = i % 100 == 0;
        synth.simpleTraining(time, target, 0.003, printLoss);
    }

    cout << "Target: ";
    for (float sample : target) {
        cout << sample << endl;
    }
    cout << endl;

    vector<double> output = synth.synthesize(time);
    cout << "Output:" << endl;
    for (double sample : output) {
        cout << sample << endl;
    }

    return 0;
}
