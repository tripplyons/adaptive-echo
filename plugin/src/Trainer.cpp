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
    params.oscillatorA.lowModulation.phaseShift = -5.0;
    params.oscillatorA.lowModulation.noiseLevel = -2.0;
    params.oscillatorA.lowModulation.warmth = 0.0;
    params.oscillatorA.lowModulation.harshness = 0.0;
    params.oscillatorA.lowModulation.amplitude = 1.0;

    double frequencyHertz = sqrt(10.0 * 10000.0);
    double numSamples = 6;
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

    for (unsigned int i = 0; i < 1000; i++) {
        bool printLoss = i % 100 == 0;
        synth.simpleTraining(time, target, 0.01, printLoss);
    }

    return 0;
}
