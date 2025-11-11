#define _USE_MATH_DEFINES
#include <cmath>

#include "Normalization.hpp"
#include "Synth.hpp"
#include "WavHandler.hpp"

#include <iostream>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main() {
    std::cout << "Starting trainer" << std::endl;

    std::string model_path = "../adaptive_echo_python/graphs/synth.pt";

    Synth synth(model_path);
    synth.randomizeParameters();

    Synth synth2(synth);
    synth2.randomizeParameters();

    unsigned int sample_rate = 48000;
    unsigned int num_samples = sample_rate * 5;
    vector<float> times(num_samples);
    for (unsigned int i = 0; i < num_samples; i++) {
        times[i] = i / (float)sample_rate;
    }

    vector<float> targets = synth2.generate(times);
    vector<double> targetsDouble(targets.begin(), targets.end());
    vector<int32_t> targetsInt = normalize(targetsDouble);
    writeData("target.wav", targetsInt, sample_rate);

    vector<float> initialOutput = synth.generate(times);
    vector<double> initialOutputDouble(initialOutput.begin(),
                                       initialOutput.end());
    vector<int32_t> initialOutputInt = normalize(initialOutputDouble);
    writeData("initial_output.wav", initialOutputInt, sample_rate);

    std::cout << "Starting training" << std::endl;

    float loss = synth.simpleTrain(times, targets, 0.01f, 100, true);
    std::cout << "Loss: " << loss << std::endl;

    vector<float> finalOutput = synth.generate(times);
    vector<double> finalOutputDouble(finalOutput.begin(), finalOutput.end());
    vector<int32_t> finalOutputInt = normalize(finalOutputDouble);
    writeData("final_output.wav", finalOutputInt, sample_rate);
    return 0;
}
