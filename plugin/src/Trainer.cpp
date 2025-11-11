#define _USE_MATH_DEFINES
#include <cmath>

#include "Normalization.hpp"
#include "Synth.hpp"
#include "WavHandler.hpp"
#include "TwoEncoders.hpp"

#include <iostream>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main() {
    std::cout << "Starting trainer" << std::endl;

    TwoEncoders twoEncoders("two_encoders.pt");
    twoEncoders.train(100, 0.0003f, 1000, true);

    return 0;
}
