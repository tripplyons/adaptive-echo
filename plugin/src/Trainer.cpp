#include "Parameters.hpp"
#include <Eigen/Dense>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <iostream>
#include <vector>

using namespace autodiff;

autodiff::var forward(autodiff::VectorXvar &params) {
    std::vector<autodiff::var> paramVec(params.data(),
                                        params.data() + params.size());
    SynthesizerParameters synthesizerParams(paramVec);
    return 1.0 * synthesizerParams.oscillatorA.lowModulation.frequency +
           2.0 * synthesizerParams.oscillatorB.lowModulation.frequency +
           3.0 * synthesizerParams.oscillatorA.lowModulation.phaseShift +
           4.0 * synthesizerParams.oscillatorB.lowModulation.phaseShift;
}

SynthesizerParameters backward(SynthesizerParameters &params) {
    autodiff::VectorXvar paramVector = params.toVectorX();
    autodiff::var y = forward(paramVector);
    autodiff::VectorXvar gradients = autodiff::gradient(y, paramVector);
    return SynthesizerParameters(std::vector<autodiff::var>(
        gradients.data(), gradients.data() + gradients.size()));
}

int main() {
    SynthesizerParameters params;
    // Set some non-zero values for the parameters we're using
    params.oscillatorA.lowModulation.frequency = 1.0;
    params.oscillatorB.lowModulation.frequency = 2.0;
    params.oscillatorA.lowModulation.phaseShift = 0.5;
    params.oscillatorB.lowModulation.phaseShift = 1.5;

    SynthesizerParameters gradients = backward(params);

    std::cout << "Gradients: " << gradients.oscillatorA.lowModulation.frequency
              << ", " << gradients.oscillatorB.lowModulation.frequency << ", "
              << gradients.oscillatorA.lowModulation.phaseShift << ", "
              << gradients.oscillatorB.lowModulation.phaseShift << std::endl;

    return 0;
}
