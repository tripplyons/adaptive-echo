#pragma once

#include "Interpolation.hpp"
#include "Oscillator.hpp"
#include "Parameters.hpp"
#include <iostream>

// synthesize() takes in autodiff::var time, SynthesizerParameters params
// output is a value (autodiff::var) 
// analagous to synth() function in synth.py

class Synth {
  public:
    Synth(SynthesizerParameters params) : params(params) {}
    ~Synth() {}
    SynthesizerParameters simpleGradient(std::vector<float> time,
                                         std::vector<float> target,
                                         bool printLoss) {
        std::mt19937 rng(0);
        autodiff::VectorXvar paramsVector = params.toVectorX();
        SynthesizerParameters currentParams(paramsVector);
        std::vector<autodiff::var> output(time.size());
        for (unsigned int i = 0; i < time.size(); i++) {
            output[i] = osc_uniform(
                rng, time[i],
                sigmoid(currentParams.oscillatorA.lowModulation.frequency),
                sigmoid(currentParams.oscillatorA.lowModulation.phaseShift),
                sigmoid(currentParams.oscillatorA.lowModulation.warmth),
                sigmoid(currentParams.oscillatorA.lowModulation.harshness),
                sigmoid(currentParams.oscillatorA.lowModulation.amplitude),
                sigmoid(currentParams.oscillatorA.lowModulation.noiseLevel),
                0.0, 0.0);
        }
        autodiff::var loss = 0;
        for (unsigned int i = 0; i < time.size(); i++) {
            loss += (output[i] - target[i]) * (output[i] - target[i]);
        }
        if (printLoss) {
            std::cout << "Loss: " << loss << std::endl;
        }
        autodiff::VectorXvar gradients = autodiff::gradient(loss, paramsVector);
        return SynthesizerParameters(gradients);
    }
    void simpleTraining(std::vector<float> time, std::vector<float> target,
                        float learningRate, bool printLoss) {
        SynthesizerParameters gradients =
            simpleGradient(time, target, printLoss);
        std::vector<autodiff::var> gradientsVar = gradients.toVector();
        std::vector<autodiff::var> paramsVar = params.toVector();

        std::vector<double> newParams(gradientsVar.size());
        for (unsigned int i = 0; i < gradientsVar.size(); i++) {
            newParams[i] =
                double(paramsVar[i]) - learningRate * double(gradientsVar[i]);
        }
        std::vector<autodiff::var> newParamsVar(newParams.size());
        for (unsigned int i = 0; i < newParams.size(); i++) {
            newParamsVar[i] = autodiff::var(newParams[i]);
        }
        params = SynthesizerParameters(newParamsVar);
    }
    SynthesizerParameters params;

    /*
    SynthesizerParameters:
        OscillatorParameters oscillatorA;
        OscillatorParameters oscillatorB;
        EnvelopeParameters highLowModulation;
        EnvelopeParameters volumeA;
        EnvelopeParameters volumeB;
        EnvelopeParameters fmAmount;
        autodiff::var startFmAmount;
        autodiff::var endFmAmount;

    OscillatorParameters:
        SingleOscillatorParameters lowModulation;
        SingleOscillatorParameters highModulation;

    SingleOscillatorParameters:
        autodiff::var frequency;
        autodiff::var phaseShift;
        autodiff::var warmth;
        autodiff::var harshness;
        autodiff::var amplitude;
        autodiff::var noiseLevel;

    EnvelopeParameters:
        autodiff::var length;
        autodiff::var attack;
        autodiff::var decay;
        autodiff::var sustain;
        autodiff::var release;
    */

    /*
    From python synth() args:
        rng,  # random number generator
        time,  # time
        env_vol_a_settings,  # envelope for volume of osc_a
        env_vol_b_settings,  # envelope for volume of osc_b
        env_mod_settings,  # envelope for modulation amount
        osc_a_settings,  # settings for when osc_a is at no modulation
        osc_b_settings,  # settings for when osc_b is at no modulation
        osc_a_mod_settings,  # settings for when osc_a is at full modulation
        osc_b_mod_settings,  # settings for when osc_b is at full modulation
        env_fm_setting,  # envelope for frequency modulation amount
        fm_range,  # range of frequency modulation amount
    */
   /*
    autodiff::var synth_sample(const autodiff::var& time, const SynthesizerParameters& params){
        using autodiff::var;

        // Calculate envelopes
        
    }
        */
};