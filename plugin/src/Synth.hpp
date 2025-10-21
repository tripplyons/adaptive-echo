#pragma once

#include "Interpolation.hpp"
#include "Oscillator.hpp"
#include "Parameters.hpp"
#include "TrainingEnvelope.hpp"
#include <iostream>

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

    // Helper for synth_sample
    SingleOscillatorParameters single_osc_linear_interp(
        const SingleOscillatorParameters& a,
        const SingleOscillatorParameters& b,
        const autodiff::var& t)
    {
        SingleOscillatorParameters result;
        result.frequency = linear_interp(a.frequency, b.frequency, t);
        result.phaseShift = linear_interp(a.phaseShift, b.phaseShift, t);
        result.warmth = linear_interp(a.warmth, b.warmth, t);
        result.harshness = linear_interp(a.harshness, b.harshness, t);
        result.amplitude = linear_interp(a.amplitude, b.amplitude, t);
        result.noiseLevel = linear_interp(a.noiseLevel, b.noiseLevel, t);
        return result;
    }

    // Helper for synth_sample
    autodiff::var single_osc_uniform(
        std::mt19937& rng,
        const var& time,
        const SingleOscillatorParameters& params,
        const var& modulation,
        const var& fm_amount)   
    {
        return osc_uniform(rng, time,
                        params.frequency, params.phaseShift,
                        params.warmth, params.harshness,
                        params.amplitude, params.noiseLevel,
                        modulation, fm_amount);
    }
   
    autodiff::var synth_sample(std::mt19937& rng, const autodiff::var& time, const SynthesizerParameters& params){
        using autodiff::var;

        // Calculate envelopes
        var env_vol_a = env_uniform(time, params.volumeA);
        var env_vol_b = env_uniform(time, params.volumeB);
        var env_mod = env_uniform(time, params.highLowModulation);

        // Interpolate settings from modulation
        SingleOscillatorParameters osc_a_settings_modulated = single_osc_linear_interp(
            params.oscillatorA.lowModulation,
            params.oscillatorA.highModulation,
            env_mod
        );
        SingleOscillatorParameters osc_b_settings_modulated = single_osc_linear_interp(
            params.oscillatorB.lowModulation,
            params.oscillatorB.highModulation,
            env_mod
        );

        // Calculate frequency modulation amount
        var env_fm = env_uniform(time, params.fmAmount);
        var fm_amount = linear_interp(params.startFmAmount, params.endFmAmount, env_fm);

        // Calculate oscillators
        var osc_b = single_osc_uniform(rng, time, osc_b_settings_modulated, 0.0, 0.0);
        var osc_a = single_osc_uniform(rng, time, osc_a_settings_modulated, osc_b, fm_amount);

        // Multiply by envelopes
        osc_a *= env_vol_a;
        osc_b *= env_vol_b;

        return osc_a + osc_b;

    }

};