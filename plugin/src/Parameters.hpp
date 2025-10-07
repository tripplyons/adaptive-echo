#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <vector>

class EnvelopeParameters {
  public:
    autodiff::var length;
    autodiff::var attack;
    autodiff::var decay;
    autodiff::var sustain;
    autodiff::var release;

    EnvelopeParameters()
        : length(0.0), attack(0.0), decay(0.0), sustain(0.0), release(0.0) {}

    EnvelopeParameters(const std::vector<autodiff::var> &vector) {
        if (vector.size() != 5) {
            throw std::invalid_argument(
                "EnvelopeParameters must have 5 elements");
        }
        length = vector[0];
        attack = vector[1];
        decay = vector[2];
        sustain = vector[3];
        release = vector[4];
    }

    std::vector<autodiff::var> toVector() const {
        return {length, attack, decay, sustain, release};
    }
};

class SingleOscillatorParameters {
  public:
    autodiff::var frequency;
    autodiff::var phaseShift;
    autodiff::var warmth;
    autodiff::var harshness;
    autodiff::var amplitude;
    autodiff::var noiseLevel;

    SingleOscillatorParameters()
        : frequency(0.0), phaseShift(0.0), warmth(0.0), harshness(0.0),
          amplitude(0.0), noiseLevel(0.0) {}

    SingleOscillatorParameters(const std::vector<autodiff::var> &vector) {
        if (vector.size() != 6) {
            throw std::invalid_argument(
                "SingleOscillatorParameters must have 6 elements");
        }
        frequency = vector[0];
        phaseShift = vector[1];
        warmth = vector[2];
        harshness = vector[3];
        amplitude = vector[4];
        noiseLevel = vector[5];
    }

    std::vector<autodiff::var> toVector() const {
        return {frequency, phaseShift, warmth,
                harshness, amplitude,  noiseLevel};
    }
};
using namespace autodiff;

class OscillatorParameters {
  public:
    SingleOscillatorParameters lowModulation;
    SingleOscillatorParameters highModulation;

    OscillatorParameters() : lowModulation(), highModulation() {}

    OscillatorParameters(const std::vector<autodiff::var> &vector)
        : lowModulation(
              std::vector<autodiff::var>(vector.begin(), vector.begin() + 6)),
          highModulation(
              std::vector<autodiff::var>(vector.begin() + 6, vector.end())) {
        if (vector.size() != 12) {
            throw std::invalid_argument(
                "OscillatorParameters must have 12 elements");
        }
    }

    std::vector<autodiff::var> toVector() const {
        std::vector<autodiff::var> result = lowModulation.toVector();
        std::vector<autodiff::var> highVec = highModulation.toVector();
        result.insert(result.end(), highVec.begin(), highVec.end());
        return result;
    }
};

class SynthesizerParameters {
  public:
    OscillatorParameters oscillatorA;
    OscillatorParameters oscillatorB;
    EnvelopeParameters highLowModulation;
    EnvelopeParameters volumeA;
    EnvelopeParameters volumeB;
    EnvelopeParameters fmAmount;
    autodiff::var startFmAmount;
    autodiff::var endFmAmount;

    SynthesizerParameters()
        : oscillatorA(), oscillatorB(), highLowModulation(), volumeA(),
          volumeB(), fmAmount(), startFmAmount(0.0), endFmAmount(0.0) {}

    SynthesizerParameters(const std::vector<autodiff::var> &vector)
        : oscillatorA(
              std::vector<autodiff::var>(vector.begin(), vector.begin() + 12)),
          oscillatorB(std::vector<autodiff::var>(vector.begin() + 12,
                                                 vector.begin() + 24)),
          highLowModulation(std::vector<autodiff::var>(vector.begin() + 24,
                                                       vector.begin() + 29)),
          volumeA(std::vector<autodiff::var>(vector.begin() + 29,
                                             vector.begin() + 34)),
          volumeB(std::vector<autodiff::var>(vector.begin() + 34,
                                             vector.begin() + 39)),
          fmAmount(std::vector<autodiff::var>(vector.begin() + 39,
                                              vector.begin() + 44)),
          startFmAmount(vector[44]), endFmAmount(vector[45]) {
        if (vector.size() != 46) {
            throw std::invalid_argument(
                "SynthesizerParameters must have 46 elements");
        }
    }

    std::vector<autodiff::var> toVector() const {
        std::vector<autodiff::var> result = oscillatorA.toVector();
        std::vector<autodiff::var> oscBVec = oscillatorB.toVector();
        result.insert(result.end(), oscBVec.begin(), oscBVec.end());
        std::vector<autodiff::var> highLowModVec = highLowModulation.toVector();
        std::vector<autodiff::var> volumeAVec = volumeA.toVector();
        std::vector<autodiff::var> volumeBVec = volumeB.toVector();
        std::vector<autodiff::var> fmAmountVec = fmAmount.toVector();
        result.insert(result.end(), highLowModVec.begin(), highLowModVec.end());
        result.insert(result.end(), volumeAVec.begin(), volumeAVec.end());
        result.insert(result.end(), volumeBVec.begin(), volumeBVec.end());
        result.insert(result.end(), fmAmountVec.begin(), fmAmountVec.end());
        result.push_back(startFmAmount);
        result.push_back(endFmAmount);
        return result;
    }

    autodiff::VectorXvar toVectorX() const {
        const std::vector<autodiff::var> &vec = toVector();
        autodiff::VectorXvar result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = vec[i];
        }
        return result;
    }
};
