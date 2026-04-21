#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/loss.hpp"

namespace {

using adaptive_echo::LossFunction;

constexpr float kPi = 3.14159265358979323846f;

std::vector<float> make_sine(float frequency_hz, float phase_offset = 0.0f) {
    std::vector<float> audio(static_cast<size_t>(adaptive_echo::constants::NUM_SAMPLES));
    const float sample_rate = static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE);
    for (size_t i = 0; i < audio.size(); ++i) {
        const float phase =
            static_cast<float>(i) * frequency_hz * 2.0f * kPi / sample_rate + phase_offset;
        audio[i] = 0.6f * std::sin(phase);
    }
    return audio;
}

std::vector<float> scale_audio(const std::vector<float>& audio, float gain) {
    auto scaled = audio;
    for (float& sample : scaled) {
        sample *= gain;
    }
    return scaled;
}

std::vector<float> offset_audio(const std::vector<float>& audio, size_t offset) {
    std::vector<float> shifted(audio.size(), 0.0f);
    if (offset >= audio.size()) {
        return shifted;
    }
    std::copy(audio.begin(), audio.end() - static_cast<std::ptrdiff_t>(offset),
              shifted.begin() + static_cast<std::ptrdiff_t>(offset));
    return shifted;
}

std::vector<float> shuffle_halves(const std::vector<float>& audio) {
    auto shuffled = audio;
    std::rotate(shuffled.begin(), shuffled.begin() + static_cast<std::ptrdiff_t>(shuffled.size() / 2),
                shuffled.end());
    return shuffled;
}

bool approx_equal(float lhs, float rhs, float tolerance = 1e-6f) {
    return std::abs(lhs - rhs) <= tolerance;
}

bool approx_equal(const std::vector<float>& lhs, const std::vector<float>& rhs,
                  float tolerance = 1e-6f) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!approx_equal(lhs[i], rhs[i], tolerance)) {
            return false;
        }
    }
    return true;
}

bool any_difference(const std::vector<float>& lhs, const std::vector<float>& rhs,
                    float tolerance = 1e-5f) {
    if (lhs.size() != rhs.size()) {
        return true;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - rhs[i]) > tolerance) {
            return true;
        }
    }
    return false;
}

int require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "loss_smoke_test failed: " << message << '\n';
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    const auto target = make_sine(220.0f);
    const auto gain_variant = scale_audio(target, 0.7f);
    const auto offset_variant = offset_audio(target, 97);
    const auto shuffled_variant = shuffle_halves(target);

    const std::vector<std::vector<float>> batch = {gain_variant, offset_variant, shuffled_variant};

    LossFunction<float> seeded_a(target, 1337);
    LossFunction<float> seeded_b(target, 1337);
    const auto losses_a = seeded_a.compute_batch(batch);
    const auto losses_b = seeded_b.compute_batch(batch);
    if (const int rc = require(approx_equal(losses_a, losses_b),
                               "same loss seed and evaluation order should match")) {
        return rc;
    }

    const auto next_losses = seeded_a.compute_batch(batch);
    if (const int rc = require(any_difference(losses_a, next_losses),
                               "successive batch evaluations should use new schedules")) {
        return rc;
    }

    LossFunction<float> shared_batch_loss(target, 2024);
    const auto pair_losses = shared_batch_loss.compute_batch({gain_variant, offset_variant});
    LossFunction<float> single_gain_loss(target, 2024);
    LossFunction<float> single_offset_loss(target, 2024);
    const auto gain_only = single_gain_loss.compute_batch({gain_variant});
    const auto offset_only = single_offset_loss.compute_batch({offset_variant});
    if (const int rc = require(approx_equal(pair_losses[0], gain_only[0]) &&
                                   approx_equal(pair_losses[1], offset_only[0]),
                               "items within one batch should share the sampled schedule")) {
        return rc;
    }

    LossFunction<float> identical_loss(target, 77);
    const float perfect_match = identical_loss(target);
    if (const int rc =
            require(perfect_match < 1e-6f, "identical audio should have near-zero loss")) {
        return rc;
    }

    LossFunction<float> perturbation_loss(target, 91);
    const auto perturbation_scores =
        perturbation_loss.compute_batch({target, gain_variant, offset_variant, shuffled_variant});
    if (const int rc = require(perturbation_scores[1] > perturbation_scores[0] &&
                                   perturbation_scores[2] > perturbation_scores[0] &&
                                   perturbation_scores[3] > perturbation_scores[0],
                               "simple perturbations should increase the loss")) {
        return rc;
    }

    std::cout << "loss_smoke_test passed\n";
    return 0;
}
