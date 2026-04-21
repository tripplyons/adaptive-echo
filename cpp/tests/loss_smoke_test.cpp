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

template <typename T>
bool same_feature_signature(const adaptive_echo::RandomLossWindow<T>& lhs,
                            const adaptive_echo::RandomLossWindow<T>& rhs) {
    if (lhs.features.size() != rhs.features.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.features.size(); ++i) {
        const auto& lhs_feature = lhs.features[i];
        const auto& rhs_feature = rhs.features[i];
        if (lhs_feature.family != rhs_feature.family ||
            lhs_feature.coefficient != rhs_feature.coefficient ||
            lhs_feature.band_count != rhs_feature.band_count) {
            return false;
        }
    }
    return true;
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
    {
        const std::vector<float> frequency_weights = {1.0f, 2.0f, 3.0f, 4.0f,
                                                      5.0f, 6.0f, 7.0f, 8.0f};
        const auto grouped_edges = adaptive_echo::detail::compute_linear_group_edges<float>(8, 4);
        const auto grouped_weights =
            adaptive_echo::detail::aggregate_band_weights(frequency_weights, grouped_edges);
        const std::vector<float> target_spectrum = {1.0f, 2.0f, 3.0f, 4.0f,
                                                    5.0f, 6.0f, 7.0f, 8.0f};
        const auto grouped_target =
            adaptive_echo::detail::compute_grouped_magnitude_sums(target_spectrum, grouped_edges);
        const auto grouped_identical =
            adaptive_echo::detail::compute_grouped_magnitude_sums(target_spectrum, grouped_edges);
        const float grouped_identical_loss = adaptive_echo::detail::compute_weighted_log_magnitude_loss(
            grouped_identical, grouped_target, grouped_weights);
        if (const int rc = require(grouped_identical_loss < 1e-6f,
                                   "grouped spectral sums should have zero loss for exact matches")) {
            return rc;
        }

        const std::vector<float> shifted_energy_spectrum = {1.0f, 2.0f, 4.0f, 5.0f,
                                                            5.0f, 6.0f, 8.0f, 9.0f};
        const auto grouped_shifted = adaptive_echo::detail::compute_grouped_magnitude_sums(
            shifted_energy_spectrum, grouped_edges);
        const float grouped_shifted_loss = adaptive_echo::detail::compute_weighted_log_magnitude_loss(
            grouped_shifted, grouped_target, grouped_weights);
        if (const int rc = require(grouped_shifted_loss > 0.0f,
                                   "grouped spectral sums should respond to grouped energy changes")) {
            return rc;
        }
    }

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

    const auto schedule = adaptive_echo::detail::make_random_loss_schedule(target, 1337);
    bool found_distinct_window_features = false;
    for (size_t i = 1; i < schedule.windows.size(); ++i) {
        if (!same_feature_signature(schedule.windows.front(), schedule.windows[i])) {
            found_distinct_window_features = true;
            break;
        }
    }
    if (const int rc = require(found_distinct_window_features,
                               "different windows should sample different cepstral feature sets")) {
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

    {
        adaptive_echo::RandomLossWindow<float> fixed_window;
        fixed_window.window =
            adaptive_echo::RandomWindowSpec<float> {1024, 0, adaptive_echo::RandomWindowType::Hann, 0.0f};
        const auto fixed_context =
            adaptive_echo::detail::prepare_window_context(target, fixed_window);
        const auto identical_spectrum =
            adaptive_echo::detail::compute_windowed_magnitude_spectrum(target, fixed_context.window);
        const auto offset_spectrum = adaptive_echo::detail::compute_windowed_magnitude_spectrum(
            offset_variant, fixed_context.window);
        const float grouped_16_identical = adaptive_echo::detail::compute_weighted_log_magnitude_loss(
            adaptive_echo::detail::compute_grouped_magnitude_sums(
                identical_spectrum, fixed_context.spectral_group_edges_16),
            fixed_context.target_grouped_spectrum_16, fixed_context.spectral_group_weights_16);
        const float grouped_16_offset = adaptive_echo::detail::compute_weighted_log_magnitude_loss(
            adaptive_echo::detail::compute_grouped_magnitude_sums(
                offset_spectrum, fixed_context.spectral_group_edges_16),
            fixed_context.target_grouped_spectrum_16, fixed_context.spectral_group_weights_16);
        const float grouped_4_identical = adaptive_echo::detail::compute_weighted_log_magnitude_loss(
            adaptive_echo::detail::compute_grouped_magnitude_sums(
                identical_spectrum, fixed_context.spectral_group_edges_4),
            fixed_context.target_grouped_spectrum_4, fixed_context.spectral_group_weights_4);
        const float grouped_4_offset = adaptive_echo::detail::compute_weighted_log_magnitude_loss(
            adaptive_echo::detail::compute_grouped_magnitude_sums(
                offset_spectrum, fixed_context.spectral_group_edges_4),
            fixed_context.target_grouped_spectrum_4, fixed_context.spectral_group_weights_4);
        if (const int rc = require(grouped_16_identical < 1e-6f && grouped_16_offset > 0.0f,
                                   "16-bin grouped spectral term should detect perturbations")) {
            return rc;
        }
        if (const int rc = require(grouped_4_identical < 1e-6f && grouped_4_offset > 0.0f,
                                   "4-bin grouped spectral term should detect perturbations")) {
            return rc;
        }
    }

    std::cout << "loss_smoke_test passed\n";
    return 0;
}
