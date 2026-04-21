#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "adaptive_echo/cmaes_optimizer.hpp"

namespace adaptive_echo {

using TrainingResult = CRFMNESResult<float>;
using TrainingProgress = CRFMNESProgress<float>;
using TrainingProgressCallback = std::function<void(const TrainingProgress&)>;

struct CoarseSearchOptions {
    float time_fraction = 0.25f;
    int candidate_multiplier = 4;
    int min_candidates = 20;
    float wide_noise_std = 0.33762205f;
    float medium_noise_std = 0.18796611f;
    float summary_default_mix = 0.3803275f;
    float exploratory_uniform_mix = 0.35553938f;
    float exploratory_summary_mix = 0.42940134f;
    float exploratory_default_mix = 0.21505929f;
};

std::vector<float> default_settings();

std::vector<float> make_time_axis(double duration_seconds, double sample_rate);

std::vector<float> preprocess_target_audio(const std::vector<float>& audio, double input_sample_rate,
                                           int target_length);

float map_normalized_envelope_length(float value);

double max_envelope_duration_seconds(const std::vector<float>& settings);

float normalized_frequency_to_hz(float normalized_value);

float hz_to_normalized_frequency(float hz);

std::vector<float> retune_settings_for_note(const std::vector<float>& settings,
                                            float reference_frequency_hz, int midi_note,
                                            bool pitch_track_osc_a = true,
                                            bool pitch_track_osc_b = true);

std::vector<float> render_note_audio(const std::vector<float>& settings,
                                     float reference_frequency_hz, int midi_note,
                                     double output_sample_rate,
                                     bool pitch_track_osc_a = true,
                                     bool pitch_track_osc_b = true);

TrainingResult train_synth(
    const std::vector<float>& target_audio,
    int population_size = kDefaultCRFMNESPopulationSize,
    float initial_sigma = kDefaultCRFMNESInitialSigma, float time_limit = 60.0f,
    bool verbose = false,
    TrainingProgressCallback progress_callback = {}, uint32_t loss_seed = 0);

TrainingResult train_synth_with_coarse_options(
    const std::vector<float>& target_audio, const CoarseSearchOptions& coarse_options,
    int population_size = kDefaultCRFMNESPopulationSize,
    float initial_sigma = kDefaultCRFMNESInitialSigma, float time_limit = 60.0f,
    bool verbose = false,
    TrainingProgressCallback progress_callback = {}, uint32_t loss_seed = 0);

std::string serialize_settings(const std::vector<float>& settings);

std::vector<float> deserialize_settings(const std::string& text);

}  // namespace adaptive_echo
