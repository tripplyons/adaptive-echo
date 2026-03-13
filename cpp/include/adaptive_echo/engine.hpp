#pragma once

#include <array>
#include <string>
#include <vector>

#include "adaptive_echo/cmaes_optimizer.hpp"

namespace adaptive_echo {

std::vector<float> default_settings();

std::vector<float> make_time_axis(double duration_seconds, double sample_rate);

std::vector<float> preprocess_target_audio(const std::vector<float>& audio, double input_sample_rate,
                                           int target_length);

float map_normalized_envelope_length(float value);

double max_envelope_duration_seconds(const std::vector<float>& settings);

float normalized_frequency_to_hz(float normalized_value);

float hz_to_normalized_frequency(float hz);

std::vector<float> retune_settings_for_note(const std::vector<float>& settings,
                                            float reference_frequency_hz, int midi_note);

std::vector<float> render_note_audio(const std::vector<float>& settings,
                                     float reference_frequency_hz, int midi_note,
                                     double output_sample_rate);

CMAESResult<float> train_synth(const std::vector<float>& target_audio, int population_size = 32,
                               float initial_sigma = 3.0f, float time_limit = 60.0f,
                               bool verbose = false);

std::string serialize_settings(const std::vector<float>& settings);

std::vector<float> deserialize_settings(const std::string& text);

}  // namespace adaptive_echo
