#include "adaptive_echo/engine.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/interpolation.hpp"
#include "adaptive_echo/loss.hpp"
#include "adaptive_echo/resample.hpp"
#include "adaptive_echo/synth.hpp"

namespace adaptive_echo {
namespace {

constexpr float kMinFrequencyHz = 50.0f;
constexpr float kMaxFrequencyHz = 2000.0f;
constexpr float kDefaultReferenceFrequencyHz = 440.0f;

float sanitize_frequency(float frequency_hz) {
    return std::clamp(frequency_hz, kMinFrequencyHz, kMaxFrequencyHz);
}

void normalize_audio(std::vector<float>& audio, float target_peak) {
    float max_value = 0.0f;
    for (float sample : audio) {
        max_value = std::max(max_value, std::abs(sample));
    }

    if (max_value <= 0.0f) {
        return;
    }

    const float scale = target_peak / max_value;
    for (float& sample : audio) {
        sample *= scale;
    }
}

}  // namespace

std::vector<float> default_settings() {
    return std::vector<float>(constants::NUM_SETTINGS, 0.5f);
}

std::vector<float> make_time_axis(double duration_seconds, double sample_rate) {
    const auto sample_count =
        std::max(1, static_cast<int>(std::ceil(duration_seconds * sample_rate)));
    std::vector<float> time(static_cast<size_t>(sample_count));

    if (sample_count == 1) {
        time[0] = 0.0f;
        return time;
    }

    const auto step = static_cast<float>(duration_seconds / static_cast<double>(sample_count - 1));
    for (int i = 0; i < sample_count; ++i) {
        time[static_cast<size_t>(i)] = static_cast<float>(i) * step;
    }
    return time;
}

std::vector<float> preprocess_target_audio(const std::vector<float>& audio, double input_sample_rate,
                                           int target_length) {
    std::vector<float> processed = audio;
    if (processed.empty() || input_sample_rate <= 0.0) {
        return std::vector<float>(static_cast<size_t>(target_length), 0.0f);
    }

    if (std::abs(input_sample_rate - constants::TRAINING_SAMPLE_RATE) > 0.5) {
        const auto ratio = static_cast<float>(constants::TRAINING_SAMPLE_RATE / input_sample_rate);
        const auto new_length = std::max(1, static_cast<int>(processed.size() * ratio));
        processed = resample_fft(processed, static_cast<size_t>(new_length));
    }

    if (static_cast<int>(processed.size()) > target_length) {
        processed.resize(static_cast<size_t>(target_length));
    } else {
        processed.resize(static_cast<size_t>(target_length), 0.0f);
    }

    normalize_audio(processed, 1.0f);
    normalize_audio(processed, 0.5f);
    return processed;
}

float map_normalized_envelope_length(float value) {
    return exp_interp(0.2f, static_cast<float>(constants::NUM_SECONDS), value);
}

double max_envelope_duration_seconds(const std::vector<float>& settings) {
    if (settings.size() < constants::NUM_SETTINGS) {
        return static_cast<double>(constants::NUM_SECONDS);
    }

    const std::array<int, 4> envelope_length_indices = {0, 5, 10, 39};
    double max_duration = 0.2;
    for (int index : envelope_length_indices) {
        max_duration = std::max(max_duration, static_cast<double>(map_normalized_envelope_length(
                                                   settings[static_cast<size_t>(index)])));
    }
    return max_duration;
}

float normalized_frequency_to_hz(float normalized_value) {
    const float min_frequency_log = 12.0f * std::log2(kMinFrequencyHz);
    const float max_frequency_log = 12.0f * std::log2(kMaxFrequencyHz);
    const float semitones =
        min_frequency_log + (max_frequency_log - min_frequency_log) * normalized_value;
    return std::pow(2.0f, semitones / 12.0f);
}

float hz_to_normalized_frequency(float hz) {
    const float clamped_hz = sanitize_frequency(hz);
    const float min_frequency_log = 12.0f * std::log2(kMinFrequencyHz);
    const float max_frequency_log = 12.0f * std::log2(kMaxFrequencyHz);
    const float semitones = 12.0f * std::log2(clamped_hz);
    return std::clamp((semitones - min_frequency_log) / (max_frequency_log - min_frequency_log),
                      0.0f, 1.0f);
}

std::vector<float> retune_settings_for_note(const std::vector<float>& settings,
                                            float reference_frequency_hz, int midi_note) {
    std::vector<float> retuned = settings;
    if (retuned.size() < constants::NUM_SETTINGS) {
        retuned.resize(constants::NUM_SETTINGS, 0.5f);
    }

    const float safe_reference =
        std::max(1.0f, reference_frequency_hz > 0.0f ? reference_frequency_hz
                                                     : kDefaultReferenceFrequencyHz);
    const float note_hz =
        440.0f * std::pow(2.0f, (static_cast<float>(midi_note) - 69.0f) / 12.0f);
    const float semitone_offset = 12.0f * std::log2(note_hz / safe_reference);
    const float ratio = std::pow(2.0f, semitone_offset / 12.0f);

    const std::array<int, 4> frequency_indices = {
        constants::OSC_A_FREQ_LOW_INDEX, constants::OSC_A_FREQ_HIGH_INDEX,
        constants::OSC_B_FREQ_LOW_INDEX, constants::OSC_B_FREQ_HIGH_INDEX};

    for (int index : frequency_indices) {
        const auto current_hz = normalized_frequency_to_hz(retuned[static_cast<size_t>(index)]);
        retuned[static_cast<size_t>(index)] = hz_to_normalized_frequency(current_hz * ratio);
    }

    return retuned;
}

std::vector<float> render_note_audio(const std::vector<float>& settings,
                                     float reference_frequency_hz, int midi_note,
                                     double output_sample_rate) {
    const auto duration_seconds = max_envelope_duration_seconds(settings);
    const auto time = make_time_axis(duration_seconds, output_sample_rate);
    auto note_settings = retune_settings_for_note(settings, reference_frequency_hz, midi_note);
    return synth(note_settings, time, static_cast<float>(output_sample_rate));
}

TrainingResult train_synth(const std::vector<float>& target_audio, int population_size,
                           float initial_sigma, float time_limit, bool verbose,
                           TrainingProgressCallback progress_callback) {
    LossFunction<float> loss_fn(target_audio);
    const auto time = make_time_axis(constants::NUM_SECONDS, constants::TRAINING_SAMPLE_RATE);
    auto synth_fn = [](const std::vector<float>& settings, const std::vector<float>& time_axis) {
        return synth(settings, time_axis, static_cast<float>(constants::TRAINING_SAMPLE_RATE));
    };

    return run_crfmnes_optimization<float>(loss_fn, time, synth_fn, population_size,
                                           initial_sigma, time_limit, 10000, verbose,
                                           std::move(progress_callback));
}

std::string serialize_settings(const std::vector<float>& settings) {
    std::ostringstream stream;
    for (size_t i = 0; i < settings.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << settings[i];
    }
    return stream.str();
}

std::vector<float> deserialize_settings(const std::string& text) {
    auto settings = default_settings();
    std::istringstream stream(text);
    std::string token;
    size_t index = 0;
    while (std::getline(stream, token, ',') && index < settings.size()) {
        if (!token.empty()) {
            settings[index] = std::stof(token);
        }
        ++index;
    }
    return settings;
}

}  // namespace adaptive_echo
