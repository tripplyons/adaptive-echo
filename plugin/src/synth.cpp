#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <string>
#include <fstream>

// Simple WAV file writer
class WavWriter {
private:
    std::ofstream file;
    
    void write_int(uint32_t value) {
        file.write(reinterpret_cast<const char*>(&value), 4);
    }
    
    void write_short(uint16_t value) {
        file.write(reinterpret_cast<const char*>(&value), 2);
    }

public:
    bool open(const std::string& filename, int sample_rate, int num_samples) {
        file.open(filename, std::ios::binary);
        if (!file.is_open()) return false;
        
        // RIFF header
        file.write("RIFF", 4);
        write_int(36 + num_samples * 2);  // File size - 8
        file.write("WAVE", 4);
        
        // Format chunk
        file.write("fmt ", 4);
        write_int(16);  // PCM format chunk size
        write_short(1);  // PCM format
        write_short(1);  // Mono
        write_int(sample_rate);
        write_int(sample_rate * 2);  // Byte rate
        write_short(2);  // Block align
        write_short(16);  // Bits per sample
        
        // Data chunk
        file.write("data", 4);
        write_int(num_samples * 2);
        
        return true;
    }
    
    void write_sample(float sample) {
        int16_t int_sample = static_cast<int16_t>(std::clamp(sample * 32767.0f, -32768.0f, 32767.0f));
        file.write(reinterpret_cast<const char*>(&int_sample), 2);
    }
    
    void close() {
        file.close();
    }
};

// Normalize audio data
std::vector<float> normalize_wav(const std::vector<float>& data) {
    if (data.empty()) return data;
    
    float max_val = 0.0f;
    for (float sample : data) {
        max_val = std::max(max_val, std::abs(sample));
    }
    
    if (max_val < 1e-6f) return data;
    
    std::vector<float> normalized(data.size());
    float scale = 0.95f / max_val;  // Leave some headroom
    for (size_t i = 0; i < data.size(); ++i) {
        normalized[i] = data[i] * scale;
    }
    
    return normalized;
}

// Save WAV file
void save_wav(const std::string& path, const std::vector<float>& data, int sr) {
    WavWriter writer;
    if (writer.open(path, sr, static_cast<int>(data.size()))) {
        for (float sample : data) {
            writer.write_sample(sample);
        }
        writer.close();
        std::cout << "Saved WAV to " << path << std::endl;
    } else {
        std::cerr << "Failed to save WAV to " << path << std::endl;
    }
}

// Utility functions
float linear_interp(float a, float b, float t) {
    return a + (b - a) * t;
}

float exp_interp(float a, float b, float t) {
    return a * std::pow(b / a, t);
}

// Random number generator wrapper
class RNG {
private:
    std::mt19937 generator;
    std::normal_distribution<float> normal_dist;
    
public:
    RNG(uint32_t seed = 0) : generator(seed), normal_dist(0.0f, 1.0f) {}
    
    void seed(uint32_t seed) {
        generator.seed(seed);
    }
    
    float normal() {
        return normal_dist(generator);
    }
    
    std::vector<float> normal_vector(size_t size) {
        std::vector<float> result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = normal_dist(generator);
        }
        return result;
    }
    
    // Split RNG (approximation of JAX's split)
    RNG split() {
        // Simple approach: use different seeds
        static uint32_t counter = 1;
        return RNG(generator() + counter++);
    }
};

// Envelope generator
float env(float time, float length, float attack, float decay, float sustain, float release) {
    float value;
    if (time < attack) {
        value = time / attack;
    } else if (time < attack + decay) {
        value = 1.0f - (1.0f - sustain) * (time - attack) / decay;
    } else if (time < length - release) {
        value = sustain;
    } else {
        value = sustain * (length - time) / release;
    }
    return std::clamp(value, 0.0f, 1.0f);
}

// Waveform + noise generator
float osc(RNG& rng, float time, float freq, float phase_shift, float warmth, 
          float harshness, float amplitude, float noise_level, 
          float modulation = 0.0f, float fm_amount = 0.0f) {
    
    float noise = rng.normal() * 0.2f;
    
    float phase = time * freq + phase_shift;
    if (modulation != 0.0f) {
        phase += modulation * fm_amount;
    }
    phase = std::fmod(phase, 1.0f);
    
    phase = 0.5f * (std::pow(phase, warmth) - std::pow(1.0f - phase, warmth) + 1.0f);
    
    phase *= 2.0f * M_PI;
    
    float sin_val = std::sin(phase);
    
    float wave = std::copysign(std::pow(std::abs(sin_val), harshness), sin_val) * amplitude;
    
    float noise_interp = 0.2f * noise_level;
    
    return linear_interp(wave, noise, noise_interp);
}

// Use envelope generator with inputs from 0 to 1
float env_uniform(float time, float length, float attack, float decay, float sustain, float release) {
    float min_length = 0.1f;
    float max_length = 1.0f;
    length = exp_interp(min_length, max_length, length);
    
    float min_attack = 0.05f;
    float max_attack = 0.5f;
    attack = exp_interp(min_attack, max_attack, attack);
    
    float min_decay = 0.05f;
    float max_decay = 0.5f;
    decay = exp_interp(min_decay, max_decay, decay);
    
    float min_sustain = 0.1f;
    float max_sustain = 1.0f;
    sustain = linear_interp(min_sustain, max_sustain, sustain);
    
    float min_release = 0.05f;
    float max_release = 0.5f;
    release = exp_interp(min_release, max_release, release);
    
    return env(time, length, attack, decay, sustain, release);
}

// Use oscillator generator with inputs from 0 to 1
float osc_uniform(RNG& rng, float time, float freq, float phase_shift, float warmth,
                  float harshness, float amplitude, float noise_level,
                  float modulation = 0.0f, float fm_amount = 0.0f) {
    
    float min_freq = std::log2(10.0f) * 12.0f;
    float max_freq = std::log2(20000.0f) * 12.0f;
    float semitones = linear_interp(min_freq, max_freq, freq);
    freq = std::pow(2.0f, semitones / 12.0f);
    
    float min_phase_shift = 0.0f;
    float max_phase_shift = 1.0f;
    phase_shift = linear_interp(min_phase_shift, max_phase_shift, phase_shift);
    
    float min_warmth = 1.0f / 5.0f;
    float max_warmth = 5.0f;
    warmth = exp_interp(min_warmth, max_warmth, warmth);
    
    float min_harshness = 1.0f / 5.0f;
    float max_harshness = 5.0f;
    harshness = exp_interp(min_harshness, max_harshness, harshness);
    
    float min_amplitude = 0.1f;
    float max_amplitude = 1.0f;
    amplitude = linear_interp(min_amplitude, max_amplitude, amplitude);
    
    return osc(rng, time, freq, phase_shift, warmth, harshness, amplitude, noise_level, modulation, fm_amount);
}

// Synthesize a single sample at a time
float synth(RNG& rng, float time,
            const std::vector<float>& env_vol_a_settings,
            const std::vector<float>& env_vol_b_settings,
            const std::vector<float>& env_mod_settings,
            const std::vector<float>& osc_a_settings,
            const std::vector<float>& osc_b_settings,
            const std::vector<float>& osc_a_mod_settings,
            const std::vector<float>& osc_b_mod_settings,
            const std::vector<float>& env_fm_setting,
            const std::vector<float>& fm_range) {
    
    // Calculate envelopes
    float env_vol_a = env_uniform(time, env_vol_a_settings[0], env_vol_a_settings[1],
                                 env_vol_a_settings[2], env_vol_a_settings[3], env_vol_a_settings[4]);
    float env_vol_b = env_uniform(time, env_vol_b_settings[0], env_vol_b_settings[1],
                                 env_vol_b_settings[2], env_vol_b_settings[3], env_vol_b_settings[4]);
    float env_mod = env_uniform(time, env_mod_settings[0], env_mod_settings[1],
                              env_mod_settings[2], env_mod_settings[3], env_mod_settings[4]);
    
    // Interpolate settings from modulation
    std::vector<float> osc_a_settings_modulated(6);
    std::vector<float> osc_b_settings_modulated(6);
    for (int i = 0; i < 6; ++i) {
        osc_a_settings_modulated[i] = linear_interp(osc_a_settings[i], osc_a_mod_settings[i], env_mod);
        osc_b_settings_modulated[i] = linear_interp(osc_b_settings[i], osc_b_mod_settings[i], env_mod);
    }
    
    // Calculate frequency modulation amount
    float min_fm = 0.005f;
    float max_fm = 0.5f;
    float start_fm = exp_interp(min_fm, max_fm, fm_range[0]);
    float end_fm = exp_interp(min_fm, max_fm, fm_range[1]);
    float env_fm = env_uniform(time, env_fm_setting[0], env_fm_setting[1],
                              env_fm_setting[2], env_fm_setting[3], env_fm_setting[4]);
    float fm_amount = linear_interp(start_fm, end_fm, env_fm);
    
    // Calculate oscillators
    RNG rng_b = rng.split();
    float osc_b = osc_uniform(rng_b, time, osc_b_settings_modulated[0], osc_b_settings_modulated[1],
                             osc_b_settings_modulated[2], osc_b_settings_modulated[3],
                             osc_b_settings_modulated[4], osc_b_settings_modulated[5]);
    
    // A is carrier, B is modulator for FM
    float osc_a = osc_uniform(rng, time, osc_a_settings_modulated[0], osc_a_settings_modulated[1],
                             osc_a_settings_modulated[2], osc_a_settings_modulated[3],
                             osc_a_settings_modulated[4], osc_a_settings_modulated[5],
                             osc_b, fm_amount);
    
    // Multiply by envelopes and add together
    return osc_a * env_vol_a + osc_b * env_vol_b;
}

// Parallel synthesis across multiple times/samples
std::vector<float> synth_parallel(RNG& rng, const std::vector<float>& times,
                                 const std::vector<float>& env_vol_a_settings,
                                 const std::vector<float>& env_vol_b_settings,
                                 const std::vector<float>& env_mod_settings,
                                 const std::vector<float>& osc_a_settings,
                                 const std::vector<float>& osc_b_settings,
                                 const std::vector<float>& osc_a_mod_settings,
                                 const std::vector<float>& osc_b_mod_settings,
                                 const std::vector<float>& env_fm_setting,
                                 const std::vector<float>& fm_range) {
    
    std::vector<float> result(times.size());
    for (size_t i = 0; i < times.size(); ++i) {
        result[i] = synth(rng, times[i], env_vol_a_settings, env_vol_b_settings,
                         env_mod_settings, osc_a_settings, osc_b_settings,
                         osc_a_mod_settings, osc_b_mod_settings, env_fm_setting, fm_range);
    }
    return result;
}

// Synthesize a sample from a set of parameters
std::vector<float> forward(const std::vector<float>& times,
                          const std::map<std::string, std::vector<float>>& params,
                          uint32_t seed = 0) {
    
    RNG rng(seed);
    return synth_parallel(rng, times,
                         params.at("env_vol_a_settings"),
                         params.at("env_vol_b_settings"),
                         params.at("env_mod_settings"),
                         params.at("osc_a_settings"),
                         params.at("osc_b_settings"),
                         params.at("osc_a_mod_settings"),
                         params.at("osc_b_mod_settings"),
                         params.at("env_fm_setting"),
                         params.at("fm_range"));
}

// Generate random settings for the synthesizer
std::map<std::string, std::vector<float>> get_initial_settings(RNG& rng) {
    std::map<std::string, std::vector<float>> params;
    
    params["env_vol_a_settings"] = rng.normal_vector(5);
    params["env_vol_b_settings"] = rng.normal_vector(5);
    params["env_mod_settings"] = rng.normal_vector(5);
    params["osc_a_settings"] = rng.normal_vector(6);
    params["osc_b_settings"] = rng.normal_vector(6);
    params["osc_a_mod_settings"] = rng.normal_vector(6);
    params["osc_b_mod_settings"] = rng.normal_vector(6);
    params["env_fm_setting"] = rng.normal_vector(5);
    params["fm_range"] = rng.normal_vector(2);
    
    return params;
}

// Sigmoid function to convert from real numbers to 0 to 1
float sigmoid(float x) {
    return 0.001f + 0.998f / (1.0f + std::exp(-x));
}

std::vector<float> sigmoid_vector(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}

// Convert from real parameters to 0 to 1 parameters and generate a sample
std::vector<float> sigmoid_forward(const std::vector<float>& times,
                                  const std::map<std::string, std::vector<float>>& params,
                                  uint32_t seed = 0) {
    
    std::map<std::string, std::vector<float>> sigmoid_params;
    for (const auto& [key, value] : params) {
        sigmoid_params[key] = sigmoid_vector(value);
    }
    
    return forward(times, sigmoid_params, seed);
}

int main() {
    for (int seed = 0; seed < 10; ++seed) {
        RNG rng(seed);
        auto params = get_initial_settings(rng);
        
        // A standard sample rate
        int sr = 44100;
        // A length of 2 seconds
        float length = 2.0f;
        // Calculate the number of samples
        int num_samples = static_cast<int>(sr * length);
        // Create an array of the time of each sample
        std::vector<float> times(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            times[i] = static_cast<float>(i) / sr;
        }
        
        // Save the WAV file
        std::string path = "example_sound_" + (seed < 10 ? "0" : "") + std::to_string(seed) + ".wav";
        auto outputs = sigmoid_forward(times, params, seed);
        auto normalized_outputs = normalize_wav(outputs);
        save_wav(path, normalized_outputs, sr);
    }
    
    return 0;
}