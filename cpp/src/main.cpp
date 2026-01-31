/**
 * Generate sound by optimizing synthesizer parameters using Differential Evolution on STFT
 * similarity.
 *
 * This is a C++ port of the Python generate_sound.py from adaptive_echo_jax.
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/hybrid_evolution.hpp"
#include "adaptive_echo/loss.hpp"
#include "adaptive_echo/resample.hpp"
#include "adaptive_echo/synth.hpp"

// Simple WAV file writer (RIFF/WAVE format)
struct WavWriter {
    static bool write(const std::string& filename, const std::vector<float>& samples,
                      int sample_rate) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;

        int num_samples = static_cast<int>(samples.size());
        int num_channels = 1;
        int bits_per_sample = 16;
        int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
        int block_align = num_channels * bits_per_sample / 8;
        int data_size = num_samples * num_channels * bits_per_sample / 8;

        // RIFF header
        file.write("RIFF", 4);
        int chunk_size = 36 + data_size;
        file.write(reinterpret_cast<const char*>(&chunk_size), 4);
        file.write("WAVE", 4);

        // fmt subchunk
        file.write("fmt ", 4);
        int subchunk1_size = 16;
        short audio_format = 1;  // PCM
        file.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
        file.write(reinterpret_cast<const char*>(&audio_format), 2);
        file.write(reinterpret_cast<const char*>(&num_channels), 2);
        file.write(reinterpret_cast<const char*>(&sample_rate), 4);
        file.write(reinterpret_cast<const char*>(&byte_rate), 4);
        file.write(reinterpret_cast<const char*>(&block_align), 2);
        file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

        // data subchunk
        file.write("data", 4);
        file.write(reinterpret_cast<const char*>(&data_size), 4);

        // Write samples (convert float to int16)
        for (float sample : samples) {
            short s = static_cast<short>(std::clamp(sample * 32767.0f, -32768.0f, 32767.0f));
            file.write(reinterpret_cast<const char*>(&s), 2);
        }

        return file.good();
    }
};

// Load and preprocess target audio from file
std::vector<float> load_target_audio(const std::string& audio_path, int target_length) {
    std::ifstream file(audio_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Audio file not found: " + audio_path);
    }

    // Check if it's a WAV file
    char riff[4];
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("Only WAV files are supported: " + audio_path);
    }

    // Parse WAV header
    file.seekg(8);
    char wave[4];
    file.read(wave, 4);

    // Find fmt chunk
    while (true) {
        char chunk_id[4];
        int chunk_size;
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::string(chunk_id, 4) == "fmt ") {
            short audio_format, num_channels;
            int sample_rate, byte_rate;
            short block_align, bits_per_sample;
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.read(reinterpret_cast<char*>(&byte_rate), 4);
            file.read(reinterpret_cast<char*>(&block_align), 2);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

            // Skip remaining fmt chunk data
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }

            // Now find data chunk
            while (true) {
                char data_id[4];
                int data_size;
                file.read(data_id, 4);
                file.read(reinterpret_cast<char*>(&data_size), 4);

                if (std::string(data_id, 4) == "data") {
                    // Read audio data
                    int num_samples = data_size / (num_channels * bits_per_sample / 8);
                    std::vector<float> audio(num_samples);

                    if (bits_per_sample == 16) {
                        for (int i = 0; i < num_samples; ++i) {
                            short s = 0;
                            float sum = 0;
                            for (int ch = 0; ch < num_channels; ++ch) {
                                file.read(reinterpret_cast<char*>(&s), 2);
                                sum += s / 32768.0f;
                            }
                            audio[i] = sum / num_channels;  // Average channels
                        }
                    } else if (bits_per_sample == 24) {
                        for (int i = 0; i < num_samples; ++i) {
                            unsigned char buf[3];
                            float sum = 0;
                            for (int ch = 0; ch < num_channels; ++ch) {
                                file.read(reinterpret_cast<char*>(buf), 3);
                                int s = (buf[2] & 0x80)
                                            ? (0xFF << 24) | (buf[2] << 16) | (buf[1] << 8) | buf[0]
                                            : (buf[2] << 16) | (buf[1] << 8) | buf[0];
                                sum += s / 8388608.0f;
                            }
                            audio[i] = sum / num_channels;
                        }
                    } else if (bits_per_sample == 32) {
                        for (int i = 0; i < num_samples; ++i) {
                            int s = 0;
                            float sum = 0;
                            for (int ch = 0; ch < num_channels; ++ch) {
                                file.read(reinterpret_cast<char*>(&s), 4);
                                sum += s / 2147483648.0f;
                            }
                            audio[i] = sum / num_channels;
                        }
                    }

                    // Resample if needed (FFT-based, matches scipy.signal.resample)
                    if (sample_rate != adaptive_echo::constants::TRAINING_SAMPLE_RATE) {
                        std::cout << "Resampling from " << sample_rate << " Hz to "
                                  << adaptive_echo::constants::TRAINING_SAMPLE_RATE << " Hz"
                                  << std::endl;

                        float ratio =
                            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE) /
                            sample_rate;
                        int new_length = static_cast<int>(audio.size() * ratio);
                        audio = adaptive_echo::resample_fft(audio,
                                                            static_cast<std::size_t>(new_length));
                    }

                    // Trim or pad to target length
                    if (static_cast<int>(audio.size()) > target_length) {
                        audio.resize(target_length);
                    } else if (static_cast<int>(audio.size()) < target_length) {
                        audio.resize(target_length, 0.0f);
                    }

                    // Normalize
                    float mean = std::accumulate(audio.begin(), audio.end(), 0.0f) / audio.size();
                    for (auto& s : audio) s -= mean;

                    float max_val = 0.0f;
                    for (auto s : audio) max_val = std::max(max_val, std::abs(s));
                    if (max_val > 0) {
                        for (auto& s : audio) s /= max_val;
                    }

                    return audio;
                } else {
                    // Skip this chunk
                    file.seekg(data_size, std::ios::cur);
                }
            }
        } else {
            // Skip this chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [target_audio.wav] [options]\n\n"
              << "Optimize synthesizer parameters using Differential Evolution.\n\n"
              << "Options:\n"
              << "  --population <n>     Population size (default: 128)\n"
              << "  --iterations <n>     Number of iterations (default: 100)\n"
              << "  --trials <n>         Number of trials per parent (default: 12)\n"
              << "  --stft-weight <f>    Weight for STFT loss (default: 1.0)\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " input.wav\n"
              << "  " << program_name << " input.wav --population 128 --iterations 200\n";
}

int main(int argc, char* argv[]) {
    using namespace adaptive_echo;
    using namespace adaptive_echo::constants;

    // Default parameters - match JAX CLI defaults
    std::string target_path;
    int population_size = 64;        // Match JAX CLI default
    int num_iterations = 100;        // Match JAX CLI default
    int num_trials_per_parent = 64;  // Match JAX CLI default
    float stft_weight = 1.0f;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--population" && i + 1 < argc) {
            population_size = std::atoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            num_iterations = std::atoi(argv[++i]);
        } else if (arg == "--trials" && i + 1 < argc) {
            num_trials_per_parent = std::atoi(argv[++i]);
        } else if (arg == "--stft-weight" && i + 1 < argc) {
            stft_weight = std::atof(argv[++i]);
        } else if (arg[0] != '-') {
            target_path = arg;
        }
    }

    // Create time arrays - match Python np.linspace(0, NUM_SECONDS, NUM_SAMPLES)
    std::vector<float> time_train(NUM_SAMPLES);
    if (NUM_SAMPLES > 1) {
        float step = static_cast<float>(NUM_SECONDS) / (NUM_SAMPLES - 1);
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            time_train[i] = static_cast<float>(i) * step;
        }
    } else if (NUM_SAMPLES == 1) {
        time_train[0] = 0.0f;
    }

    // Load or create target audio
    std::vector<float> target_audio;
    if (!target_path.empty()) {
        std::cout << "Loading target audio from: " << target_path << std::endl;
        try {
            target_audio = load_target_audio(target_path, NUM_SAMPLES);
        } catch (const std::exception& e) {
            std::cerr << "Error loading audio: " << e.what() << std::endl;
            return 1;
        }
    } else {
        // Default: 440 Hz sine wave
        std::cout << "No target audio specified, using 440 Hz sine wave." << std::endl;
        target_audio.resize(NUM_SAMPLES);
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            target_audio[i] = std::sin(2.0f * M_PI * 440.0f * time_train[i]);
        }
    }

    // Create loss function
    std::cout << "Precomputing target features (STFT)..." << std::endl;
    LossFunction<float> loss_fn(target_audio, stft_weight);
    // Sanity check: loss of target against itself should be near zero.
    float self_loss = loss_fn(target_audio);
    std::cout << "Self loss (target vs target): " << self_loss << std::endl;

    // Create synth function wrapper
    auto synth_fn = [](const std::vector<float>& settings, const std::vector<float>& time) {
        return synth(settings, time);
    };

    // Run optimization
    std::cout << "Running Hybrid Evolution optimization with STFT loss (weight=" << stft_weight
              << ")..." << std::endl;

    auto result =
        run_hybrid_evolution(loss_fn, time_train, synth_fn, population_size, num_iterations,
                             0.7f,   // sigma_init
                             0.05f,  // sigma_min
                             2.0f,   // sigma_max
                             0.8f,   // F_scale_start
                             0.2f,   // F_scale_end
                             0.8f,   // crossover_rate_start
                             0.4f,   // crossover_rate_end
                             0.1f,   // mutation_rate
                             0.25f,  // mutation_sigma
                             0.1f,   // elite_fraction
                             -1.0f   // No time limit
        );

    std::cout << "\nFinal best loss: " << result.best_loss << std::endl;

    // Generate output audio at output sample rate
    // Match Python: np.linspace(0, num_seconds, int(num_seconds * output_sample_rate))
    int eval_samples = NUM_SECONDS * OUTPUT_SAMPLE_RATE;
    std::vector<float> eval_time(eval_samples);
    if (eval_samples > 1) {
        float step = static_cast<float>(NUM_SECONDS) / (eval_samples - 1);
        for (int i = 0; i < eval_samples; ++i) {
            eval_time[i] = static_cast<float>(i) * step;
        }
    } else if (eval_samples == 1) {
        eval_time[0] = 0.0f;
    }

    auto eval_audio = synth(result.best_settings, eval_time);

    // Normalize audio to prevent clipping (match Python behavior)
    float max_val = 0.0f;
    for (float s : eval_audio) {
        max_val = std::max(max_val, std::abs(s));
    }
    if (max_val > 1.0f) {
        std::cout << "Normalizing audio (max value: " << max_val << ")" << std::endl;
        for (float& s : eval_audio) {
            s /= max_val;
        }
    }

    // Save to WAV file
    if (WavWriter::write("eval_audio_de.wav", eval_audio, OUTPUT_SAMPLE_RATE)) {
        std::cout << "Saved: eval_audio_de.wav" << std::endl;
    } else {
        std::cerr << "Error: Failed to write output file" << std::endl;
        return 1;
    }

    return 0;
}
