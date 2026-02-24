#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "adaptive_echo/cmaes_optimizer.hpp"
#include "adaptive_echo/constants.hpp"
#include "adaptive_echo/loss.hpp"
#include "adaptive_echo/resample.hpp"
#include "adaptive_echo/synth.hpp"

namespace adaptive_echo {

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

        // Write audio data (16-bit PCM)
        for (float sample : samples) {
            short sample_16bit =
                static_cast<short>(std::clamp(sample * 32767.0f, -32768.0f, 32767.0f));
            file.write(reinterpret_cast<const char*>(&sample_16bit), 2);
        }

        return true;
    }
};

// Load and preprocess target audio from file
std::vector<float> load_target_audio(const std::string& audio_path, int target_length) {
    std::ifstream file(audio_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Audio file not found: " + audio_path);
    }

    char riff[4];
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("Only WAV files are supported: " + audio_path);
    }

    file.seekg(8);
    char wave[4];
    file.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV file: " + audio_path);
    }

    // Search for fmt and data chunks
    short num_channels = 0;
    int sample_rate = 0;
    short bits_per_sample = 0;
    std::vector<float> audio;

    while (file.good()) {
        char chunk_id[4];
        int chunk_size;
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::string(chunk_id, 4) == "fmt ") {
            short audio_format;
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.seekg(6, std::ios::cur);  // skip byte_rate, block_align
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (chunk_size > 16) file.seekg(chunk_size - 16, std::ios::cur);
        } else if (std::string(chunk_id, 4) == "data") {
            int bytes_per_sample = bits_per_sample / 8;
            int num_samples = chunk_size / (num_channels * bytes_per_sample);
            audio.resize(num_samples);

            for (int i = 0; i < num_samples; ++i) {
                float sum = 0;
                for (int ch = 0; ch < num_channels; ++ch) {
                    if (bits_per_sample == 16) {
                        short s;
                        file.read(reinterpret_cast<char*>(&s), 2);
                        sum += s / 32768.0f;
                    } else if (bits_per_sample == 24) {
                        unsigned char b[3];
                        file.read(reinterpret_cast<char*>(b), 3);
                        int s = (b[2] << 16) | (b[1] << 8) | b[0];
                        if (s & 0x800000) s |= 0xFF000000;
                        sum += s / 8388608.0f;
                    } else {
                        file.seekg(bytes_per_sample, std::ios::cur);
                    }
                }
                audio[i] = sum / num_channels;
            }
            break;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (audio.empty()) throw std::runtime_error("No audio data found in " + audio_path);

    // Resample
    if (sample_rate != adaptive_echo::constants::TRAINING_SAMPLE_RATE) {
        float ratio =
            static_cast<float>(adaptive_echo::constants::TRAINING_SAMPLE_RATE) / sample_rate;
        int new_length = static_cast<int>(audio.size() * ratio);
        audio = adaptive_echo::resample_fft(audio, static_cast<size_t>(new_length));
    }

    // Trim/Pad
    if (static_cast<int>(audio.size()) > target_length) {
        audio.resize(target_length);
    } else {
        audio.resize(target_length, 0.0f);
    }

    // Normalize
    float max_val = 0.0f;
    for (float s : audio) max_val = std::max(max_val, std::abs(s));
    if (max_val > 0) {
        for (float& s : audio) s /= max_val;
    }

    return audio;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [target_audio.wav] [options]\n\n"
              << "Options:\n"
              << "  -p, --population <n>       Population size (lambda)\n"
              << "  -s, --sigma <f>            Initial sigma for restarts\n"
              << "  -t, --time-limit <f>       Time limit in seconds\n"
              << "  -o, --output <path>        Output WAV file path\n"
              << "      --json                 Output results as JSON to stdout\n"
              << "  -h, --help                 Show this help message\n";
}

}  // namespace adaptive_echo

int main(int argc, char* argv[]) {
    using namespace adaptive_echo;
    using namespace adaptive_echo::constants;

    std::string target_path;
    std::string output_path = "output.wav";
    int population_size = 32;
    float initial_sigma = 3.0f;
    float time_limit = 60.0f;
    bool output_json = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "--population" || arg == "-p") && i + 1 < argc) {
            population_size = std::atoi(argv[++i]);
        } else if ((arg == "--sigma" || arg == "-s") && i + 1 < argc) {
            initial_sigma = std::atof(argv[++i]);
        } else if ((arg == "--time-limit" || arg == "-t") && i + 1 < argc) {
            time_limit = std::atof(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--json") {
            output_json = true;
        } else if (arg[0] != '-') {
            target_path = arg;
        }
    }

    std::vector<float> time_train(NUM_SAMPLES);
    float step = static_cast<float>(NUM_SECONDS) / (NUM_SAMPLES - 1);
    for (int i = 0; i < NUM_SAMPLES; ++i) time_train[i] = i * step;

    std::vector<float> target_audio;
    if (!target_path.empty()) {
        if (!output_json) {
            std::cout << "Loading target: " << target_path << std::endl;
        }
        target_audio = load_target_audio(target_path, NUM_SAMPLES);

        // Normalize input to 0.5 max amplitude
        float target_max = 0;
        for (float s : target_audio) target_max = std::max(target_max, std::abs(s));
        if (target_max > 0) {
            float input_scale = 0.5f / target_max;
            for (float& s : target_audio) s *= input_scale;
        }
    } else {
        if (!output_json) {
            std::cout << "No target, using 440Hz sine." << std::endl;
        }
        target_audio.resize(NUM_SAMPLES);
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            target_audio[i] = std::sin(2.0f * M_PI * 440.0f * time_train[i]);
        }
    }

    LossFunction<float> loss_fn(target_audio);
    auto synth_fn = [](const std::vector<float>& settings, const std::vector<float>& time) {
        return synth(settings, time);
    };

    if (!output_json) {
        std::cout << "Optimizing..." << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = run_cmaes_optimization<float>(loss_fn, time_train, synth_fn, population_size,
                                                initial_sigma, time_limit, 10000, !output_json);
    float best_loss = result.best_loss;
    std::vector<float> best_settings = result.best_settings;
    int iterations_completed = result.iterations_completed;

    auto end_time = std::chrono::high_resolution_clock::now();

    float time_elapsed = std::chrono::duration<float>(end_time - start_time).count();

    if (!output_json) {
        std::cout << "Best loss: " << best_loss << std::endl;
    }

    int eval_samples = NUM_SECONDS * OUTPUT_SAMPLE_RATE;
    std::vector<float> eval_time(eval_samples);
    float eval_step = static_cast<float>(NUM_SECONDS) / (eval_samples - 1);
    for (int i = 0; i < eval_samples; ++i) eval_time[i] = i * eval_step;

    auto eval_audio = synth(best_settings, eval_time);

    float max_val = 0;
    for (float s : eval_audio) max_val = std::max(max_val, std::abs(s));
    if (max_val > 0) {
        float scale = 0.95f / max_val;
        for (float& s : eval_audio) s *= scale;
    }

    if (WavWriter::write(output_path, eval_audio, OUTPUT_SAMPLE_RATE)) {
        if (!output_json) {
            std::cout << "Saved: " << output_path << std::endl;
        }
    }

    if (output_json) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n";
        std::cout << "  \"best_loss\": " << best_loss << ",\n";
        std::cout << "  \"best_settings\": [";
        for (size_t i = 0; i < best_settings.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << best_settings[i];
        }
        std::cout << "],\n";
        std::cout << "  \"iterations_completed\": " << iterations_completed << ",\n";
        std::cout << "  \"time_elapsed\": " << time_elapsed << ",\n";
        std::cout << "  \"population_size\": " << population_size << ",\n";
        std::cout << "  \"initial_sigma\": " << initial_sigma << ",\n";
        std::cout << "  \"time_limit\": " << time_limit << "\n";
        std::cout << "}" << std::endl;
        std::cout.flush();
    }

    return 0;
}
