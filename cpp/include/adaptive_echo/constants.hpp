#pragma once

/**
 * Constants used for audio generation.
 */

namespace adaptive_echo {
namespace constants {

constexpr int NUM_SECONDS = 2;
constexpr int TRAINING_SAMPLE_RATE = 16384;  // Sample rate for optimization (faster)
constexpr int OUTPUT_SAMPLE_RATE = 48000;    // Sample rate for output files
constexpr int NUM_SAMPLES = TRAINING_SAMPLE_RATE * NUM_SECONDS;
constexpr int NUM_SETTINGS = 50;  // Synth settings size (46 + 4 filter parameters)

// Filter parameter indices
constexpr int HIGH_PASS_CUTOFF_INDEX = 46;
constexpr int HIGH_PASS_SLOPE_INDEX = 47;
constexpr int LOW_PASS_CUTOFF_INDEX = 48;
constexpr int LOW_PASS_SLOPE_INDEX = 49;

}  // namespace constants
}  // namespace adaptive_echo
