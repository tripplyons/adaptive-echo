#pragma once

/**
 * Constants used for audio generation.
 */

namespace adaptive_echo {
namespace constants {

constexpr int NUM_SECONDS = 2;
constexpr int TRAINING_SAMPLE_RATE = 48000;  // Sample rate for optimization (faster)
constexpr int OUTPUT_SAMPLE_RATE = 48000;    // Sample rate for output files
constexpr int NUM_SAMPLES = TRAINING_SAMPLE_RATE * NUM_SECONDS;
constexpr int NUM_SETTINGS = 51;  // Synth settings size (46 + 4 filter + 1 distortion)

// Oscillator parameter indices
constexpr int OSC_A_FREQ_LOW_INDEX = 15;
constexpr int OSC_A_FREQ_HIGH_INDEX = 16;
constexpr int OSC_B_FREQ_LOW_INDEX = 27;
constexpr int OSC_B_FREQ_HIGH_INDEX = 28;

// Effect parameter indices
constexpr int HIGH_PASS_CUTOFF_INDEX = 46;
constexpr int HIGH_PASS_SLOPE_INDEX = 47;
constexpr int LOW_PASS_CUTOFF_INDEX = 48;
constexpr int LOW_PASS_SLOPE_INDEX = 49;
constexpr int DISTORTION_INDEX = 50;

}  // namespace constants
}  // namespace adaptive_echo
