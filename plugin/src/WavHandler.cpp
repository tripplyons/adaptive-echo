#include "WavHandler.hpp"
#include <vector>
#include <string>
#include <utility>
#include <cstdint>

// Writes the data and sample rate to the file
void writeData(const std::string &filename, const std::vector<int32_t> &data,
                   int sample_rate) {
    // The below header looks really scary but its just hex representations of different header aspects
    // Because it is all just being dumped to the file in binary
    uint32_t FileTypeBlocID = 0x52494646; // "RIFF"
    uint32_t FileFormatID = 0x57415645; // "WAVE"
    uint32_t FormatBlocID = 0x666D7420;   // "fmt "
    uint32_t BlocSize = 0x10;             // Chunk size minus 8 bytes, which is 16 bytes here
    uint16_t AudioFormat = 0x1; // PCM = 1
    uint16_t NbrChannels = 0x1;           // Mono = 1, Stereo = 2
    uint32_t Frequency = sample_rate;
    uint16_t BitsPerSample = 0x20; // Number of bits per sample ... after byte per bloc
    uint16_t BytePerBloc = NbrChannels * BitsPerSample / 8; // Number of bytes per block ... after byte per sec
    uint32_t BytePerSec = Frequency * BytePerBloc; // Number of bytes to read per second
    uint32_t DataBlocID = 0x64617461; // "data"
    uint32_t DataSize = data.size() * NbrChannels * BitsPerSample / 8; // Size of the data section in bytes
    uint32_t FileSize = DataSize + 36; // Overall file size - 8 bytes
}
    // Returns the data and sample rate as a pair
std::pair<std::vector<float>, int> loadData(const std::string &filename);
