#include "WavHandler.hpp"
#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include <fstream>

// Writes the data and sample rate to the file
void writeData(const std::string& filename, const std::vector<int32_t>& data, int sample_rate) {
    // The below header looks really scary but its just hex representations of different header aspects
    // Because it is all just being dumped to the file in binary
    uint32_t FileTypeBlocID = 0x52494646;                               // "RIFF"
    uint32_t FileFormatID = 0x57415645;                                 // "WAVE"
    uint32_t FormatBlocID = 0x666D7420;                                 // "fmt "
    uint32_t BlocSize = 0x10;                                           // Chunk size minus 8 bytes, which is 16 bytes here
    uint16_t AudioFormat = 0x1;                                         // PCM = 1
    uint16_t NbrChannels = 0x1;                                         // Mono = 1, Stereo = 2
    uint32_t Frequency = sample_rate;                                   // Sample rate... normally 48kHz             
    uint16_t BitsPerSample = 0x20;                                      // Number of bits per sample ... after byte per bloc
    uint16_t BytePerBloc = NbrChannels * BitsPerSample / 8;             // Number of bytes per block ... after byte per sec
    uint32_t BytePerSec = Frequency * BytePerBloc;                      // Number of bytes to read per second
    uint32_t DataBlocID = 0x64617461;                                   // "data"
    uint32_t DataSize = data.size() * NbrChannels * BitsPerSample / 8;  // Size of the data section in bytes
    uint32_t FileSize = DataSize + 36; // Overall file size - 8 bytes
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file for writing: " + filename);
    // Output Master RIFF chunk
    file.write(reinterpret_cast<const char *>(&FileTypeBlocID), sizeof(FileTypeBlocID));
    file.write(reinterpret_cast<const char *>(&FileSize), sizeof(FileSize));
    file.write(reinterpret_cast<const char *>(&FileFormatID), sizeof(FileFormatID));
    // Output chunk describing the data format
    file.write(reinterpret_cast<const char *>(&FormatBlocID), sizeof(FormatBlocID));
    file.write(reinterpret_cast<const char *>(&BlocSize), sizeof(BlocSize));
    file.write(reinterpret_cast<const char *>(&AudioFormat), sizeof(AudioFormat));
    file.write(reinterpret_cast<const char *>(&NbrChannels), sizeof(NbrChannels));
    file.write(reinterpret_cast<const char *>(&Frequency), sizeof(Frequency));
    file.write(reinterpret_cast<const char *>(&BytePerSec), sizeof(BytePerSec));
    file.write(reinterpret_cast<const char *>(&BytePerBloc), sizeof(BytePerBloc));
    file.write(reinterpret_cast<const char *>(&BitsPerSample), sizeof(BitsPerSample));
    // Output chunk containing the sampled data
    file.write(reinterpret_cast<const char *>(&DataBlocID), sizeof(DataBlocID));
    file.write(reinterpret_cast<const char *>(&DataSize), sizeof(DataSize));
    file.write(reinterpret_cast<const char*>(data.data()),data.size() * sizeof(int32_t));
    file.close();
    return;
}

// Returns the data and sample rate as a pair
std::pair<std::vector<int32_t>*, int> loadData(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file for reading: " + filename);
    // Read the byte data for wavelength in as a uint32_t
    file.seekg(25, std::ios::beg);
    uint32_t sampleRate = 0;
    file.read(reinterpret_cast<char *>(&sampleRate), sizeof(sampleRate));
    // Get vector<int32_t> from bytes 45 onward
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    if (fileSize < 45) throw std::runtime_error("File too small to be a valid WAV file: " + filename);
    std::size_t count = (fileSize-45) / sizeof(int32_t);
    file.seekg(45, std::ios::beg);
    std::vector<int32_t>* data = new std::vector<int32_t>(count);
    file.read(reinterpret_cast<char*>((*data).data()), count * sizeof(int32_t));
    if (!file) {
        delete data;
        throw std::runtime_error("Reading in the vector<int32_t> failed: " + filename);
    }
    return {data, sampleRate};
}
