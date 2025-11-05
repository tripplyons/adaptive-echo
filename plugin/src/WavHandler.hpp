#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

template <typename T> void writeLittleEndian(std::ofstream &file, T value) {
    for (size_t i = 0; i < sizeof(T); i++) {
        file.put(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

// Writes the data and sample rate to the file
void writeData(const std::string &filename, const std::vector<int32_t> &data,
               int sample_rate) {
    // The below header looks really scary but its just hex representations of
    // different header aspects Because it is all just being dumped to the file
    // in binary
    uint32_t FileTypeBlocID = 0x46464952; // "RIFF"
    uint32_t FileFormatID = 0x45564157;   // "WAVE"
    uint32_t FormatBlocID = 0x20746D66;   // "fmt "
    uint32_t BlocSize =
        0x10; // Chunk size minus 8 bytes, which is 16 bytes here
    uint16_t AudioFormat = 0x1;       // PCM = 1
    uint16_t NbrChannels = 0x1;       // Mono = 1, Stereo = 2
    uint32_t Frequency = sample_rate; // Sample rate... normally 48kHz
    uint16_t BitsPerSample =
        0x20; // Number of bits per sample ... after byte per bloc
    uint16_t BytePerBloc =
        NbrChannels * BitsPerSample /
        8; // Number of bytes per block ... after byte per sec
    uint32_t BytePerSec =
        Frequency * BytePerBloc;      // Number of bytes to read per second
    uint32_t DataBlocID = 0x61746164; // "data"
    uint32_t DataSize = data.size() * NbrChannels * BitsPerSample /
                        8;             // Size of the data section in bytes
    uint32_t FileSize = DataSize + 36; // Overall file size - 8 bytes
    std::ofstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open file for writing: " +
                                 filename);
    // Output Master RIFF chunk
    writeLittleEndian(file, FileTypeBlocID);
    writeLittleEndian(file, FileSize);
    writeLittleEndian(file, FileFormatID);
    // Output chunk describing the data format
    writeLittleEndian(file, FormatBlocID);
    writeLittleEndian(file, BlocSize);
    writeLittleEndian(file, AudioFormat);
    writeLittleEndian(file, NbrChannels);
    writeLittleEndian(file, Frequency);
    writeLittleEndian(file, BytePerSec);
    writeLittleEndian(file, BytePerBloc);
    writeLittleEndian(file, BitsPerSample);
    // Output chunk containing the sampled data
    writeLittleEndian(file, DataBlocID);
    writeLittleEndian(file, DataSize);

    for (size_t i = 0; i < data.size(); i++) {
        writeLittleEndian(file, data[i]);
    }
    file.close();
    return;
}

// Returns the data and sample rate as a pair
std::pair<std::vector<int32_t> *, int> loadData(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open file for reading: " +
                                 filename);
    // Read the byte data for wavelength in as a uint32_t
    file.seekg(24, std::ios::beg);
    uint32_t sampleRate = 0;
    file.read(reinterpret_cast<char *>(&sampleRate), sizeof(sampleRate));
    // Get vector<int32_t> from bytes 45 onward
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    if (fileSize < 45)
        throw std::runtime_error("File too small to be a valid WAV file: " +
                                 filename);
    std::streampos pos(44);
    std::size_t count = (fileSize - pos) / sizeof(int32_t);
    file.seekg(44, std::ios::beg);
    std::vector<int32_t> *data = new std::vector<int32_t>(count);
    file.read(reinterpret_cast<char *>((*data).data()),
              count * sizeof(int32_t));
    if (!file) {
        delete data;
        throw std::runtime_error("Reading in the vector<int32_t> failed: " +
                                 filename);
    }
    return {data, sampleRate};
}