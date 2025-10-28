#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
// Writes the data and sample rate to the file
void writeData(const std::string &filename, const std::vector<int32_t> &data,
               int sample_rate);
// Returns the data and sample rate as a pair
std::pair<std::vector<int32_t> *, int> loadData(const std::string &filename);