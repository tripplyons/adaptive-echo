#include <vector>
#include <string>
#include <utility>
// Writes the data and sample rate to the file
void writeData(const std::string& filename, const std::vector<float>& data, int sample_rate);
// Returns the data and sample rate as a pair
std::pair<std::vector<float>,int> loadData(const std::string& filename);
