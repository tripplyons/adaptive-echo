#include <iostream>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <thread>
#include "../../plugin/src/WavHandler.hpp"
#include "../../plugin/src/Oscillator.hpp"
void runProgram(int numFiles, char *outputPath, int nThreads);
double** initializeCsv(char* outputPath, int numFiles);
void generateAllFiles(char *outputPath, double **weightArr, int weightArrLen, int nThreads);
void generateFilesForThread(char *outputPath, double **weightArr, int weightArrLen, int threadNum, int nThreads);
void generateFile(const char *outputPath, double* weights);

// Runs on startup, self explanatory
// Many checks to see if args are valid, and then calls rest of code
int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cout << "Usage: adaptive-echo <num-files> <local-path-to-output> <nthreads>" << std::endl;
            return -1;
        }
        try {
            int numFiles = std::stoi(argv[1]);
            if (numFiles <= 0) {
                throw std::invalid_argument("Argument less than or equal to 0");
            }
            int nThreads = std::stoi(argv[3]);
            if (nThreads <= 0) {
                throw std::invalid_argument("Argument less than or equal to 0");
            }
        } catch (const std::exception &e) {
            std::cout << "Usage: argument 1,3 must be a positive integer" << std::endl;
            return -1;
        }
        runProgram(std::stoi(argv[1]), argv[2], std::stoi(argv[3]));
        return 0;
    // A general catch all to avoid ugly popup on uncaught error. Blanket catch all
    } catch (const std::exception &e) {
        std::cout << "An unexpected error occured... exiting safely" << std::endl;
        return -1;
    }
}
// Creates csv then creates files
void runProgram(int numFiles, char *outputPath, int nThreads) {
    double** weightArr = initializeCsv(outputPath, numFiles);
    generateAllFiles(outputPath, weightArr, numFiles, nThreads);
    for (int i = 0; i < numFiles; i++) {
        delete[] weightArr[i];
    }
    delete[] weightArr;
}

// Creates a 2d array of doubles
// Each row is is the list of values associated with
double** initializeCsv(char* outputPath, int numFiles) {
    // Sets up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double** weightArr = new double *[numFiles];
    for (int i = 0; i < numFiles; i++) {
        weightArr[i] = new double[8];
        for (int j = 0; j < 8; j++) {
            // Random weights across the matrix
            weightArr[i][j] = dist(gen);
        }
    }
    std::ofstream ofs(string(outputPath) + string("data.csv"));
    for (int i = 0; i < numFiles; i++) {
        ofs << "file" << (i+1) << ".wav";
        for (int j = 0; j < 8; j++) {
            ofs << "," << weightArr[i][j];
        }
        ofs << "\n";
    }
    ofs.close();
    return weightArr;
}
// ------------------ FILE GENERATION BELOW --------------------------
// Create individual file with given settings
void generateFile(const char *outputPath, double* weights) {
    double maxTime = 5;
    int sampleRate = 44100;
    double timeInterval = 1.0 / sampleRate;
    int nSamples = maxTime * sampleRate;
    vector<int32_t>* data = new vector<int32_t>();
    (*data).reserve(nSamples);
    // Add samples in order by time
    std::mt19937 rng(std::random_device{}());
    for (int i=0; i<nSamples; i++) {
        var time = i * timeInterval;
        var w0 = weights[0]; var w1 = weights[1]; var w2 = weights[2];
        var w3 = weights[3]; var w4 = weights[4]; var w5 = weights[5];
        var w6 = weights[6]; var w7 = weights[7];
        double sample = val(osc_uniform(rng,time,w0,w1,w2,w3,w4,w5,w6,w7));
        double clipped = clamp(sample, -1.0, 1.0);
        int32_t sampleConverted = static_cast<int32_t>(clipped * static_cast<double>(std::numeric_limits<int32_t>::max()));
        (*data).push_back(sampleConverted);
    }
    // Write data to file and free memory
    writeData(string(outputPath), *data, sampleRate);
    delete data;
}
// Executes file generation for files assigned to a specific thread
void generateFilesForThread(char* outputPath, double** weightArr, int weightArrLen, int threadNum, int nThreads) {
    for (int i = threadNum; i < weightArrLen; i+=nThreads) {
        std::string output = "Working on item: " + std::to_string(i + 1) + '\n';
        std::cout << output << std::flush;
        generateFile((string(outputPath) + string("file") + std::to_string(i+1)+string(".wav")).c_str(), weightArr[i]);
    }
}
// Manages which threads generate which files
void generateAllFiles(char* outputPath, double** weightArr, int weightArrLen, int nThreads) {
    std::thread* threads = new std::thread[nThreads];
    for (int i = 0; i < nThreads; i++) {
        threads[i] = std::thread([=]() {
            generateFilesForThread(outputPath, weightArr, weightArrLen, i, nThreads);
        });
    }
    for (int i = 0; i < nThreads; i++) {
        threads[i].join();
    }
    delete[] threads;
}