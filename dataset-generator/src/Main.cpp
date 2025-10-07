#include <iostream>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <thread>
void runProgram(int numFiles, char *outputPath, int nWeights, int nThreads);
double** initializeCsv(char* outputPath, int numFiles, int nWeights);
void generateAllFiles(char *outputPath, double **weightArr, int weightArrLen, int rowLen, int nThreads);
void generateFilesForThread(char *outputPath, double **weightArr, int weightArrLen, int rowLen, int threadNum, int nThreads);
void generateFile(char *outputPath, double* weights, int rowLen);

// Runs on startup, self explanatory
// Many checks to see if args are valid, and then calls rest of code
int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cout << "Usage: adaptive-echo <num-files> <local-path-to-output> <nweights> <nthreads>" << std::endl;
            return -1;
        }
        try {
            int numFiles = std::stoi(argv[1]);
            if (numFiles <= 0) {
                throw std::invalid_argument("Argument less than or equal to 0");
            }
            int nWeights = std::stoi(argv[3]);
            if (nWeights <= 0) {
                throw std::invalid_argument("Argument less than or equal to 0");
            }
            int nThreads = std::stoi(argv[4]);
            if (nThreads <= 0) {
                throw std::invalid_argument("Argument less than or equal to 0");
            }
        } catch (const std::exception &e) {
            std::cout << "Usage: argument 1,3,4 must be a positive integer" << std::endl;
            return -1;
        }
        runProgram(std::stoi(argv[1]), argv[2], std::stoi(argv[3]), std::stoi(argv[4]));
        return 0;
    // A general catch all to avoid ugly popup on uncaught error. Blanket catch all
    } catch (const std::exception &e) {
        std::cout << "An unexpected error occured... exiting safely" << std::endl;
        return -1;
    }
}
// Creates csv then creates files
void runProgram(int numFiles, char *outputPath, int nWeights, int nThreads) {
    double** weightArr = initializeCsv(outputPath, numFiles, nWeights);
    generateAllFiles(outputPath, weightArr, numFiles, nWeights, nThreads);
    for (int i = 0; i < numFiles; i++) {
        delete[] weightArr[i];
    }
    delete[] weightArr;
}

// Creates a 2d array of doubles
// Each row is is the list of values associated with
double** initializeCsv(char* outputPath, int numFiles, int nWeights) {

    // Sets up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double** weightArr = new double *[numFiles];
    for (int i = 0; i < numFiles; i++) {
        weightArr[i] = new double[nWeights];
        for (int j = 0; j < nWeights; j++) {
            // Random weights across the matrix
            weightArr[i][j] = dist(gen);
        }
    }
    std::ofstream ofs(outputPath);
    for (int i = 0; i < numFiles; i++) {
        ofs << "file" << (i+1) << ".wav";
        for (int j = 0; j < nWeights; j++) {
            ofs << "," << weightArr[i][j];
        }
        ofs << "\n";
    }
    ofs.close();
    return weightArr;
}
// ------------------ FILE GENERATION BELOW --------------------------
// Create individual file with given settings
void generateFile(char *outputPath, double* weights, int rowLen) {

}
// Executes file generation for files assigned to a specific thread
void generateFilesForThread(char* outputPath, double** weightArr, int weightArrLen, int rowLen, int threadNum, int nThreads) {
    for (int i = threadNum; i < weightArrLen; i+=nThreads) {
        std::string output = "Working on item: " + std::to_string(i + 1) + '\n';
        std::cout << output << std::flush;
        generateFile(outputPath, weightArr[i], rowLen);
    }
}
// Manages which threads generate which files
void generateAllFiles(char* outputPath, double** weightArr, int weightArrLen, int rowLen, int nThreads) {
    std::thread* threads = new std::thread[nThreads];
    for (int i = 0; i < nThreads; i++) {
        threads[i] = std::thread([=]() {
            generateFilesForThread(outputPath, weightArr, weightArrLen, rowLen, i, nThreads);
        });
    }
    for (int i = 0; i < nThreads; i++) {
        threads[i].join();
    }
    delete[] threads;
}