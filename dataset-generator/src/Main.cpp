#include <iostream>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>

void runProgram(int numFiles, char *outputPath);
double** initializeCsv(char* outputPath, int numFiles);
void generateFile(char* outputPath);

// Since this generator runs without any user input, it can run off of minimal
// external libraries The main dependency this will rely on is a generator from
// the plugin which takes parameters and outputs the sound file
int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cout << "Usage: adaptive-echo <num-files> <local-path-to-output>" << std::endl;
            return -1;
        }
        try {
            int numFiles = std::stoi(argv[1]);
            if (numFiles <= 0) {
                throw std::invalid_argument("Argument less than or equal to 0");
            }
        } catch (const std::exception &e) {
            std::cout << "Usage: argument 1 must be a positive integer" << std::endl;
            return -1;
        }
        runProgram(std::stoi(argv[1]), argv[2]);
        return 0;
    // A general catch all to avoid ugly popup on uncaught error. Blanket catch all
    } catch (const std::exception &e) {
        std::cout << "An unexpected error occured... exiting safely" << std::endl;
        return -1;
    }
}

void runProgram(int numFiles, char *outputPath) {
    initializeCsv(outputPath, numFiles);
    for (int i = 0; i < numFiles; i++) {
        generateFile(outputPath);
    }
}

// Creates a 2d array of doubles
// Each row is is the list of values associated with
double** initializeCsv(char* outputPath, int numFiles) {
    // numKnobs is a constant for the number of values it randomizes for each file
    const int numKnobs = 10;

    // Sets up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double** weightArr = new double *[numFiles];
    for (int i = 0; i < numFiles; i++) {
        weightArr[i] = new double[numKnobs];
        for (int j = 0; j < numKnobs; j++) {
            // Random weights across the matrix
            weightArr[i][j] = dist(gen);
        }
    }
    std::ofstream ofs(outputPath);
    for (int i = 0; i < numFiles; i++) {
        ofs << "file" << (i+1) << ".wav";
        for (int j = 0; j < numKnobs; j++) {
            ofs << "," << weightArr[i][j];
        }
        ofs << "\n";
    }
    ofs.close();
    return weightArr;
}

void generateFile(char* outputPath) {
    // This is pretty much placeholder since the implementation is TBD, but it will latch onto the main plugin to do its calculations
    // Critical that this gets implemented as a multithreaded operation, since the idea to produce massive number of files for a dataset
}