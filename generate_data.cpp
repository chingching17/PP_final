#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

void generateMatrix(int row, int col, int percent, ofstream &file) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            int random = rand() % 100 + 1; // Generate a random number between 1 and 100
            if (random <= percent) {
                file << rand() % 1000;
            } else {
                file << "0";
            }
            if (j < col - 1) {
                file << " ";
            }
        }
        file << endl;
    }
    for (int j = 0; j < col; ++j){
        int num = rand() % 100;
        file << num << endl;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <number_of_data> <matrix_size> <percent_1> <mode>" << endl;
        return 1;
    }

    int numData = atoi(argv[1]);
    int matrixSize = atoi(argv[2]);
    int percent_for_1 = atoi(argv[3]);
    int mode = atoi(argv[4]);

    srand(time(0)); // Seed for random number generation

    ofstream outputFile("matrix_data.txt");

    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open the output file." << endl;
        return 1;
    }

    outputFile << numData << endl;

    for (int i = 0; i < numData; ++i) {
        if (mode == 1){
            int row = rand() % matrixSize + 1;
            int col = rand() % matrixSize + 1;
            outputFile << row << " " << col << " " << "1" << endl;
            generateMatrix(row, col, percent_for_1, outputFile);
            outputFile << endl;
        }
        if(mode == 2){
            outputFile << matrixSize << " " << matrixSize << " " << "1" << endl;
            generateMatrix(matrixSize, matrixSize, percent_for_1, outputFile);
            outputFile << endl;
        }
    }

    outputFile.close();

    cout << "Matrix data has been written to matrix_data.txt" << endl;

    return 0;
}
