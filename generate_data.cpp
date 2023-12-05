#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

void generateMatrix(int row, int col, int matrixSize, int percent, ofstream &file) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            int random = rand() % matrixSize + 5; // Generate a random number between 5 and matrixSize
            if (random <= percent) {
                file << "1 ";
            } else {
                file << "0 ";
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
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <number_of_data> <matrix_size> <percent_1>" << endl;
        return 1;
    }

    int numData = atoi(argv[1]);
    int matrixSize = atoi(argv[2]);
    int percent_for_1 = atoi(argv[3]);

    srand(time(0)); // Seed for random number generation

    ofstream outputFile("matrix_data.txt");

    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open the output file." << endl;
        return 1;
    }

    for (int i = 0; i < numData; ++i) {
        int row = rand() % matrixSize + 1;
        int col = rand() % matrixSize + 1;
        outputFile << row << " " << col << " " << "1" << endl;
        generateMatrix(row, col, matrixSize, percent_for_1, outputFile);
        outputFile << endl;
    }

    outputFile.close();

    cout << "Matrix data has been written to matrix_data.txt" << endl;

    return 0;
}
