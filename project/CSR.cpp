// CPP program to find sparse matrix rep-
// resentation using CSR
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

typedef std::vector<int> vi;

typedef vector<vector<int> > matrix;

// Utility Function to print a Matrix
void printMatrix(const matrix& M)
{
	int m = M.size();
	int n = M[0].size();
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) 
			cout << M[i][j] << " ";	 
		cout << endl;
	}
}

// Utility Function to print A, IA, JA vectors
// with some decoration.
void printVector(const vi& V, char* msg)
{

	cout << msg << "[ ";
	for_each(V.begin(), V.end(), [](int a) {
		cout << a << " ";
	});
	cout << "]" << endl;
}

// Generate the three vectors A, IA, JA 
void sparesify(const matrix& M)
{
	int m = M.size();
	int n = M[0].size(), i, j;
	vi A;
	vi IA = { 0 }; // IA matrix has N+1 rows
	vi JA;
	int NNZ = 0;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (M[i][j] != 0) {
				A.push_back(M[i][j]);
				JA.push_back(j);

				// Count Number of Non Zero 
				// Elements in row i
				NNZ++;
			}
		}
		IA.push_back(NNZ);
	}

	printMatrix(M);
	printVector(A, (char*)"A = ");
	printVector(IA, (char*)"IA = ");
	printVector(JA, (char*)"JA = ");
}

// Driver code
int main()
{
	matrix M = {
		{ 0, 0, 0, 0, 1 },
		{ 5, 8, 0, 0, 0 },
		{ 0, 0, 3, 0, 0 },
		{ 0, 6, 0, 0, 1 },
	};

	sparesify(M);

	return 0;
}
