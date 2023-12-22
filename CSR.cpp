// CPP program to find sparse matrix rep-
// resentation using CSR
#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

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

auto s0 = std::chrono::steady_clock::now();
auto s1 = std::chrono::steady_clock::now();

// Generate the three vectors A, IA, JA 
void sparesify(const matrix& M, ofstream &output)
{
	
	int m = M.size();
	int n = M[0].size(), i, j;
	// vi A;
	// vi IA = { 0 }; // IA matrix has N+1 rows
	// vi JA;
	int *A=new int[m*n];
	int *IA=new int[m+1];
	int *JA=new int[m*n];
	int NNZ = 0;
	int indexA=0, indexIA=0, indexJA=0;

	s0 = std::chrono::steady_clock::now();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (M[i][j] != 0) {
				// A.push_back(M[i][j]);
				// JA.push_back(j);
				A[indexA]=M[i][j];
				++indexA;
				JA[indexJA]=j;
				++indexJA;
				// Count Number of Non Zero 
				// Elements in row i
				NNZ++;
			}
		}
		//IA.push_back(NNZ);
		IA[indexIA]=NNZ;
		++indexIA;
	}
	s1 = std::chrono::steady_clock::now();

	// printMatrix(M);
	// printVector(A, (char*)"A = ");
	// printVector(IA, (char*)"IA = ");
	// printVector(JA, (char*)"JA = ");
	

	output << "A = [ ";
	for(int i=0; i<=indexA; i++)
		output << A[i] << " ";
	output << "]" << endl;

	output << "IA = [ ";
	for(int i=0; i<=indexIA; i++)
		output << IA[i] << " ";
	output << "]" << endl;

	output << "JA = [ ";
	for(int i=0; i<=indexJA; i++)
		output << JA[i] << " ";
	output << "]" << endl;

	// output << "A = [ ";
	// for(auto n:A)
	// 	output << n << " ";
	// output << "]" << endl;

	// output << "IA = [ ";
	// for(auto n:IA)
	// 	output << n << " ";
	// output << "]" << endl;

	// output << "JA = [ ";
	// for(auto n:JA)
	// 	output << n << " ";
	// output << "]" << endl;
}

// Driver code
int main()
{
	matrix M;
	ofstream output;
	output.open("matrix_csr.txt");
	int num,n,m,l,tmp;
	cin >> num;
	output << num << endl;
	for(int k=0; k<num; k++){
		cin >> n >> m >> l;
		output << n << " " << m << " " << l << endl;
		M.clear();
		M.resize(n);
		auto t0 = std::chrono::steady_clock::now();
		for(int i=0; i<n; i++){
			for(int j=0; j<m; j++){
				cin >> tmp;
				M[i].push_back(tmp);
			}
		}
		auto t1 = std::chrono::steady_clock::now();

		sparesify(M,output);
		
		for(int i=0; i<m; i++){
			cin >> tmp;
			output << tmp << endl;
		}
		output << endl;
		auto t2 = std::chrono::steady_clock::now();
        cout << "Read time: "<<chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms. ";
        cout << "Sparesify time: "<<chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count() << " ms. ";
        cout << "Total time: "<<chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count() << " ms." << endl;
	}
	output.close();
	return 0;
}
