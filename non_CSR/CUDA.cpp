#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>
#include "kernel.cu"
using namespace std;


void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr) {
    std::cin >> *n_ptr >> *m_ptr >> *l_ptr;

    *a_mat_ptr = new int[*n_ptr * *m_ptr];
    *b_mat_ptr = new int[*m_ptr * *l_ptr];

    for (int i = 0; i < *n_ptr; ++i) {
        for (int j = 0; j < *m_ptr; ++j) {
            std::cin >> (*a_mat_ptr)[i * *m_ptr + j];
        }
    }

    for (int i = 0; i < *m_ptr; ++i) {
        for (int j = 0; j < *l_ptr; ++j) {
            std::cin >> (*b_mat_ptr)[i * *l_ptr + j];
        }
    }
}

void matrix_multiply_parallel(const int *a_mat, const int *b_mat, int *result_mat, int N) {

}

void destruct_matrices(int *a_mat, int *b_mat) {
    delete[] a_mat;
    delete[] b_mat;
}


int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0);    
    int num;
    cin >> num;
    for(int i=0; i<num; i++){
        int n, m, l;
        int *a_mat, *b_mat;
        construct_matrices(&n, &m, &l, &a_mat, &b_mat);
        int *result_mat = new int[n * l];
        int N = n;
        auto t1 = std::chrono::steady_clock::now();
        matrix_multiply_parallel(a_mat, b_mat, result_mat, N);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::steady_clock::now();
        // for (int i = 0; i < n; ++i) {
        //     for (int j = 0; j < l; ++j) {
        //         printf("%d ", result_mat[i * l + j]);
        //     }
        //     printf("\n");
        // }
        destruct_matrices(a_mat, b_mat);
        delete[] result_mat;
        cout << chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl;
    }
    return 0;
}
