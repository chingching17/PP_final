#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr){
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

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat){
    int *result_mat = new int[n * l];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
            result_mat[i * l + j] = 0;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < m; ++k) {
            for (int j = 0; j < l; ++j) {
                result_mat[i * l + j] += a_mat[i * m + k] * b_mat[k * l + j];
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
            printf("%d ",result_mat[i * l + j]);
        }
        printf("\n");
    }
    delete[] result_mat;
}

void destruct_matrices(int *a_mat, int *b_mat){
    delete[] a_mat;
    delete[] b_mat;
}

int main () {
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    
    int n, m, l;
    int *a_mat, *b_mat;
    construct_matrices(&n, &m, &l, &a_mat, &b_mat);
    auto t1 = std::chrono::steady_clock::now();
    matrix_multiply(n, m, l, a_mat, b_mat);
    auto t2 = std::chrono::steady_clock::now();
    destruct_matrices(a_mat, b_mat);
    
    cout << chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl;
    return 0;
}