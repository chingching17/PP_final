#include <iostream>
#include <fstream>
#include <string>
#include <pthread.h>
#include <chrono>

using namespace std;

struct ThreadData {
    int start;
    int end;
    const int *a_mat;
    const int *b_mat;
    int *result_mat;
    int m;
    int l;
};

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

void *matrix_multiply_thread(void *arg) {
    ThreadData *data = reinterpret_cast<ThreadData*>(arg);

    // for (int i = data->start; i < data->end; ++i) {
    //     for (int j = 0; j < data->l; ++j) {
    //         data->result_mat[i * data->l + j] = 0;
    //         for (int k = 0; k < data->m; ++k) {
    //             data->result_mat[i * data->l + j] += data->a_mat[i * data->m + k] * data->b_mat[k * data->l + j];
    //         }
    //     }
    // }
    for (int i = data->start; i < data->end; ++i) {
        data->result_mat[i] = 0;
        for (int k = 0; k < data->m; ++k) {
            data->result_mat[i] += data->a_mat[i * data->m + k] * data->b_mat[k];
        }
    }
    pthread_exit(NULL);
}

void matrix_multiply_parallel(const int n, const int m, const int l,
                               const int *a_mat, const int *b_mat, int *result_mat, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    int chunk_size = n / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
        thread_data[i].a_mat = a_mat;
        thread_data[i].b_mat = b_mat;
        thread_data[i].result_mat = result_mat;
        thread_data[i].m = m;
        thread_data[i].l = l;

        pthread_create(&threads[i], NULL, matrix_multiply_thread, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
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
        auto t0 = std::chrono::steady_clock::now();
        construct_matrices(&n, &m, &l, &a_mat, &b_mat);
        int *result_mat = new int[n * l];
        int num_threads = 4;
        auto t1 = std::chrono::steady_clock::now();
        matrix_multiply_parallel(n, m, l, a_mat, b_mat, result_mat, num_threads);
        auto t2 = std::chrono::steady_clock::now();
        // for (int i = 0; i < n; ++i) {
        //     for (int j = 0; j < l; ++j) {
        //         printf("%d ", result_mat[i * l + j]);
        //     }
        //     printf("\n");
        // }

        destruct_matrices(a_mat, b_mat);
        delete[] result_mat;
        auto t3 = std::chrono::steady_clock::now();
        cout << "Read time: "<<chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms. ";
        cout << "Multiply time: "<<chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms. ";
        cout << "Total time: "<<chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count() << " ms." << endl;
    }
    return 0;
}
