#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <pthread.h>

using namespace std;

struct ThreadData {
    int* result_mat;
    const int* A;
    const int* IA;
    const int* JA;
    const int* b_mat;
    int start;
    int end;
};

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int *A_size, int *IA_size, int *JA_size,
                        int **A, int **IA, int **JA, int **b_mat_ptr){
    cin >> *n_ptr >> *m_ptr >> *l_ptr;

    *A = nullptr;
    *IA = nullptr;
    *JA = nullptr;
    vector<int> vec;
    *b_mat_ptr = new int[*m_ptr * *l_ptr];
    string keyword;
    int count=0;
    cin >> keyword >> keyword >> keyword >> keyword;
    while(keyword!="]"){
        vec.push_back(stoi(keyword));
        cin >> keyword;
    }
    *A = new int[vec.size()];
    *A_size=vec.size();
    for(int i=0; i<vec.size(); i++){
        (*A)[i]=vec[i];
    }

    vec.clear();
    cin >> keyword >> keyword >> keyword >> keyword;
    while(keyword!="]"){
        vec.push_back(stoi(keyword));
        cin >> keyword;
    }
    *IA = new int[vec.size()];
    *IA_size=vec.size();
    for(int i=0; i<vec.size(); i++){
        (*IA)[i]=vec[i];
    }

    vec.clear();
    cin >> keyword >> keyword >> keyword >> keyword;
    while(keyword!="]"){
        vec.push_back(stoi(keyword));
        cin >> keyword;
    }
    *JA = new int[vec.size()];
    *JA_size=vec.size();
    for(int i=0; i<vec.size(); i++){
        (*JA)[i]=vec[i];
    }
    for (int i = 0; i < *m_ptr; ++i) {
        for (int j = 0; j < *l_ptr; ++j) {
            cin >> (*b_mat_ptr)[i * *l_ptr + j];
        }
    }
}

void* matrix_multiply_parallel(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    for (int i = data->start; i < data->end; i++){
        for(int j=data->IA[i]; j<data->IA[i+1]; j++){
            data->result_mat[i] += data->A[j] * data->b_mat[data->JA[j]];
        }
    }

    pthread_exit(NULL);
}

void matrix_multiply(const int n, const int m, const int l, const int A_size, const int IA_size, const int JA_size,
                     const int *A, const int *IA, const int *JA, const int *b_mat){
    int *result_mat = new int[n * l];
    int num_threads=4;
    int chunk_size = (IA_size-1) / num_threads;
    for (int i = 0; i < n; ++i) {
        // for (int j = 0; j < l; ++j) {
        //     result_mat[i * l + j] = 0;
        // }
        result_mat[i] = 0;
    }
    pthread_t threads[num_threads];
    vector<ThreadData> threadDataArray(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threadDataArray[i].result_mat = result_mat;
        threadDataArray[i].A = A;
        threadDataArray[i].IA = IA;
        threadDataArray[i].JA = JA;
        threadDataArray[i].b_mat = b_mat;
        threadDataArray[i].start = i * chunk_size;
        threadDataArray[i].end = (i == num_threads - 1) ? IA_size : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, matrix_multiply_parallel, &threadDataArray[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < l; ++j) {
    //         printf("%d ",result_mat[i * l + j]);
    //     }
    //     printf("\n");
    // }
    delete[] result_mat;
}

void destruct_matrices(int *A, int *IA, int *JA, int *b_mat){
    delete[] A;
    delete[] IA;
    delete[] JA;
    delete[] b_mat;
}

int main () {
    ios_base::sync_with_stdio(false);
    cin.tie(0);    
    int num;
    cin >> num;
    for(int i=0; i<num; i++){
        int n, m, l, A_size, IA_size, JA_size;
        int *A, *IA, *JA, *b_mat;
        auto t0 = std::chrono::steady_clock::now();
        construct_matrices(&n, &m, &l, &A_size, &IA_size, &JA_size, &A, &IA, &JA, &b_mat);
        auto t1 = std::chrono::steady_clock::now();
        matrix_multiply(n, m, l, A_size, IA_size, JA_size, A, IA, JA, b_mat);
        auto t2 = std::chrono::steady_clock::now();
        destruct_matrices(A, IA, JA, b_mat);
        auto t3 = std::chrono::steady_clock::now();
        cout << "Read time: "<<chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms. ";
        cout << "Multiply time: "<<chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms. ";
        cout << "Total time: "<<chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count() << " ms." << endl;
    }
    return 0;
}