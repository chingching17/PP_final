#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <mpi.h>
#include <cstring>

using namespace std;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int *A_size, int *IA_size, int *JA_size,
                        int **A, int **IA, int **JA, int **b_mat_ptr){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank!=0)
        return;                            
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

void matrix_multiply(int n, int m, int l, int A_size, int IA_size, int JA_size,
                     const int *A, const int *IA, const int *JA, const int *b_mat){
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int block=IA_size/world_size;
    int *local_A=new int[A_size];
    int *local_IA=new int[IA_size];
    int *local_JA=new int[JA_size];
    int *local_result=new int[block];
    int *result_mat=nullptr;
    if(world_rank==0){
        std::memcpy(local_A, A, sizeof(int) * A_size);
        std::memcpy(local_IA, IA, sizeof(int) * IA_size); 
        std::memcpy(local_JA, JA, sizeof(int) * JA_size);
        result_mat = new int[n * l];     
        for (int i = 0; i < n; ++i) {
            // for (int j = 0; j < l; ++j){
            //     result_mat[i * l + j] = 0;
            // }
            result_mat[i] = 0;
        }
    }
    MPI_Bcast(local_A, A_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_IA, IA_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(local_JA, JA_size, MPI_INT, 0, MPI_COMM_WORLD);   

    for(int i=world_rank*block; i<(world_rank + 1) * block; ++i){
        local_result[i-world_rank*block]=0;
        for(int j=local_IA[i]; j<local_IA[i+1]; j++){
            local_result[i-world_rank*block]+=local_A[j]*b_mat[JA[j]];
        }
    }

    MPI_Gather(local_result, block, MPI_INT, result_mat, block, MPI_INT, 0, MPI_COMM_WORLD);

    if(world_rank==0){
        if(IA_size%world_size){
            int begin=world_size*block;
            for (int i = begin; i < IA_size; ++i) {
                for(int j=IA[i]; j<IA[i+1]; j++){
                    result_mat[i]+=A[j]*b_mat[JA[j]];
                }
            }            
        }
    }

    // if(world_rank==0){
    //     for (int i = 0; i < n; ++i) {
    //         for (int j = 0; j < l; ++j) {
    //             printf("%d ",result_mat[i * l + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    delete[] local_A;
    delete[] local_IA;
    delete[] local_JA;
    delete[] local_result;
    if (world_rank == 0)
        delete[] result_mat;
}

void destruct_matrices(int *A, int *IA, int *JA, int *b_mat){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank!=0)
        return;
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
    MPI_Init(NULL, NULL);
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
    MPI_Finalize();
    return 0;
}