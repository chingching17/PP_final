#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
#include <cstring>
#include <chrono>

using namespace std;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank!=0)
        return;
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
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;
    int n_len, m_len, l_len;
    if(world_rank==0){
        n_len=n;
        m_len=m;
        l_len=l;
    }
    MPI_Bcast(&n_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int block=n_len/world_size;
    int a_size = block*m_len; 
    int size=block*l_len;
    int *local_a=new int[a_size];
    int *local_b=new int[m_len*l_len];
    int *local_result = new int[size];
    int *result_mat = nullptr;
    for(int i=0; i<size; ++i)
        local_result[i]=0;    

    if(world_rank == 0){
        for(int i=1; i<world_size; ++i){
            MPI_Send(a_mat+i*a_size, a_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        result_mat = new int[n * l];
        std::memcpy(local_a, a_mat, sizeof(int) * a_size);
        std::memcpy(local_b, b_mat, sizeof(int) * m_len*l_len);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < l; ++j) {
                result_mat[i * l + j] = 0;
            }
        }
    }
    else{
        MPI_Recv(local_a, a_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Bcast(local_b, m_len*l_len, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i=0; i<block; ++i){
        for (int k = 0; k < m_len; ++k) {
            for (int j = 0; j < l_len; ++j) {
                local_result[i * l_len + j] += local_a[i * m_len + k] * local_b[k * l_len + j];
            }
        }        
    }

    MPI_Gather(local_result, block*l_len, MPI_INT, result_mat, block*l_len, MPI_INT, 0, MPI_COMM_WORLD);

    if(world_rank==0){
        if(n%world_size){
            int begin=world_size*block;
            for (int i = begin; i < n; ++i) {
                for (int k = 0; k < m; ++k) {
                    for (int j = 0; j < l; ++j) {
                        result_mat[i * l + j] += a_mat[i * m + k] * b_mat[k * l + j];
                    }
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
    delete[] local_a;
    delete[] local_b;
    delete[] local_result;
    if (world_rank == 0)
        delete[] result_mat;
}

void destruct_matrices(int *a_mat, int *b_mat){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank!=0)
        return;
    delete[] a_mat;
    delete[] b_mat;
}


int main () {
    ios_base::sync_with_stdio(false);
    cin.tie(0);    
    int num;
    cin >> num;
    MPI_Init(NULL, NULL);
    for(int i=0; i<num; i++){
        int n, m, l;
        int *a_mat, *b_mat;
        //double start_time = MPI_Wtime();
        auto t0 = std::chrono::steady_clock::now();
        construct_matrices(&n, &m, &l, &a_mat, &b_mat);
        auto t1 = std::chrono::steady_clock::now();
        matrix_multiply(n, m, l, a_mat, b_mat);
        auto t2 = std::chrono::steady_clock::now();
        destruct_matrices(a_mat, b_mat);
        auto t3 = std::chrono::steady_clock::now();
        //double end_time = MPI_Wtime();
        //printf("MPI running time: %lf Seconds\n", end_time - start_time);
        cout << "Read time: "<<chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms. ";
        cout << "Multiply time: "<<chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms. ";
        cout << "Total time: "<<chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count() << " ms." << endl;
    }
    MPI_Finalize();
    return 0;
}
