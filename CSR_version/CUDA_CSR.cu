#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <chrono>


using namespace std;

#define BLOCK_SIZE 16
// row_ptr = IA
// col_ind = JA
// values = A
// y = result(c)
// x = b_mat
// num_rows = # of rows = m?
__global__ void gpu_matrix_mult(int m, int n, int k, const int A_size, const int IA_size, const int JA_size, const int *A, const int *IA, const int *JA, const int *b_mat, int *c)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int row_start = IA[row];
        int row_end = IA[row + 1];
        float sum = 0.0;

        for (int i = row_start; i < row_end; i++) {
            int a_col = JA[i];
            float a_val = A[i];
            float b_val = b_mat[a_col * k + col];
            sum += a_val * b_val;
        }

        c[row * k + col] = sum;
    }
} 

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

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

void matrix_multiply_cpu(const int n, const int m, const int l, const int A_size, const int IA_size, const int JA_size,
                     const int *A, const int *IA, const int *JA, const int *b_mat, int *d_cc){
    int *result_mat = new int[n * l];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
            result_mat[i * l + j] = 0;
        }
    }

    for(int i=1; i<IA_size; i++){
        for(int j=IA[i-1]; j<IA[i]; j++){
            result_mat[i-1]+=A[j]*b_mat[JA[j]];
        }
    }

    
    cout << "cpu_ver" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
            printf("%d ",result_mat[i * l + j]);
        }
        printf("\n");
    }
    cout << "okay" << endl;
    
    delete[] result_mat;
}

int main(int argc, char const *argv[])
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    int num;
    cin >> num;
    for(int i = 0; i < num; i++){
        float gpu_total_time;
        cudaEvent_t start_total, stop_total;
        cudaEventCreate(&start_total);
        cudaEventCreate(&stop_total);
        cudaEventRecord(start_total,0);

        float read_time;
        cudaEvent_t start_read, stop_read;
        cudaEventCreate(&start_read);
        cudaEventCreate(&stop_read);


        srand(3333);
        int m, n, k, A_size, IA_size, JA_size;
        int *A, *IA, *JA, *b_mat;

        cudaEventRecord(start_read,0);
        construct_matrices(&m, &n, &k, &A_size, &IA_size, &JA_size, &A, &IA, &JA, &b_mat);
        cudaEventRecord(stop_read, 0);
        cudaEventSynchronize(stop_read);

        // cout << IA_size << ' ' << JA_size << ' ' << A_size << endl;
        int *h_b, *h_c, *h_cc, *h_A, *h_JA, *h_IA;
        cudaMalloc((void **) &h_b, sizeof(int)*n*k);
        cudaMalloc((void **) &h_c, sizeof(int)*m*k);
        cudaMalloc((void **) &h_cc, sizeof(int)*m*k);
        cudaMalloc((void **) &h_A, sizeof(int)* A_size);
        cudaMalloc((void **) &h_IA, sizeof(int)* IA_size);
        cudaMalloc((void **) &h_JA, sizeof(int)* JA_size);

        cudaMemcpy(h_b, b_mat, sizeof(int)*n*k, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_A, A, sizeof(int)*A_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_IA, IA, sizeof(int)*IA_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_JA, JA, sizeof(int)*JA_size, cudaMemcpyDeviceToHost);
        
        float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

        // some events to count the execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        if(m == n && n == k)
        {
            // gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);    
        }
        else
        {
            // start to count execution time of GPU version
            cudaEventRecord(start, 0);
            gpu_matrix_mult<<<dimGrid, dimBlock>>>(m, n, k,  A_size, IA_size, JA_size, h_A, h_IA, h_JA, h_b, h_c); 
            cudaEventRecord(stop, 0);
        }
        // Transefr results from device to host 
        cudaMemcpy(h_cc, h_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        // time counting terminate
        cudaEventSynchronize(stop);
        cudaEventSynchronize(stop_total);
        // cout << "gpu_ver" << endl;
        // cout << m << ' ' << k << ' ' << sizeof(h_cc) << ' ' << sizeof(h_c) << endl;

        cudaEventRecord(stop_total, 0);

        cudaEventElapsedTime(&read_time, start_read, stop_read);
        printf("Read time: %f ms.",  read_time);

        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        printf("Mul time: %f ms. ", gpu_elapsed_time_ms);

        cudaEventElapsedTime(&gpu_total_time, start_total, stop_total);
        printf("Total time: %f ms.\n", gpu_total_time);

        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);


        /*
        // cpu version
        // start the CPU version
        int all_ok = 1;
        cudaEventRecord(start, 0);

        matrix_multiply_cpu(m, n, k, A_size, IA_size, JA_size, A, IA, JA, b_mat, h_cc);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n", m, n, n, k, cpu_elapsed_time_ms);

        // validate results computed by GPU
        // for (int i = 0; i < m; ++i)
        // {
        //     for (int j = 0; j < k; ++j)
        //     {
        //         //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, d_cc[i*k + j], i, j, d_c[i*k + j]);
        //         if(d_cc[i*k + j] != h_cc[i*k + j])
        //         {
        //             all_ok = 0;
        //         }
        //     }
        //     //printf("\n");
        // }

        // roughly compute speedup
        // if(all_ok)
        // {
        //     printf("all results are correct!!!, speedup = %f\n\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
        // }
        // else
        // {
        //     printf("incorrect results\n\n");
        // }*/

        // free memory
        // cudaFree(d_a);
        cudaFree(h_b);
        cudaFree(h_c);
        cudaFree(h_cc);
        cudaFree(h_A);
        cudaFree(h_IA);
        cudaFree(h_JA);
    }
    return 0;
}