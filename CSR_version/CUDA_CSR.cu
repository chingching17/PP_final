/*
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>

using namespace std;

#define BLOCK_SIZE 16

/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
// __global__ void spmv_csr(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y)
// row_ptr = IA
// col_ind = JA
// values = A
// y = result(c)
// x = b_mat
// num_rows = # of rows = m?
__global__ void gpu_matrix_mult(int m, int n, int k, const int A_size, const int IA_size, const int JA_size, const int *A, const int *IA, const int *JA, const int *b_mat, int *c)
{ 
    // int row = blockIdx.y * blockDim.y + threadIdx.y; 
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int sum = 0;
    // if( col < k && row < m) 
    // {
    //     // for(int i = 0; i < n; i++) 
    //     // {
    //     //     sum += a[row * n + i] * b[i * k + col];
    //     // }
    //     // c[row * k + col] = sum;
        
    //     // for(int i=1; i<IA_size; i++){
    //     //     for(int j=IA[i-1]; j<IA[i]; j++){
    //     //         c[i-1]+=A[j]*b_mat[JA[j]];
    //     //     }
    //     // }
    // }
    int num_rows = m;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x * gridDim.x) {
        float dotProduct = 0;
        const int row_start = IA[i];
        const int row_end = IA[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            dotProduct += A[j] * b_mat[JA[j]];
        }
        
        c[i] = dotProduct;
    }
} 

/*
*********************************************************************
function name: gpu_square_matrix_mult

description: dot product of two matrix (not only square) in GPU

parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:

        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU, 
             for validating GPU results

parameters: 
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C) 
            to store the result
return: none
*********************************************************************
*/
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

/*
*********************************************************************
function name: main

description: test and compare

parameters: 
            none

return: none
*********************************************************************
*/
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

int main(int argc, char const *argv[])
{
    int num;
    cin >> num;
    for(int i = 0; i < num; i++){
        /* Fixed seed for illustration */
        srand(3333);
        int m, n, k, A_size, IA_size, JA_size;
        int *A, *IA, *JA, *b_mat;
        construct_matrices(&m, &n, &k, &A_size, &IA_size, &JA_size, &A, &IA, &JA, &b_mat);
        cout << "construct ok" << endl;

        // allocate memory in host RAM, h_cc is used to store CPU result
        int *h_a, *h_b, *h_c, *h_cc;
        cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
        cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
        cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
        cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

        float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

        // some events to count the execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // start to count execution time of GPU version
        cudaEventRecord(start, 0);
        // Allocate memory space on the device 
        int *d_a, *d_b, *d_c;
        cudaMalloc((void **) &d_a, sizeof(int)*m*n);
        cudaMalloc((void **) &d_b, sizeof(int)*n*k);
        cudaMalloc((void **) &d_c, sizeof(int)*m*k);

        // copy matrix A and B from host to device memory
        cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

        unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        if(m == n && n == k)
        {
            gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);    
        }
        else
        {
            gpu_matrix_mult<<<dimGrid, dimBlock>>>(m, n, k,  A_size, IA_size, JA_size, A, IA, JA, d_b, d_c);  
            cout << "mul ok" << endl;  
        }
        // Transefr results from device to host 
        cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        // time counting terminate
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapse on GPU computing
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n", m, n, n, k, gpu_elapsed_time_ms);

        // start the CPU version
        cudaEventRecord(start, 0);

        cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n", m, n, n, k, cpu_elapsed_time_ms);

        // validate results computed by GPU
        int all_ok = 1;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
                if(h_cc[i*k + j] != h_c[i*k + j])
                {
                    all_ok = 0;
                }
            }
            //printf("\n");
        }

        // roughly compute speedup
        if(all_ok)
        {
            printf("all results are correct!!!, speedup = %f\n\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
        }
        else
        {
            printf("incorrect results\n\n");
        }

        // free memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        cudaFreeHost(h_cc);
    }
    return 0;
}