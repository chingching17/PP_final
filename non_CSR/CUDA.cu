#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
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

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr) {

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

int main(int argc, char const *argv[])
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    int num;
    cin >> num;
    for(int i = 0; i < num; i++){
        int m, n, k;
        srand(3333);
        cin >> m >> n >> k;
        cout << m << n << k;

        float gpu_total_time;
        cudaEvent_t start_total, stop_total;
        cudaEventCreate(&start_total);
        cudaEventCreate(&stop_total);
        cudaEventRecord(start_total,0);

        float read_time;
        cudaEvent_t start_read, stop_read;
        cudaEventCreate(&start_read);
        cudaEventCreate(&stop_read);

        

        int *h_a, *h_b, *h_c, *h_cc;
        cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
        cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
        cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
        cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

        cudaEventRecord(start_read,0);
        construct_matrices(&m, &n, &k, &h_a, &h_b);
        cudaEventRecord(stop_read, 0);
        cudaEventSynchronize(stop_read);


        float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        
        int *d_a, *d_b, *d_c;
        cudaMalloc((void **) &d_a, sizeof(int)*m*n);
        cudaMalloc((void **) &d_b, sizeof(int)*n*k);
        cudaMalloc((void **) &d_c, sizeof(int)*m*k);

        cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

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
            cudaEventRecord(start, 0);
            gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);   
            cudaEventRecord(stop, 0);

        }

        cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        cudaEventSynchronize(stop);
        cudaEventSynchronize(stop_total);
        cudaEventRecord(stop_total, 0);


        // compute time elapse on GPU computing
        cudaEventElapsedTime(&read_time, start_read, stop_read);
        printf("Read time: %f ms.",  read_time);

        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        printf("Mul time: %f ms.",  gpu_elapsed_time_ms);

        cudaEventElapsedTime(&gpu_total_time, start_total, stop_total);
        printf("Total time: %f ms.\n",  gpu_total_time);

        cudaEventDestroy(start_read);
        cudaEventDestroy(stop_read);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
/*
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
            printf("incorrect results\n");
        }*/

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