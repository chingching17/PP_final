nonCSR:
	g++ -o serial serial.cpp; g++ -o Pthread Pthread.cpp -lpthread; g++ -fopenmp -o OpenMP OpenMP.cpp; mpicxx -o MPI MPI.cpp; nvcc -o cuda CUDA.cu;

CSR:
	g++ -o serial_CSR serial_CSR.cpp; g++ -o Pthread_CSR Pthread_CSR.cpp -lpthread; g++ -fopenmp -o OpenMP_CSR OpenMP_CSR.cpp; mpicxx -o MPI_CSR MPI_CSR.cpp;

data:
	g++ -o data generate_data.cpp; g++ -o csr CSR.cpp;

clean:
	rm -f matmul serial Pthread OpenMP MPI serial_CSR Pthread_CSR OpenMP_CSR MPI_CSR data csr