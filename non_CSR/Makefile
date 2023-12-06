make:
	g++ -o serial serial.cpp; g++ -o Pthread Pthread.cpp -lpthread; g++ -fopenmp -o OpenMP OpenMP.cpp; mpicxx -o MPI MPI.cpp; g++ -o serial_CSR serial_CSR.cpp;

clean:
	rm -f matmul serial Pthread OpenMP MPI serial_CSR