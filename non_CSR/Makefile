make:
	g++ -o serial serial.cpp; g++ -o Pthread Pthread.cpp -lpthread; g++ -fopenmp -o OpenMP OpenMP.cpp; mpicxx -o MPI MPI.cpp

clean:
	rm -f matmul serial Pthread OpenMP MPI