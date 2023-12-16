# PP final
## step 1: generate data  
mode 1 for random size of matrix  
mode 2 for fix size based on input parameters  
```
g++ generate_data.cpp -o generate
./generate <number_of_data> <matrix_size> <percent_for_nonzero> <mode>
```
## step 2: generate CSR format for parallelization
```
g++ CSR.cpp -o csr
./csr < matrix_data.txt
```
