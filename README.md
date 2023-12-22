# PP final
## step 1: generate data  
mode 1 for random size of matrix  
mode 2 for fix size based on input parameters  
ps: percent_for_nonzero range for 1~100  
```
g++ generate_data.cpp -o generate
./generate <number_of_data> <matrix_size> <percent_for_nonzero> <mode>
```
## step 2: generate CSR format for parallelization
```
g++ CSR.cpp -o csr
./csr < matrix_data.txt
```
## step 3: make 
1. cd to non_CSR/ CSR_version to make  
2. run the test file using .sh  
- for non_CSR  
```
sh run_serial.sh
```
- for CSR_version  
```
sh run_CSR.sh
```
