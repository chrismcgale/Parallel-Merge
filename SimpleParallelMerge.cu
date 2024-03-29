__global__ void merge_basic_kernel(int * A, int m, int *B, int n, int *C) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int elementsPerThread = ceil((m + n) / (blockDim.x*gridDim.x));
    int k_curr = tid*elementsPerThread; // start output index
    int k_next = min((tid + 1) * elementsPerThread, m+n); // end output index
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    // (&Array[start], end) -> Another way of sending slices Array[start:end] 
    sequential_merge(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}