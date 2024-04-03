__host__ merge_sort(int *A, int m, int *C, int tile_size) {
    cudaMalloc((void**)&A, sizeof(A) / sizeof(A[0]));
    cudaMalloc((void**)&C, sizeof(C) / sizeof(C[0]));

    // sort each block
    sort_tile<<<(n / tile_size), tile_size>>>(A, tile_size);

    for (int i = tile_size; i < n / 2; i *= 2) {
        merge_sort_circular_kernel<<<(n / (2 * i)), 2 * i>>>(A, n, C, i);
    }
    cudaFree(A);
    cudaFree(C);
}

// Sort each tile once then merge later
__global__ sort_tile(int* A, tile_size) {
    std::sort(A, A + tile_size);
}


// Deficiency: Loads tile_size * 2 items per iteration but only uses tile_size
__global__ void merge_sort_circular_kernel(int *A, int m, int *C, int tile_size) {
    unsigned int A_block = blockIdx.x*blockDim.x; unsigned int B_block = (blockIdx.x)*blockDim.x + tile_size;

    // Part 1 Find A and B start and end co_rank for the current output subarray
    extern __shared__ int shareAB[]; // Size is 2 * tile_size
    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];
    int C_start = blockIdx.x * ceil((m + n) / gridDim.x); // start of output subarray
    int C_end = min((blockIdx.x + 1) * ceil((m + n) / gridDim.x), m + n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_start, A[A_block], stride, A[B_block], n); // Make tile co-rank visible to all threads without running for all
        A_S[1] = co_rank(C_end, A[A_block], stride, A[B_block], n);
    }

    __syncthreads();
    int A_start = A_S[0];
    int A_end = A_S[1];
    int B_start = C_start - A_start;
    int B_end = C_end - A_end;
    __syncthreads();

    int counter = 0;
    int C_length = C_start - C_end, A_length = A_end - A_start, B_length = B_end - B_start;
    // Threads are coarsened such that ceil(C_length / tile_size) iterations are needed
    int total_iterations = ceil(C_length / tile_size);
    int C_completed = 0, A_consumed = 0, B_consumed = 0;
    int A_S_start = 0, A_S_consumed = tile_size, B_S_start = 0, B_S_consumed = tile_size;

    while (counter < total_iterations) {
        // Part 2 load tile-size A and B elements into shared Memory

        // Load A_S_consumed elements back into A_S
        for (int i = 0; i < A_S_consumed; i+=blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed && (i + threadIdx.x) < A_S_consumed) {
                A_S[(A_S_start + (tile_size - A_S_consumed) + i + threadIdx.x) % tile_size] = A[A_start + A_consumed + i + threadIdx.x];
            }
        }

        for (int i = 0; i < B_S_consumed; i+=blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed && (i + threadIdx.x) < B_S_consumed) {
                B_S[(B_S_start + (tile_size - B_S_consumed) + i + threadIdx.x) % tile_size] = B[B_start + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        // Part 3 Merge subarrays in parallel
        int C_curr = min(threadIdx.x * (tile_size/blockDim.x), C_length - C_completed);
        int C_next = min((threadIdx.x + 1) * (tile_size/blockDim.x), C_length - C_completed);

        int A_curr = co_rank_circular(C_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed), A_S_start, B_S_start, tile_size);
        int A_next = co_rank_circular(C_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed), A_S_start, B_S_start, tile_size);

        int B_curr = C_curr - A_curr;
        int B_next = C_next - A_next;

        // All threads call this
        sequential_merge_circular(A_S, a_next-a_curr, B_S, B_next-B_curr, C+C_start+C_completed+C_curr, A_S_start + A_curr, B_S_start + B_curr, tile_size);

        // Update the number of elements consumed so far
        A_S_consumed += co_rank_circular(min(tile_size, C_length - C_completed), A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed), A_S_start, B_S_start, tile_size);
        B_S_consumed = min(tile_size, C_length - C_completed) - A_S_consumed;
        C_completed += min(tile_size, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;
        counter++;
        __syncthreads(); 
    }
}