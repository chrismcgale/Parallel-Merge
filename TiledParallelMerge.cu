// Deficiency: Loads tile_size * 2 items per iteration but only uses tile_size
__global__ void merge_tiled_kernel(int *A, int m, int *B, int n, int *C, int tile_size) {
    // Part 1 Find A and B start and end co_rank for the current output subarray
    extern __shared__ int shareAB[]; // Size is 2 * tile_size
    int * A_S = &shareAB[0];
    int * B_S = &shareAB[tile_size];
    int C_start = blockIdx.x * ceil((m + n) / gridDim.x); // start of output subarray
    int C_end = min((blockIdx.x + 1) * ceil((m + n) / gridDim.x), m + n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n); // Make tile co-rank visible to all threads without running for all
        A_S[1] = co_rank(C_next, A, m, B, n);
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

    while (counter < total_iterations) {
        // Part 2 load tile-size A and B elements into shared Memory
        for (int i = 0; i < tile_size; i+=blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                A_S[i + threadIdx.x] = A[A_start + A_consumed + i + threadIdx.x];
            }
        }

        for (int i = 0; i < tile_size; i+=blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed) {
                B_S[i + threadIdx.x] = B[B_start + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        // Part 3 Merge subarrays in parallel
        int C_curr = min(threadIdx.x * (tile_size/blockDim.x), C_length - C_completed);
        int C_next = min((threadIdx.x + 1) * (tile_size/blockDim.x), C_length - C_completed);

        int A_curr = co_rank(C_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        int A_next = co_rank(C_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));

        int B_curr = C_curr - A_curr;
        int B_next = C_next - A_next;

        // All threads call this
        sequential_merge(A_S+a_curr, a_next-a_curr, B_S+B_curr, B_next-B_curr, C+C_start+C_completed+C_curr);

        // Update the number of elements consumed so far
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed = C_completed - A_consumed;
        counter++;
        __syncthreads(); 
    }
}