__host__ __device__ void sequential_merge(int *A, int m, int *B, int n, int *C) {
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) { // Done with A[], handle remaining B[]
        while (j < n) {
            C[k++] = B[j++];
        }
    } else {
        while (i < m) { // Done with B[], handle remaining A[]
            C[k++] = A[i++];
        }
    }
}