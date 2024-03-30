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

__host__ __device__ void sequential_merge_circular(int *A, int m, int *B, int n, int *C, int A_S_start, B_S_start, tile_size) {
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < m) && (j < n)) {
        int i_cir = (A_S_start + i) % tile_size;
        int j_cir = (B_S_start + j) % tile_size;
        if (A[i_cir] <= B[j_cir]) {
            C[k++] = A[i_cir]; i++;
        } else {
            C[k++] = B[j_cir]; j++;
        }
    }
    if (i == m) { // Done with A[], handle remaining B[]
        for (;j < n; j++) {
            int j_cir = (B_S_start + j) % tile_size;
            C[k++] = B[j_cir];
        }
    } else {
        for (;i < m; i++) { // Done with B[], handle remaining A[]
            C[k++] = A[i_cir];
        }
    }
}