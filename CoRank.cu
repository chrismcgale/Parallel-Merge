// Uses the ranges i_low - i and j_low - j to perform a binary search to find i and j such that for all l <= i and h <= j A[l] and B[h] will be to the left of C[k] in the merged array
// Invarient: i + j = k
__host__ __device__ int co_rank(int k, int* A, int m, int *B, int n) {
    int i = min(k, m);
    int j = k - i;
    int i_low = max(0, k - n);
    int j_low = max(0, k - m);
    int delta;
    for (;;) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ceil((i - i_low) / 2.0); // ((i - i_low + 1) >> 1)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = ceil((j - j_low) / 2.0); // ((j - j_low + 1) >> 1)
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            break;
        }
    }
    return i;
}
