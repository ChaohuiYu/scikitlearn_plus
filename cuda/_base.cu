__global__ void clipf(float* X, const float min, const float max) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    X[x] = fmaxf(fminf(X[x], max), min);
}

__global__ void clip(double* X, const double min, const double max) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    X[x] = fmax(fmin(X[x], max), min);
}