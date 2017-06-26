__global__ void sigmoidf(float* X, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    X[x] = 1 / (1 + expf(-X[x]));
}

__global__ void sigmoid(double* X, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;  

    X[x] = 1 / (1 + exp(-X[x]));
}

__global__ void myTanhf(float* X, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    X[x] = tanhf(X[x]);
}

__global__ void myTanh(double* X, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    X[x] = tanh(X[x]);
}

__global__ void clipf(float* X, const float min, const float max, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    X[x] = fmaxf(fminf(X[x], max), min);
}

__global__ void clip(double* X, const double min, const double max, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    X[x] = fmax(fmin(X[x], max), min);
}

__global__ void inplaceReluDerivativef(float* X, float* Y, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    if(X[x] == 0)
        Y[x] = 0;
}

__global__ void inplaceReluDerivative(double* X, double* Y, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    if(X[x] == 0)
        Y[x] = 0;
}

__global__ void squaredErrorf(float* X, float* Y, float* Z, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    Z[x] = powf((X[x] - Y[x]), 2);
}

__global__ void squaredError(double* X, double* Y, double* Z, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    Z[x] = pow((X[x] - Y[x]), 2);
}

__global__ void logLossf(float* X, float* Y,float* out, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    out[x] = X[x] * logf(Y[x]);
}

__global__ void logLoss(double* X, float* Y,double* out, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    out[x] = X[x] * log(Y[x]);
}

__global__ void binaryLogLossf(float* X, float* Y,float* out, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    out[x] = X[x] * logf(Y[x]) + (1 - X[x]) * logf(1 - Y[x]);
}

__global__ void binaryLogLoss(double* X, float* Y,double* out, int size) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= size) return;

    out[x] = X[x] * log(Y[x]) + (1 - X[x]) * log(1 - Y[x]);
}
