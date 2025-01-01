#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <stdexcept>

using namespace std;

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

// -----------------------------------------------------------------------------
// Utility: check for CUDA errors
// -----------------------------------------------------------------------------
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA error at " << msg << ": " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------------
// Parse command-line arguments
// -----------------------------------------------------------------------------
void parse_arguments(int argc, char* argv[], string& train_file, double& learning_rate,
                     int& iterations, double& train_ratio, int& hidden_size) {
    // Default values
    train_file = "data/train.csv";
    learning_rate = 0.1;
    iterations = 40;
    train_ratio = 0.8;
    hidden_size = 10;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--train_file" && i + 1 < argc) {
            train_file = argv[++i];
        } else if (arg == "--learning_rate" && i + 1 < argc) {
            learning_rate = stod(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = stoi(argv[++i]);
        } else if (arg == "--train_ratio" && i + 1 < argc) {
            train_ratio = stod(argv[++i]);
        } else if (arg == "--hidden_size" && i + 1 < argc) {
            hidden_size = stoi(argv[++i]);
        } else {
            throw invalid_argument("Invalid argument: " + arg);
        }
    }
}

// -----------------------------------------------------------------------------
// CPU utilities
// -----------------------------------------------------------------------------
vector<vector<double>> initialize_matrix_cpu(int rows, int cols, double min_val = -0.5, double max_val = 0.5) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);
    
    vector<vector<double>> mat(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = dis(gen);
    return mat;
}

vector<double> initialize_vector_cpu(int size, double min_val = -0.5, double max_val = 0.5) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);

    vector<double> vec(size);
    for (double &v : vec)
        v = dis(gen);
    return vec;
}

vector<vector<double>> load_csv(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> data;
    string line;

    // Skip the header row
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;
        while (getline(ss, value, ',')) {
            try {
                row.push_back(stod(value));
            } catch (const invalid_argument& e) {
                cerr << "Invalid data in CSV: " << value << endl;
                exit(EXIT_FAILURE);
            }
        }
        data.push_back(row);
    }
    return data;
}

// -----------------------------------------------------------------------------
// GPU kernels
// -----------------------------------------------------------------------------

// Forward pass kernel:
//  - Compute Z1 = W1 * X + B1, then A1 = relu(Z1)
//  - Compute Z2 = W2 * A1 + B2, then A2 = softmax(Z2)
// Here we assume the entire batch is processed in parallel.
// Each thread processes exactly one training example (index i).
// -----------------------------------------------------------------------------
//
// Notation for GPU arrays (all in row-major):
//   d_X        : size = train_size * INPUT_SIZE
//   d_W1       : size = hidden_size * INPUT_SIZE
//   d_B1       : size = hidden_size
//   d_Z1       : size = train_size * hidden_size
//   d_A1       : size = train_size * hidden_size
//   d_W2       : size = OUTPUT_SIZE * hidden_size
//   d_B2       : size = OUTPUT_SIZE
//   d_Z2       : size = train_size * OUTPUT_SIZE
//   d_A2       : size = train_size * OUTPUT_SIZE
//
// gridDim.x * blockDim.x >= train_size
// one thread = one sample (index i)

__global__ void forwardPassKernel(
    const double* __restrict__ d_X,
    const double* __restrict__ d_W1,
    const double* __restrict__ d_B1,
    double* __restrict__ d_Z1,
    double* __restrict__ d_A1,
    const double* __restrict__ d_W2,
    const double* __restrict__ d_B2,
    double* __restrict__ d_Z2,
    double* __restrict__ d_A2,
    int train_size,
    int hidden_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= train_size) return;  // out of range

    // Pointers to the i-th data row
    const double* x_i = d_X + i * INPUT_SIZE;

    // Compute Z1[i, *] = W1 * x_i + B1
    // dimension: (hidden_size x INPUT_SIZE) * (INPUT_SIZE x 1) = (hidden_size x 1)
    double* z1_i = d_Z1 + i * hidden_size;
    double* a1_i = d_A1 + i * hidden_size;

    for (int h = 0; h < hidden_size; ++h) {
        double sum_val = 0.0;
        // dot product
        for (int k = 0; k < INPUT_SIZE; ++k) {
            sum_val += d_W1[h * INPUT_SIZE + k] * x_i[k];
        }
        sum_val += d_B1[h];
        z1_i[h] = sum_val;
        a1_i[h] = (sum_val > 0.0) ? sum_val : 0.0;  // ReLU
    }

    // Now compute Z2[i, *] = W2 * A1[i, *] + B2
    // dimension: (OUTPUT_SIZE x hidden_size) * (hidden_size x 1) = (OUTPUT_SIZE x 1)
    double* z2_i = d_Z2 + i * OUTPUT_SIZE;
    double* a2_i = d_A2 + i * OUTPUT_SIZE;

    for (int out = 0; out < OUTPUT_SIZE; ++out) {
        double sum_val = 0.0;
        for (int h = 0; h < hidden_size; ++h) {
            sum_val += d_W2[out * hidden_size + h] * a1_i[h];
        }
        sum_val += d_B2[out];
        z2_i[out] = sum_val;
    }

    // Softmax for Z2[i, *]
    // First compute denominator = sum(exp(z2_i))
    double max_val = z2_i[0];
    for (int out = 1; out < OUTPUT_SIZE; ++out) {
        if (z2_i[out] > max_val) max_val = z2_i[out];
    }
    double exp_sum = 0.0;
    for (int out = 0; out < OUTPUT_SIZE; ++out) {
        exp_sum += exp(z2_i[out] - max_val); 
    }
    // Compute final
    for (int out = 0; out < OUTPUT_SIZE; ++out) {
        double e = exp(z2_i[out] - max_val);
        a2_i[out] = e / exp_sum;
    }
}

// Backward pass kernel:
// For each sample i, compute partial derivatives w.r.t. W2, B2, W1, B1.
//
// We store the partial derivative for each sample in d_localdW1, d_localdB1, d_localdW2, d_localdB2.
// Then we will reduce them afterwards in a separate kernel or on the CPU.
//
// Steps:
//   dZ2 = A2 - one_hot(Y[i])
//   dW2 += dZ2 * A1^T
//   dB2 += dZ2
//   dZ1 = (W2^T * dZ2) * relu'(Z1)
//   dW1 += dZ1 * X^T
//   dB1 += dZ1
//
// Each thread = one sample (index i).

__global__ void backwardPassKernel(
    const double* __restrict__ d_X,
    const int*    __restrict__ d_Y,
    const double* __restrict__ d_Z1,
    const double* __restrict__ d_A1,
    const double* __restrict__ d_A2,
    const double* __restrict__ d_W2,
    double* __restrict__ d_localdW1,
    double* __restrict__ d_localdB1,
    double* __restrict__ d_localdW2,
    double* __restrict__ d_localdB2,
    int train_size,
    int hidden_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= train_size) return;

    // One-hot of Y[i]
    int label = d_Y[i];
    // dZ2 = A2 - one_hot
    // store in a temporary array of size OUTPUT_SIZE
    double dZ2[OUTPUT_SIZE];
    const double* a2_i = d_A2 + i * OUTPUT_SIZE;
    for (int out = 0; out < OUTPUT_SIZE; ++out) {
        double oh = (out == label) ? 1.0 : 0.0;
        dZ2[out] = a2_i[out] - oh;
    }

    // dB2 accumulation: each sample contributes dZ2
    // dW2 accumulation: dZ2 * A1[i, *]^T
    const double* a1_i = d_A1 + i * hidden_size;
    for (int out = 0; out < OUTPUT_SIZE; ++out) {
        double val = dZ2[out];
        // B2
        d_localdB2[i * OUTPUT_SIZE + out] = val;
        // W2
        for (int h = 0; h < hidden_size; ++h) {
            d_localdW2[i * OUTPUT_SIZE * hidden_size + out * hidden_size + h] = val * a1_i[h];
        }
    }

    // dZ1 = (W2^T * dZ2) * relu'(Z1[i])
    // Z1[i] in d_Z1 + i*hidden_size
    const double* z1_i = d_Z1 + i * hidden_size;
    double dZ1[1024]; // enough size for hidden layer
    for (int h = 0; h < hidden_size; ++h) {
        // sum_k=0..OUTPUT_SIZE-1 of W2[k,h] * dZ2[k]
        double sum_val = 0.0;
        for (int out = 0; out < OUTPUT_SIZE; ++out) {
            sum_val += d_W2[out * hidden_size + h] * dZ2[out];
        }
        // relu'(z1_i[h])
        double relu_grad = (z1_i[h] > 0.0) ? 1.0 : 0.0;
        dZ1[h] = sum_val * relu_grad;
    }

    // dB1
    for (int h = 0; h < hidden_size; ++h) {
        d_localdB1[i * hidden_size + h] = dZ1[h];
    }

    // dW1 = dZ1 * X[i]^T
    const double* x_i = d_X + i * INPUT_SIZE;
    for (int h = 0; h < hidden_size; ++h) {
        for (int in = 0; in < INPUT_SIZE; ++in) {
            d_localdW1[i * hidden_size * INPUT_SIZE + h * INPUT_SIZE + in] = dZ1[h] * x_i[in];
        }
    }
}

// Reduction kernel for summing partial derivatives of size (train_size * param_shape),
// and then dividing by train_size, for final gradient. Each parameter is updated:
//   W -= learning_rate * (1/train_size)*dW
// We do separate kernels for W1/B1 and W2/B2 to keep code simpler.

// reduce d_localdW2 into d_W2, reduce d_localdB2 into d_B2
__global__ void reduceAndUpdateW2B2(
    double* __restrict__ d_W2,
    double* __restrict__ d_B2,
    const double* __restrict__ d_localdW2,
    const double* __restrict__ d_localdB2,
    double learning_rate,
    int train_size,
    int hidden_size)
{
    // We have OUTPUT_SIZE * hidden_size + OUTPUT_SIZE for B2
    // parallelize across all these elements

    // First update W2 (OUTPUT_SIZE * hidden_size)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_w2 = OUTPUT_SIZE * hidden_size;
    // reduce d_localdW2 for the idx-th weight
    if (idx < total_w2) {
        double sum_val = 0.0;
        for (int i = 0; i < train_size; ++i) {
            sum_val += d_localdW2[i * total_w2 + idx];
        }
        sum_val /= (double)train_size;
        d_W2[idx] -= learning_rate * sum_val;
    }

    // Then update B2 (OUTPUT_SIZE)
    int idxB = idx - total_w2;
    if (idxB >= 0 && idxB < OUTPUT_SIZE) {
        double sum_val = 0.0;
        for (int i = 0; i < train_size; ++i) {
            sum_val += d_localdB2[i * OUTPUT_SIZE + idxB];
        }
        sum_val /= (double)train_size;
        d_B2[idxB] -= learning_rate * sum_val;
    }
}

// reduce d_localdW1 into d_W1, reduce d_localdB1 into d_B1
__global__ void reduceAndUpdateW1B1(
    double* __restrict__ d_W1,
    double* __restrict__ d_B1,
    const double* __restrict__ d_localdW1,
    const double* __restrict__ d_localdB1,
    double learning_rate,
    int train_size,
    int hidden_size)
{
    // W1: hidden_size*INPUT_SIZE
    // B1: hidden_size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_w1 = hidden_size * INPUT_SIZE;

    // W1 update
    if (idx < total_w1) {
        double sum_val = 0.0;
        for (int i = 0; i < train_size; ++i) {
            sum_val += d_localdW1[i * total_w1 + idx];
        }
        sum_val /= (double)train_size;
        d_W1[idx] -= learning_rate * sum_val;
    }

    // B1 update
    int idxB = idx - total_w1;
    if (idxB >= 0 && idxB < hidden_size) {
        double sum_val = 0.0;
        for (int i = 0; i < train_size; ++i) {
            sum_val += d_localdB1[i * hidden_size + idxB];
        }
        sum_val /= (double)train_size;
        d_B1[idxB] -= learning_rate * sum_val;
    }
}

// -----------------------------------------------------------------------------
// CPU helper for accuracy
// -----------------------------------------------------------------------------
vector<int> get_predictions(const vector<double>& A2, int train_size) {
    // A2 is train_size x OUTPUT_SIZE
    vector<int> preds(train_size);
    for (int i = 0; i < train_size; ++i) {
        const double* row = &A2[i * OUTPUT_SIZE];
        preds[i] = int(std::distance(row, max_element(row, row + OUTPUT_SIZE)));
    }
    return preds;
}

double get_accuracy(const vector<int>& predictions, const vector<int>& labels) {
    int correct = 0;
    for (int i = 0; i < (int)predictions.size(); ++i) {
        if (predictions[i] == labels[i]) correct++;
    }
    return double(correct) / predictions.size();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Variables for input parameters
    string train_file;
    double learning_rate;
    int iterations;
    double train_ratio;
    int hidden_size;

    try {
        parse_arguments(argc, argv, train_file, learning_rate, iterations, train_ratio, hidden_size);
    } catch (const invalid_argument& e) {
        cerr << e.what() << endl;
        cerr << "Usage: " << argv[0]
             << " [--train_file <file>] [--learning_rate <rate>] [--iterations <count>]"
             << " [--train_ratio <ratio>] [--hidden_size <size>]" << endl;
        return EXIT_FAILURE;
    }

    // Load dataset on CPU
    auto data = load_csv(train_file);
    // Shuffle data on CPU
    random_shuffle(data.begin(), data.end());

    int m = data.size();
    int n = data[0].size();  // label + features
    int train_size = (int)(train_ratio * m);

    vector<vector<double>> train_data(data.begin(), data.begin() + train_size);
    vector<vector<double>> val_data(data.begin() + train_size, data.end());

    // X_train: train_size x (n-1)
    // Y_train: train_size
    vector<vector<double>> X_train(train_size, vector<double>(n - 1));
    vector<int> Y_train(train_size);

    for (int i = 0; i < train_size; ++i) {
        Y_train[i] = (int)train_data[i][0];
        for (int j = 1; j < n; ++j) {
            X_train[i][j - 1] = train_data[i][j] / 255.0;
        }
    }

    // X_val, Y_val
    int val_size = val_data.size();
    vector<vector<double>> X_val(val_size, vector<double>(n - 1));
    vector<int> Y_val(val_size);
    for (int i = 0; i < val_size; ++i) {
        Y_val[i] = (int)val_data[i][0];
        for (int j = 1; j < n; ++j) {
            X_val[i][j - 1] = val_data[i][j] / 255.0;
        }
    }

    // Initialize weights and biases on CPU
    auto W1_cpu = initialize_matrix_cpu(hidden_size, INPUT_SIZE);
    auto B1_cpu = initialize_vector_cpu(hidden_size);
    auto W2_cpu = initialize_matrix_cpu(OUTPUT_SIZE, hidden_size);
    auto B2_cpu = initialize_vector_cpu(OUTPUT_SIZE);

    // Flatten W1, B1, W2, B2 for GPU
    vector<double> h_W1(hidden_size * INPUT_SIZE);
    vector<double> h_B1(hidden_size);
    vector<double> h_W2(OUTPUT_SIZE * hidden_size);
    vector<double> h_B2(OUTPUT_SIZE);

    for (int h = 0; h < hidden_size; ++h) {
        for (int in = 0; in < INPUT_SIZE; ++in) {
            h_W1[h * INPUT_SIZE + in] = W1_cpu[h][in];
        }
        h_B1[h] = B1_cpu[h];
    }
    for (int out = 0; out < OUTPUT_SIZE; ++out) {
        for (int hh = 0; hh < hidden_size; ++hh) {
            h_W2[out * hidden_size + hh] = W2_cpu[out][hh];
        }
        h_B2[out] = B2_cpu[out];
    }

    // Flatten X_train
    vector<double> h_X(train_size * INPUT_SIZE);
    vector<int> h_Y(train_size);
    for (int i = 0; i < train_size; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            h_X[i * INPUT_SIZE + j] = X_train[i][j];
        }
        h_Y[i] = Y_train[i];
    }

    // Allocate device memory
    double *d_X, *d_W1, *d_B1, *d_Z1, *d_A1, *d_W2, *d_B2, *d_Z2, *d_A2;
    int* d_Y;
    // local partials
    double *d_localdW1, *d_localdB1, *d_localdW2, *d_localdB2;

    int size_X      = train_size * INPUT_SIZE * sizeof(double);
    int size_W1     = hidden_size * INPUT_SIZE * sizeof(double);
    int size_B1     = hidden_size * sizeof(double);
    int size_Z1A1   = train_size * hidden_size * sizeof(double);
    int size_W2     = OUTPUT_SIZE * hidden_size * sizeof(double);
    int size_B2     = OUTPUT_SIZE * sizeof(double);
    int size_Z2A2   = train_size * OUTPUT_SIZE * sizeof(double);
    int size_Y      = train_size * sizeof(int);

    // local partial gradients:
    int size_localdW1 = train_size * hidden_size * INPUT_SIZE * sizeof(double);
    int size_localdB1 = train_size * hidden_size * sizeof(double);
    int size_localdW2 = train_size * OUTPUT_SIZE * hidden_size * sizeof(double);
    int size_localdB2 = train_size * OUTPUT_SIZE * sizeof(double);

    checkCudaError(cudaMalloc((void**)&d_X,     size_X),      "malloc d_X");
    checkCudaError(cudaMalloc((void**)&d_Y,     size_Y),      "malloc d_Y");
    checkCudaError(cudaMalloc((void**)&d_W1,    size_W1),     "malloc d_W1");
    checkCudaError(cudaMalloc((void**)&d_B1,    size_B1),     "malloc d_B1");
    checkCudaError(cudaMalloc((void**)&d_Z1,    size_Z1A1),   "malloc d_Z1");
    checkCudaError(cudaMalloc((void**)&d_A1,    size_Z1A1),   "malloc d_A1");
    checkCudaError(cudaMalloc((void**)&d_W2,    size_W2),     "malloc d_W2");
    checkCudaError(cudaMalloc((void**)&d_B2,    size_B2),     "malloc d_B2");
    checkCudaError(cudaMalloc((void**)&d_Z2,    size_Z2A2),   "malloc d_Z2");
    checkCudaError(cudaMalloc((void**)&d_A2,    size_Z2A2),   "malloc d_A2");

    checkCudaError(cudaMalloc((void**)&d_localdW1, size_localdW1), "malloc d_localdW1");
    checkCudaError(cudaMalloc((void**)&d_localdB1, size_localdB1), "malloc d_localdB1");
    checkCudaError(cudaMalloc((void**)&d_localdW2, size_localdW2), "malloc d_localdW2");
    checkCudaError(cudaMalloc((void**)&d_localdB2, size_localdB2), "malloc d_localdB2");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_X,  h_X.data(),  size_X,  cudaMemcpyHostToDevice), "memcpy d_X");
    checkCudaError(cudaMemcpy(d_Y,  h_Y.data(),  size_Y,  cudaMemcpyHostToDevice), "memcpy d_Y");
    checkCudaError(cudaMemcpy(d_W1, h_W1.data(), size_W1, cudaMemcpyHostToDevice), "memcpy d_W1");
    checkCudaError(cudaMemcpy(d_B1, h_B1.data(), size_B1, cudaMemcpyHostToDevice), "memcpy d_B1");
    checkCudaError(cudaMemcpy(d_W2, h_W2.data(), size_W2, cudaMemcpyHostToDevice), "memcpy d_W2");
    checkCudaError(cudaMemcpy(d_B2, h_B2.data(), size_B2, cudaMemcpyHostToDevice), "memcpy d_B2");

    // Prepare CPU buffers for final inference results (A2) to compute training accuracy
    vector<double> h_A2(train_size * OUTPUT_SIZE);

    // Decide block / grid sizes
    int blockSize = 256;
    int gridSize  = (train_size + blockSize - 1) / blockSize;

    // For reduce kernels, we handle W2+B2 (OUTPUT_SIZE*hidden_size + OUTPUT_SIZE) elements total,
    // and W1+B1 (hidden_size*INPUT_SIZE + hidden_size) total.
    int totalW2B2 = OUTPUT_SIZE * hidden_size + OUTPUT_SIZE;
    int gridSizeW2B2 = (totalW2B2 + blockSize - 1) / blockSize;

    int totalW1B1 = hidden_size * INPUT_SIZE + hidden_size;
    int gridSizeW1B1 = (totalW1B1 + blockSize - 1) / blockSize;

    // Training loop
    for (int iter = 0; iter < iterations; ++iter) {

        // 1) Forward pass
        forwardPassKernel<<<gridSize, blockSize>>>(
            d_X, d_W1, d_B1, d_Z1, d_A1, d_W2, d_B2, d_Z2, d_A2,
            train_size, hidden_size
        );
        checkCudaError(cudaGetLastError(), "forwardPassKernel");

        // 2) Backward pass: fill local partial grads
        checkCudaError(cudaMemset(d_localdW1, 0, size_localdW1), "memset d_localdW1");
        checkCudaError(cudaMemset(d_localdB1, 0, size_localdB1), "memset d_localdB1");
        checkCudaError(cudaMemset(d_localdW2, 0, size_localdW2), "memset d_localdW2");
        checkCudaError(cudaMemset(d_localdB2, 0, size_localdB2), "memset d_localdB2");

        backwardPassKernel<<<gridSize, blockSize>>>(
            d_X, d_Y, d_Z1, d_A1, d_A2, d_W2,
            d_localdW1, d_localdB1, d_localdW2, d_localdB2,
            train_size, hidden_size
        );
        checkCudaError(cudaGetLastError(), "backwardPassKernel");

        // 3) Reduce & update W2, B2
        reduceAndUpdateW2B2<<<gridSizeW2B2, blockSize>>>(
            d_W2, d_B2, d_localdW2, d_localdB2, learning_rate,
            train_size, hidden_size
        );
        checkCudaError(cudaGetLastError(), "reduceAndUpdateW2B2");

        // 4) Reduce & update W1, B1
        reduceAndUpdateW1B1<<<gridSizeW1B1, blockSize>>>(
            d_W1, d_B1, d_localdW1, d_localdB1, learning_rate,
            train_size, hidden_size
        );
        checkCudaError(cudaGetLastError(), "reduceAndUpdateW1B1");

        // Optional: compute training accuracy every N steps
        if (iter % 20 == 0 || iter == iterations - 1) {
            // We already have A2 for the entire training set from the forward pass.
            // Copy it back to CPU
            checkCudaError(cudaMemcpy(h_A2.data(), d_A2, 
                                      train_size * OUTPUT_SIZE * sizeof(double), 
                                      cudaMemcpyDeviceToHost),
                           "memcpy d_A2->h_A2");
            // get predictions & compute training accuracy
            auto train_preds = get_predictions(h_A2, train_size);
            double train_acc = get_accuracy(train_preds, Y_train);

            // For validation, we can do forward pass on CPU or GPU. We'll do GPU for brevity:
            vector<double> h_Xval(val_size * INPUT_SIZE);
            for (int i = 0; i < val_size; ++i) {
                for (int j = 0; j < INPUT_SIZE; ++j) {
                    h_Xval[i * INPUT_SIZE + j] = X_val[i][j];
                }
            }
            // Copy X_val to d_X temporarily
            // (We reuse d_X, d_Z1, d_A1, d_Z2, d_A2 for inference)
            checkCudaError(cudaMemcpy(d_X, h_Xval.data(),
                                      val_size * INPUT_SIZE * sizeof(double),
                                      cudaMemcpyHostToDevice), 
                           "memcpy X_val -> d_X for val");

            // Forward pass for validation
            int valGridSize = (val_size + blockSize - 1) / blockSize;
            forwardPassKernel<<<valGridSize, blockSize>>>(
                d_X, d_W1, d_B1, d_Z1, d_A1, d_W2, d_B2, d_Z2, d_A2,
                val_size, hidden_size
            );
            checkCudaError(cudaGetLastError(), "forward val");

            // Copy A2 for val
            vector<double> h_A2val(val_size * OUTPUT_SIZE);
            checkCudaError(cudaMemcpy(h_A2val.data(), d_A2,
                                      val_size * OUTPUT_SIZE * sizeof(double),
                                      cudaMemcpyDeviceToHost),
                           "memcpy d_A2->h_A2val");

            // Compute val accuracy
            auto val_preds = get_predictions(h_A2val, val_size);
            double val_acc = get_accuracy(val_preds, Y_val);

            // Copy back the training set to d_X
            checkCudaError(cudaMemcpy(d_X, h_X.data(),
                                      size_X,
                                      cudaMemcpyHostToDevice),
                           "memcpy back train X -> d_X after val pass");

            cout << "Iteration: " << iter
                 << ", Training Accuracy: " << train_acc
                 << ", Validation Accuracy: " << val_acc << endl;
        }
    }

    cout << "Training complete!" << endl;

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_W1);
    cudaFree(d_B1);
    cudaFree(d_Z1);
    cudaFree(d_A1);
    cudaFree(d_W2);
    cudaFree(d_B2);
    cudaFree(d_Z2);
    cudaFree(d_A2);
    cudaFree(d_localdW1);
    cudaFree(d_localdB1);
    cudaFree(d_localdW2);
    cudaFree(d_localdB2);

    return 0;
}
