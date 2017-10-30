#include <wb.h>

#define wbCheck(stmt)                                       \
    do {                                                    \
        cudaError_t err = stmt;                             \
        if (err != cudaSuccess) {                           \
            wbLog(ERROR, "Failed to run stmt ", #stmt);     \
            return -1;                                      \
        }                                                   \
    } while(0)

#define BLOCK_SIZE 512 //@@ You can change this

__global__ void total(float * input, float * output, int len)
{
    //@@ Load a segment of the input vector into shared memory
    __shared__ float sum[BLOCK_SIZE * 2];
    
    int t = threadIdx.x;
    int start = 2 * blockIdx.x * BLOCK_SIZE + t;

    sum[t] = start < len ? input[start] : 0.0;
    if (start + BLOCK_SIZE < len)
        sum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE];
    else
        sum[BLOCK_SIZE + t] = 0.0;

    //@@ Traverse the reduction tree
    for (int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride)
            sum[t] += sum[t + stride];
    }

    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
        output[blockIdx.x] = sum[0];
}



int main(int argc, char ** argv)
{
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list
    int sizeInput, sizeOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0),
            &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE << 1);
    if (numInputElements % (BLOCK_SIZE << 1))
        numOutputElements++;

    sizeInput = numInputElements * sizeof(float);
    sizeOutput = numOutputElements * sizeof(float);

    hostOutput = (float*) malloc(sizeOutput);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
            numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ",
            numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc(&deviceInput, sizeInput);
    cudaMalloc(&deviceOutput, sizeOutput);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, sizeInput, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(numOutputElements, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, sizeOutput, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/

    for (int i = 1; i < numOutputElements; ++i) {
        hostOutput[0] += hostOutput[i];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}