#include <wb.h>

#define NB_STREAM  4
#define SEG_SIZE   256
#define NB_THREADS 256
#define NB_BLOCKS  (SEG_SIZE-1)/NB_THREADS+1

__global__ void vecAdd(float* in1, float* in2, float* out, int len)
{
    //@@ Insert code to implement vector addition here
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<len)
        out[i] = in1[i]+in2[i];
}

int main(int argc, char ** argv)
{
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    cudaStream_t stream0, stream1, stream2, stream3;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaMalloc((void**) &deviceInput1, inputLength*sizeof(float));
    cudaMalloc((void**) &deviceInput2, inputLength*sizeof(float));
    cudaMalloc((void**) &deviceOutput, inputLength*sizeof(float));

    for (int i=0; i<inputLength; i+=NB_STREAM*SEG_SIZE)
    {
        int idx0 = i;
        int idx1 = idx0+SEG_SIZE;
        int idx2 = idx1+SEG_SIZE;
        int idx3 = idx2+SEG_SIZE;
        int size0 = max(min(SEG_SIZE, inputLength-idx0),0);
        int size1 = max(min(SEG_SIZE, inputLength-idx1),0);
        int size2 = max(min(SEG_SIZE, inputLength-idx2),0);
        int size3 = max(min(SEG_SIZE, inputLength-idx3),0);

        cudaMemcpyAsync(deviceInput1+idx0, hostInput1+idx0, size0*sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(deviceInput2+idx0, hostInput2+idx0, size0*sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(deviceInput1+idx1, hostInput1+idx1, size1*sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(deviceInput2+idx1, hostInput2+idx1, size1*sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(deviceInput1+idx2, hostInput1+idx2, size2*sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(deviceInput2+idx2, hostInput2+idx2, size2*sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(deviceInput1+idx3, hostInput1+idx3, size3*sizeof(float), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(deviceInput2+idx3, hostInput2+idx3, size3*sizeof(float), cudaMemcpyHostToDevice, stream3);

        vecAdd<<<NB_BLOCKS, NB_THREADS, 0, stream0>>>(deviceInput1+idx0, deviceInput2+idx0, deviceOutput+idx0, size0);
        vecAdd<<<NB_BLOCKS, NB_THREADS, 0, stream1>>>(deviceInput1+idx1, deviceInput2+idx1, deviceOutput+idx1, size1);
        vecAdd<<<NB_BLOCKS, NB_THREADS, 0, stream2>>>(deviceInput1+idx2, deviceInput2+idx2, deviceOutput+idx2, size2);
        vecAdd<<<NB_BLOCKS, NB_THREADS, 0, stream3>>>(deviceInput1+idx3, deviceInput2+idx3, deviceOutput+idx3, size3);

        cudaMemcpyAsync(hostOutput+idx0, deviceOutput+idx0, size0*sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(hostOutput+idx1, deviceOutput+idx1, size1*sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(hostOutput+idx2, deviceOutput+idx2, size2*sizeof(float), cudaMemcpyDeviceToHost, stream2);
        cudaMemcpyAsync(hostOutput+idx3, deviceOutput+idx3, size3*sizeof(float), cudaMemcpyDeviceToHost, stream3);
    }
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}