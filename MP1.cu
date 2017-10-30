// MP 1  
#include <wb.h>  

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {  
    //@@ Insert code to implement vector addition here  
    // Get our global thread ID  
    int id = blockIdx.x*blockDim.x+threadIdx.x;  
    // Make sure we do not go out of bounds  
    if (id < len)  
    	out[id] = in1[id] + in2[id]; 
}  

int main(int argc, char ** argv) {  
	wbArg_t args;  
	int inputLength;  
	float * hostInput1;  
	float * hostInput2;  
	float * hostOutput;  
	float * deviceInput1;  
	float * deviceInput2;  
	float * deviceOutput;  

	args = wbArg_read(argc, argv);  

	wbTime_start(Generic, "Importing data and creating memory on host");  
	hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);  
	hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);  
	hostOutput = (float *) malloc(inputLength * sizeof(float));  
	wbTime_stop(Generic, "Importing data and creating memory on host");  

	wbLog(TRACE, "The input length is ", inputLength);  

	wbTime_start(GPU, "Allocating GPU memory.");  
	//@@ Allocate GPU memory here  
	size_t bytes = inputLength*sizeof(float);  
	cudaMalloc(&deviceInput1, bytes);  
	cudaMalloc(&deviceInput2, bytes);  
	cudaMalloc(&deviceOutput, bytes);  

	wbTime_stop(GPU, "Allocating GPU memory.");  

	wbTime_start(GPU, "Copying input memory to the GPU.");  
	//@@ Copy memory to the GPU here  
	cudaMemcpy( deviceInput1, hostInput1, bytes, cudaMemcpyHostToDevice);  
	cudaMemcpy( deviceInput2, hostInput2, bytes, cudaMemcpyHostToDevice);  

	wbTime_stop(GPU, "Copying input memory to the GPU.");  

	//@@ Initialize the grid and block dimensions here  
	// Number of threads in each thread block  
	int blockSize = 1024;  
	// Number of thread blocks in grid  
	int gridSize = (int)ceil((float)inputLength/blockSize);  

	wbTime_start(Compute, "Performing CUDA computation");  
	//@@ Launch the GPU Kernel here  
	vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);  

	cudaThreadSynchronize();  
	wbTime_stop(Compute, "Performing CUDA computation");  

	wbTime_start(Copy, "Copying output memory to the CPU");  
	//@@ Copy the GPU memory back to the CPU here  
	cudaMemcpy( hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost );  

	wbTime_stop(Copy, "Copying output memory to the CPU");  

	wbTime_start(GPU, "Freeing GPU Memory");  
	//@@ Free the GPU memory here  
	cudaFree(deviceInput1);  
	cudaFree(deviceInput2);  
	cudaFree(deviceOutput);  

	wbTime_stop(GPU, "Freeing GPU Memory");  

	wbSolution(args, hostOutput, inputLength);  

	free(hostInput1);  
	free(hostInput2); 
	free(hostOutput);  

	return 0;  
}  