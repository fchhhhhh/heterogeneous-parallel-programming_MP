#include <wb.h> //@@ wb include opencl.h for you

#define wbCheck(stmt) 
do {                                                    \
    cl_int err = stmt;                                               \
    if (err != CL_SUCCESS) {                                             \
        wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
        wbLog(ERROR, "Got OPENCL error ...  ", getErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while(0)

const char *getErrorString(cl_int error)
{
    switch(error){
            // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            
            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
            
            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

int main(int argc, char **argv)
{
    wbArg_t args;
    int inputLength;
    float *a;
    float *b;
    float *c;

    args = wbArg_read(argc, argv);
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    a = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
    b = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
    c = ( float * )malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbLog(TRACE, "The input length is ", inputLength);

    // Open CL Preparation
    cl_int clerr = CL_SUCCESS;
    cl_platform_id platform;
    clerr = clGetPlatformIDs(1, &platform, NULL);
    wbCheck(clerr);
    cl_device_id* device = (cl_device_id *)malloc(sizeof(cl_device_id)*1000);
    clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, device, NULL);
    wbCheck(clerr);
    cl_context clctx = clCreateContext(0, 1, device, NULL, NULL, &clerr);
    wbCheck(clerr);
    
    size_t parmsz;
    clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);
    wbCheck(clerr);

    cl_device_id* cldevs = (cl_device_id *) malloc(parmsz);
    clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);
    wbCheck(clerr);
    
    cl_command_queue clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr);
    wbCheck(clerr);

    const char* vaddsrc =
    "__kernel void vadd(__global const float* a, __global const float* b, __global float* res, int N)"
    "{"
    "    int i = get_global_id(0);"
    ""
    "    if (i<N)"
    "        res[i] = a[i] + b[i];"
    "}";


    cl_program clpgm;
    clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc, NULL, &clerr);
    wbCheck(clerr);
    
    char clcompileflags[4096];
    sprintf(clcompileflags, "-cl-mad-enable");
    clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL);
    wbCheck(clerr);
    cl_kernel clkern = clCreateKernel(clpgm, "vadd", &clerr);
    wbCheck(clerr);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cl_mem d_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY , inputLength*sizeof(float), NULL, NULL);
    cl_mem d_b = clCreateBuffer(clctx, CL_MEM_READ_ONLY , inputLength*sizeof(float), NULL, NULL);
    cl_mem d_c = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, inputLength*sizeof(float), NULL, NULL);
    
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    clEnqueueWriteBuffer(clcmdq, d_a, CL_TRUE, 0, inputLength*sizeof(float), (const void *)a, 0, 0, NULL);
    clEnqueueWriteBuffer(clcmdq, d_b, CL_TRUE, 0, inputLength*sizeof(float), (const void *)b, 0, 0, NULL);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing Kernel computation");
    //@@ Launch the GPU Kernel here
    clerr = clSetKernelArg(clkern, 0, sizeof(cl_mem), (void *)&d_a); wbCheck(clerr);
    clerr = clSetKernelArg(clkern, 1, sizeof(cl_mem), (void *)&d_b); wbCheck(clerr);
    clerr = clSetKernelArg(clkern, 2, sizeof(cl_mem), (void *)&d_c); wbCheck(clerr);
    clerr = clSetKernelArg(clkern, 3, sizeof(int)   , &inputLength); wbCheck(clerr);

    int    n      = 32;
    size_t Gsz[1] = {((inputLength+1)/(n-1))*n};
    size_t Lsz[1] = {n};
    cl_event event = NULL;
    clerr = clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, Gsz, Lsz, 0, NULL, &event);
    wbCheck(clerr);
    clerr = clWaitForEvents(1, &event);
    wbCheck(clerr);
    wbTime_stop(Compute, "Performing Kernel computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    clEnqueueReadBuffer(clcmdq, d_c, CL_TRUE, 0, inputLength*sizeof(float), (void *)c, 0, 0, NULL);

    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, c, inputLength);

    free(a);
    free(b);
    free(c);

    return 0;
}