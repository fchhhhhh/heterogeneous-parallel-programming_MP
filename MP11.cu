// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define I_TILE_WIDTH     16
#define SCAN_BLOCK       128

//@@ insert code here
__global__ void convertFloatToUChar(const float* __restrict__ fImageData, unsigned char* uCImageData, int iW, int iH, int iC)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    for (int c=0; c<iC; ++c)
    {
        int idx = (row*iW+col)*iC+c;
        if (row<iH && col<iW)
            uCImageData[idx] = (unsigned char) (255*fImageData[idx]);
    }
}

__global__ void convertRGBtoGray(const unsigned char* __restrict__ rgbImageData, unsigned char* grayImageData, int iW, int iH, int iC)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    int idxI = row*iW+col;
    int idxO = idxI*iC;
    int idxR = idxO;
    int idxG = idxO+1;
    int idxB = idxO+2;    

    if (row<iH && col<iW)
        grayImageData[idxI] = (unsigned char) (0.21*rgbImageData[idxR] + 0.71*rgbImageData[idxG] + 0.07*rgbImageData[idxB]);
}

__global__ void computeHistogram(const unsigned char* __restrict__ grayImageData, unsigned int* histogram, long len)
{
    __shared__ unsigned int p_hist[HISTOGRAM_LENGTH];    

    int tx = threadIdx.x;
    if (tx<HISTOGRAM_LENGTH)
        p_hist[tx] = 0;
    __syncthreads();    

    int stride = blockDim.x * gridDim.x;
    int idx    = blockIdx.x*blockDim.x + tx;
    while (idx<len)
    {
        atomicAdd(&(p_hist[grayImageData[idx]]),1);
        idx += stride;
    }
    __syncthreads();    
    if (tx<HISTOGRAM_LENGTH)
        atomicAdd(&(histogram[tx]), p_hist[tx]);
}

__global__ void computeCDF(const unsigned int* __restrict__ iHist, float* cdf, long norm)
{
    __shared__ float data[2*SCAN_BLOCK];
    int tx  = threadIdx.x;
    int tx2 = tx + SCAN_BLOCK;
    int x   = 2*blockDim.x*blockIdx.x + tx;
    int x2  = x + SCAN_BLOCK;

    float dnorm = 1./norm;
    
    data[tx]  = x <HISTOGRAM_LENGTH ? iHist[x] *dnorm : 0.;
    data[tx2] = x2<HISTOGRAM_LENGTH ? iHist[x2]*dnorm : 0.;

    for (int stride=1; stride<=SCAN_BLOCK; stride*=2)
    {
        __syncthreads();
        int idx = (tx+1)*2*stride-1;
        if (idx<2*SCAN_BLOCK)
            data[idx] += data[idx-stride];
    }

    for (int stride=SCAN_BLOCK/2; stride>0; stride/=2)
    {
        __syncthreads();
        int idx = (tx+1)*2*stride-1;
        if (idx+stride<2*SCAN_BLOCK)
            data[idx+stride] += data[idx];
    }

    __syncthreads();
    if (x<HISTOGRAM_LENGTH)
        cdf[x]  = data[tx];
    if (x2<HISTOGRAM_LENGTH)
        cdf[x2] = data[tx2];
}

__global__ void correctColor(const float* __restrict__ cdf, unsigned char* uCImageData, int iW, int iH, int iC)
{
    float cdfmin = cdf[0]; 

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    for (int c=0; c<iC; ++c)
    {
        int idx = (row*iW+col)*iC+c;
        if (row<iH && col<iW)
        {
            unsigned char val = uCImageData[idx];
            unsigned char img = (unsigned char) 255*(cdf[val] - cdfmin)/(1 - cdfmin);
            uCImageData[idx]  = min(max(img, 0), HISTOGRAM_LENGTH-1);
        }
    }
}

__global__ void convertUCharToFloat(const unsigned char* __restrict__ uCImageData, float* fImageData, int iW, int iH, int iC)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    for (int c=0; c<iC; ++c)
    {
        int idx = (row*iW+col)*iC+c;
        if (row<iH && col<iW)
            fImageData[idx] = (float) (uCImageData[idx]/255.);
    }
}

int main(int argc, char ** argv)
{
    wbArg_t     args;
    int         imageWidth;
    int         imageHeight;
    int         imageChannels;
    wbImage_t   inputImage;
    wbImage_t   outputImage;
    float*      hostInputImageData;
    float*      hostOutputImageData;
    const char* inputImageFile;

    float*      deviceInputImageData;
    float*      deviceOutputImageData;

    unsigned char* uCharDeviceImageData;
    unsigned char* grayDeviceImageData;
    unsigned int*  histogram;
    float*         cdf;
    
    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage    = wbImport(inputImageFile);
    imageWidth    = wbImage_getWidth(inputImage);
    imageHeight   = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage   = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    //@@ insert code here
    hostInputImageData  = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData , imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &uCharDeviceImageData , imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMalloc((void **) &grayDeviceImageData  , imageWidth * imageHeight * sizeof(unsigned char));
    cudaMalloc((void **) &histogram            , HISTOGRAM_LENGTH * sizeof(unsigned int));
    cudaMalloc((void **) &cdf                  , HISTOGRAM_LENGTH * sizeof(float));

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    // define dim3 objects
    long imageSize = imageWidth*imageHeight;
    int  ths       = I_TILE_WIDTH*I_TILE_WIDTH;

    dim3 blocks1 ((imageWidth-1)/I_TILE_WIDTH+1, (imageHeight-1)/I_TILE_WIDTH+1, 1);
    dim3 threads1(I_TILE_WIDTH, I_TILE_WIDTH, 1);
    
    dim3 blocks2 ((imageSize-1)/ths+1, 1, 1);
    dim3 threads2(ths                , 1, 1);

    dim3 blocks3 (1         , 1, 1);
    dim3 threads3(SCAN_BLOCK, 1, 1);

    // conversion to uChar
    convertFloatToUChar<<<blocks1, threads1>>>(deviceInputImageData, uCharDeviceImageData, imageWidth, imageHeight, imageChannels);

    // conversion to Gray
    convertRGBtoGray<<<blocks1, threads1>>>(uCharDeviceImageData, grayDeviceImageData, imageWidth, imageHeight, imageChannels);

    // compute Histogram
    computeHistogram<<<blocks2, threads2>>>(grayDeviceImageData, histogram, imageSize);

    // compute CDF
    computeCDF<<<blocks3, threads3>>>(histogram, cdf, imageSize);

    // compute correction
    correctColor<<<blocks1, threads1>>>(cdf, uCharDeviceImageData, imageWidth, imageHeight, imageChannels);

    // convert back to Float
    convertUCharToFloat<<<blocks1, threads1>>>(uCharDeviceImageData, deviceOutputImageData, imageWidth, imageHeight, imageChannels);

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbSolution(args, outputImage);

    //@@ insert code here
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(uCharDeviceImageData);
    cudaFree(grayDeviceImageData);
    cudaFree(histogram);
    cudaFree(cdf);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

