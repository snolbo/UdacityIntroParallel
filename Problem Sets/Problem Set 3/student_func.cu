/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <float.h>

__global__
void minmax_kernel(float* d_in, float* d_out, const size_t size, int findMax){
  // shared data allocated in kernel call as 3. argument
  extern __shared__ float sdata[];

  // Find local and global indices, and make sure we are within image
  int tid = threadIdx.x;
  int mid = blockIdx.x * blockDim.x + threadIdx.x;

  if(mid < size) {
      sdata[tid] = d_in[mid];
  } else {
      if(findMax)
          sdata[tid] = -FLT_MAX;
      else
          sdata[tid] = FLT_MAX;
  }
  __syncthreads();

  // Do reduction in shared memory
  for(unsigned int s = blockDim.x/2; s > 0; s /= 2) {
      if(tid < s) {
          if(findMax) {
            sdata[tid] = max(sdata[tid], sdata[tid+s]);
          } else {
            sdata[tid] = min(sdata[tid], sdata[tid+s]);
          }
      }
      __syncthreads();
  }

  // Write block's max to this block'a position in output array
  if(tid == 0){
    d_out[blockIdx.x] = sdata[0];
  }
}


// Perform required number of reductions until max element in d_all is found
float iterate_until_minmax_found(const float* const d_all, size_t size,
                                 const dim3 blockSize, const dim3 gridSize, int findMax){
  // Pointers to hold results of reductions
  float* d_in;
  float* d_out;
  // Allocate memory and copy values from d_in
  checkCudaErrors(cudaMalloc(&d_in, sizeof(float)*size));
  checkCudaErrors(cudaMemcpy(d_in, d_all, sizeof(float)*size, cudaMemcpyDeviceToDevice));
  // Defines variables to control partitioning
  int shared_memory_size = blockSize.x * sizeof(float);
  dim3 block_dim(blockSize.x);
  int array_size = (int) size;
  int num_blocks = (int) size / blockSize.x + 1;

  while(1){
    // Create memory for this iterations result
    checkCudaErrors(cudaMalloc(&d_out, sizeof(float) * num_blocks));
    // Produce new dimention of grid based on blocks needed to process reduction
    dim3 grid_dim(num_blocks);
    // Find this iterations values of block's local minmax values
    minmax_kernel<<<grid_dim, block_dim, shared_memory_size>>> (d_in, d_out, array_size, findMax);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // Free unneeded memory
    checkCudaErrors(cudaFree(d_in));

    d_in = d_out;
    // If above calculatons was performed in a single block, break from loop
    if(array_size < blockSize.x){
      break;
    }
    // Update information about partitioning
    array_size = num_blocks;
    num_blocks = (int) num_blocks / blockSize.x + 1;
  }
  // Get minmax values, free memory and return
  float minmax;
  cudaMemcpy(&minmax, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return minmax;
}



// __global__
// void histogram_kernel(const float* const d_in, unsigned int* d_out, float lumMin, float lumRange,
  //             size_t numBins, size_t size, int elements_per_thread){
  // // Shared memory for this block, initialize all values to 0
  // extern __shared__ float sdata[];
  // int init_id = threadIdx.x;
  // while(init_id < size){
  //   sdata[init_id] = 0;
  //   init_id += blockDim.x;
  // }
  // __syncthreads();
  //
  // // Find id based on partitioning, including elements processed per thread
  // int tid = threadIdx.x;
  // int mid = blockIdx.x * (blockDim.x * elements_per_thread) + threadIdx.x;
  //
  // // Fill local bins
  // //Only threads inside image gets to update with image values, rest is 0
  //   if (mid < size){
  //     for(int i = 0; i < elements_per_thread; i++){
  //       if(mid + i >= size){
  //         break; // Break if this thread wants to process elements outside image
  //       }
  //       float value = d_in[mid + i];
  //       int binNum  = (int) floor(((value - lumMin) / lumRange * numBins));
  //       sdata[mid + binNum]++;
  //     }
  //   }
  // // // Now all threads should have filled bins with values incremented by values form
  // // // local image or set to 0. Ready to reduce bins
  // __syncthreads();
  //
  // // // Perform reduction
  // for(int s = blockDim.x / 2; s > 0; s>>=1){
  //   // Half of threads each iterations reads, modifies, and writes
  //   if(tid < s){
  //     int t_index = tid * elements_per_thread;
  //     int s_index = s   * elements_per_thread;
  //     for(int i = 0; i < elements_per_thread; i++){
  //       sdata[t_index + i]  += sdata[s_index + 1];
  //     }
  //   }
  //   // Sync for each reduction
  //   __syncthreads();
  // }
  //
  // // Now this block's histogram values should be collected in the numBins first bins
  // // of shared memory
  // if(blockIdx.x == 0){
  //   for(int i = 0; i < numBins; i++){
  //     d_out[i] = atomicAdd(sdata[i], 1);
  //   }
  // }
// }



__global__
void atomic_histogram_kernel(const float* const d_in, unsigned int* d_out, float lumMin, float lumRange, size_t size, size_t numBins){
  // Shared memory for this block, initialize all values to 0
  int mid = blockIdx.x *blockDim.x + threadIdx.x;
  if(mid >= size){
    return;
  }
  int bin = ((d_in[mid]-lumMin) / lumRange) * numBins;
  atomicAdd(&d_out[bin], 1);
}


__global__
void hillis_scan_kernel(unsigned int* bins, size_t size){
  int mid = blockIdx.x * blockDim.x + threadIdx.x;
  if(mid >= size){
    return;
  }

  for(int s = 1; s <= size; s<<=1){
    int index = mid - s;
    int value = 0;
    if( index >= 0){
      value = bins[index];
    }
    __syncthreads();
    if(index >= 0){
      bins[mid] += value;
    }
    __syncthreads();
  }
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                    unsigned int* const d_cdf, // Store final result
                    float &min_logLum,  float &max_logLum,
                    const size_t numRows, const size_t numCols, const size_t numBins)
{
 // Define partitioning
  int threadsPerBlock = 1024;
  const size_t size = numCols * numRows;
  const dim3 block_dim(threadsPerBlock);
  const dim3 grid_dim( (numRows*numCols)/block_dim.x + 1);


  //TODO
  // Here are the steps you need to implement
  //   1) find the minimum and maximum value in the input logLuminance channel
  //      store in min_logLum and max_logLum
  min_logLum = iterate_until_minmax_found(d_logLuminance, size, block_dim, grid_dim, 0);
  max_logLum = iterate_until_minmax_found(d_logLuminance, size, block_dim, grid_dim, 1);
  // min_logLum = reduce_minmax(d_logLuminance, size, 0);
  // max_logLum = reduce_minmax(d_logLuminance, size, 1);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // min_logLum = 0.3;
  //   2) subtract them to find the range
  float lumRange = max_logLum - min_logLum;

  //   3) generate a histogram of all the values in the logLuminance channel using
  //      the formula: bin = (lum[i] - lumMin) / lumRange * numBins

  // int elements_per_thread = 512;
  unsigned int* histogram;
  size_t hist_size = sizeof(unsigned int) * numBins;

  int shared_memory_size = sizeof(float) * threadsPerBlock * numBins;
  checkCudaErrors(cudaMalloc(&histogram,   hist_size));
  checkCudaErrors(cudaMemset(histogram, 0, hist_size));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // histogram_kernel<<<grid_dim, block_dim, shared_memory_size>>>
  // (d_logLuminance, histogram, min_logLum, lumRange, numBins, size, elements_per_thread);

  dim3 hist_grid_dim(size/block_dim.x + 1);
  atomic_histogram_kernel<<<hist_grid_dim, block_dim>>>
  (d_logLuminance, histogram, min_logLum, lumRange, size, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Number of threads needed equals numBins

  dim3 scan_grid_dim(numBins/threadsPerBlock +1);

  hillis_scan_kernel<<<scan_grid_dim, block_dim>>>(histogram, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(d_cdf, histogram, sizeof(int) * numBins, cudaMemcpyDeviceToDevice));
  cudaFree(histogram);
}
