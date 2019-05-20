#include <iostream>
#include <stdio.h>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>
#include <omp.h>

using std::cout;
using std::endl;

typedef unsigned long long Count;
typedef std::numeric_limits<double> DblLim;

const Count WARP_SIZE = 32; // Warp size
const Count NBLOCKS = 640; // Number of total cuda cores on my GPU
const Count ITERATIONS = 1000000; // Number of points to generate (each thread)

// This kernel is 
__global__ void picount(Count *totals) {
	// Define some shared memory: all threads in this block
	__shared__ Count counter[WARP_SIZE];

	// Unique ID of the thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Initialize RNG
	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	// Initialize the counter
	counter[threadIdx.x] = 0;

	// Computation loop
	for (int i = 0; i < ITERATIONS; i++) {
		float x = curand_uniform(&rng); // Random x position in [0,1]
		float y = curand_uniform(&rng); // Random y position in [0,1]
		counter[threadIdx.x] += 1 - int(x * x + y * y); // Hit test
	}

	// The first thread in *every block* should sum the results
	if (threadIdx.x == 0) {
		// Reset count for this block
		totals[blockIdx.x] = 0;
		// Accumulate results
		for (int i = 0; i < WARP_SIZE; i++) {
			totals[blockIdx.x] += counter[i];
		}
	}
}

int main(int argc, char **argv) {
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		cout << "CUDA device missing! Do you need to use optirun?\n";
		return 1;
	}
	clock_t start_d=clock();
	cout << "Starting simulation with " << NBLOCKS << " blocks, " << WARP_SIZE << " threads, and " << ITERATIONS << " iterations\n";

	// Allocate host and device memory to store the counters
	Count *hOut, *dOut;
	hOut = new Count[NBLOCKS]; // Host memory
	cudaMalloc(&dOut, sizeof(Count) * NBLOCKS); // Device memory

	// Launch kernel
	picount<<<NBLOCKS, WARP_SIZE>>>(dOut);

	// Copy back memory used on device and free
	cudaMemcpy(hOut, dOut, sizeof(Count) * NBLOCKS, cudaMemcpyDeviceToHost);
	cudaFree(dOut);

	// Compute total hits
	Count total = 0;
	for (int i = 0; i < NBLOCKS; i++) {
		total += hOut[i];
	}
	Count tests = NBLOCKS * ITERATIONS * WARP_SIZE;
	cout << "Approximated PI using " << tests << " random tests\n";

	// Set maximum precision for decimal printing
//	cout.precision(DblLim::max_digits10);
	cout << "PI ~= " << 4.0 * (double)total/(double)tests << endl;
	clock_t end_d = clock();
	double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    printf("\n Seconds: %fs\n",time_d);


	return 0;
}
