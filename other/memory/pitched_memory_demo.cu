/*
 * pitched_memory_demo.cu
 *
 *  Created on: Oct 21, 2020
 *      Author: jingsz
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

int main(int argc, char **argv)
{
	// device pointers.
	float *d_pitch;
	float *d_normal;

	// matrix size.
	size_t cols = 256;
	size_t rows = 16;

	size_t pitch = 0;

	// alloc the data form gpu memory.
	cudaMallocPitch((void**)&d_pitch, &pitch, cols*sizeof(float), rows);
	cudaMalloc((void**)(&d_normal), rows*cols*sizeof(float));

	// test the data address.
	fprintf(stdout, "pitch = %lu \n", pitch);
	fprintf(stdout, "row size(in bytes) = %.2f*128.\n", pitch/128.0f);
	fprintf(stdout, "the head address of d_pitch  mod 128 = %x.\n", (unsigned int)((size_t)d_pitch)%128);
	fprintf(stdout, "the head address of d_normal mod 128 = %x.\n", (unsigned int)((size_t)d_normal)%128);

	cudaFree(d_pitch);
	cudaFree(d_normal);

	getchar();
	return 0;
}



