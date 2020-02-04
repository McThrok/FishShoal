/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /*
  * CUDA particle system kernel code.
  */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams params;
__constant__ float eps = 1e-7;


struct integrate_functor
{
	float deltaTime;

	__host__ __device__
		integrate_functor(float delta_time) : deltaTime(delta_time) {}

	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
		volatile float4 posData = thrust::get<0>(t);
		volatile float4 velData = thrust::get<1>(t);
		float2 pos = make_float2(posData.x, posData.y);
		float2 vel = setLength(make_float2(velData.x, velData.y), params.maxSpeed);

		pos += vel * deltaTime;

		float sqs = params.squareSize / 2;

		if (pos.x > sqs) pos.x = -sqs;
		if (pos.x < -sqs) pos.x = sqs;

		if (pos.y > sqs) pos.y = -sqs;
		if (pos.y < -sqs) pos.y = sqs;

		// store new position and velocity
		thrust::get<0>(t) = make_float4(pos.x, pos.y, 0, posData.w);
		thrust::get<1>(t) = make_float4(vel.x, vel.y, 0, velData.w);
	}
};

// calculate position in uniform grid
__device__ int2 calcGridPos(float2 p)
{
	int2 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	return gridPos;
}

// calculate address in grid from position
__device__ uint calcGridHash(int2 gridPos)
{
	//return __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
	return gridPos.y * params.gridSize.x + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint* gridParticleHash,  // output
	uint* gridParticleIndex, // output
	float4* pos,               // input: positions
	uint    numParticles)
{
	//uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	volatile float4 p = pos[index];

	// get address in grid
	int2 gridPos = calcGridPos(make_float2(p.x, p.y));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint* cellStart,        // output: cell start index
	uint* cellEnd,          // output: cell end index
	float4* sortedPos,        // output: sorted positions
	float4* sortedVel,        // output: sorted velocities
	uint* gridParticleHash, // input: sorted grid hashes
	uint* gridParticleIndex,// input: sorted particle indices
	float4* oldPos,           // input: sorted position array
	float4* oldVel,           // input: sorted velocity array
	uint    numParticles)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	cg::sync(cta);

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		float4 pos = oldPos[sortedIndex];
		float4 vel = oldVel[sortedIndex];

		sortedPos[index] = pos;
		sortedVel[index] = vel;
	}


}

__device__ float2 limit(float2 v, float l)
{
	float dist = length(v);
	return dist <= l ? v : (v / dist) * l;
}

__device__ float2 setLength(float2 v, float l)
{
	float dist = length(v);
	return dist <= eps ? v : (v / dist) * l;
}

__device__ float angleBetween(float2 v1, float2 v2)
{
	float d1 = length(v1);
	float d2 = length(v2);
	return (v1.x * v2.y - v1.y * v2.x) / d1 / d2;
}

__device__
float2 calculateAcceleration(
	uint    index,
	float2  pos,
	float2  vel,
	float4* oldPos,
	float4* oldVel,
	uint* cellStart,
	uint* cellEnd)
{
	int2 gridPos = calcGridPos(pos);

	float2 sep_sum = make_float2(0, 0);
	float2 alg_sum = make_float2(0, 0);
	float2 coh_sum = make_float2(0, 0);

	int sep_n = 0, alg_n = 0, coh_n = 0;

	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			int2 neightbourGrid = gridPos + make_int2(x, y);

			float2 offset = make_float2(0, 0);
			if (neightbourGrid.x == -1) { neightbourGrid.x = params.gridSize.x - 1; offset.x -= params.squareSize; }
			else if (neightbourGrid.x == params.gridSize.x) { neightbourGrid.x = 0; offset.x += params.squareSize; }
			if (neightbourGrid.y == -1) { neightbourGrid.y = params.gridSize.y - 1; offset.y -= params.squareSize; }
			else if (neightbourGrid.y == params.gridSize.y) { neightbourGrid.y = 0; offset.y += params.squareSize; }

			uint gridHash = calcGridHash(neightbourGrid);

			// get start of bucket for this cell
			uint startIndex = cellStart[gridHash];
			//uint startIndex = 0;

			if (startIndex != 0xffffffff)          // cell is not empty
			{
				// iterate over particles in this cell
				uint endIndex = cellEnd[gridHash];
				//uint endIndex = params.numBodies;

				for (uint j = startIndex; j < endIndex; j++)
				{
					if (j == index) continue;               // check not colliding with self

					float2 pos2 = make_float2(oldPos[j].x, oldPos[j].y) + offset;
					float2 toVec = pos2 - pos;
					float dist = length(toVec);
					float r = params.particleRadius;

					if (r < dist) continue;

					float cosVision = cosf(params.visionAngle * CUDART_PI_F / 180);
					float cosToVec = angleBetween(toVec, vel);

					if (cosVision > cosToVec) continue;

					if (r * params.separationRadius >= dist)
					{
						sep_n++;
						sep_sum -= toVec / dist / dist;
					}

					if (r * params.alignmentRadius >= dist)
					{
						alg_n++;
						alg_sum += make_float2(oldVel[j].x, oldVel[j].y);
					}

					if (r * params.cohesionRadius >= dist)
					{
						coh_n++;
						coh_sum += pos2;
					}

				}
			}
		}
	}

	float2 sep_acc = make_float2(0, 0);
	if (sep_n)
	{
		sep_acc = sep_sum / sep_n;
		sep_acc = setLength(sep_acc, params.maxSpeed) - vel;
		sep_acc = setLength(sep_acc, params.maxAcceleration) * params.separationFactor;
	}

	float2 alg_acc = make_float2(0, 0);
	if (alg_n)
	{
		alg_acc = alg_sum / alg_n;
		alg_acc = setLength(alg_acc, params.maxSpeed) - vel;
		alg_acc = setLength(alg_acc, params.maxAcceleration) * params.alignmentFactor;
	}


	float2 coh_acc = make_float2(0, 0);
	if (coh_n)
	{
		coh_acc = coh_sum / coh_n - pos;
		coh_acc = setLength(coh_acc, params.maxSpeed) -vel;
		coh_acc = setLength(coh_acc, params.maxAcceleration) * params.cohesionFactor;
	}

	return sep_acc + alg_acc + coh_acc;
}

__global__
void collideD(float4* newVel,               // output: new velocity
	float4* oldPos,               // input: sorted positions
	float4* oldVel,               // input: sorted velocities
	uint* gridParticleIndex,    // input: sorted particle indices
	uint* cellStart,
	uint* cellEnd,
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float2 pos = make_float2(oldPos[index].x, oldPos[index].y);
	float2 vel = make_float2(oldVel[index].x, oldVel[index].y);

	// examine neighbouring cells
	float2 acc = calculateAcceleration(index, pos, vel, oldPos, oldVel, cellStart, cellEnd);

	// write new velocity back to original unsorted location
	uint originalIndex = gridParticleIndex[index];
	float2 new_vel = setLength(vel + acc, params.maxSpeed);
	newVel[originalIndex] = make_float4(new_vel.x, new_vel.y, 0.0f, 0.0f);
}

#endif
