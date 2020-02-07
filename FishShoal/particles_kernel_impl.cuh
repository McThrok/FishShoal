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
		volatile float2 posData = thrust::get<0>(t);
		volatile float2 velData = thrust::get<1>(t);
		float2 pos = make_float2(posData.x, posData.y);
		float2 vel = setLength(make_float2(velData.x, velData.y), params.maxSpeed);

		pos += vel * deltaTime;

		float wdiv2 = params.width / 2;
		if (pos.x > wdiv2) pos.x = -wdiv2;
		if (pos.x < -wdiv2) pos.x = wdiv2;

		float hdiv2 = params.height / 2;
		if (pos.y > hdiv2) pos.y = -hdiv2;
		if (pos.y < -hdiv2) pos.y = hdiv2;

		
		thrust::get<0>(t) = pos;
		thrust::get<1>(t) = vel;
	}
};
__device__ int2 calcGridPos(float2 p)
{
	int2 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	return gridPos;
}
__device__ uint calcGridHash(int2 gridPos)
{
	return gridPos.y * params.gridSize.x + gridPos.x;
}
__global__
void calcHashD(uint* gridParticleHash,  
	uint* gridParticleIndex, 
	float2* pos,               
	uint    numParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	volatile float2 p = pos[index];

	
	int2 gridPos = calcGridPos(make_float2(p.x, p.y));
	uint hash = calcGridHash(gridPos);

	
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

__global__
void reorderDataAndFindCellStartD(uint* cellStart,        
	uint* cellEnd,          
	float2* sortedPos,        
	float2* sortedVel,        
	uint* gridParticleHash, 
	uint* gridParticleIndex,
	float2* oldPos,           
	float2* oldVel,           
	uint    numParticles)
{
	
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];    
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	
	if (index < numParticles)
	{
		hash = gridParticleHash[index];


		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	cg::sync(cta);

	if (index < numParticles)
	{
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

		
		uint sortedIndex = gridParticleIndex[index];
		float2 pos = oldPos[sortedIndex];
		float2 vel = oldVel[sortedIndex];

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
	return (v1.x * v2.x + v1.y * v2.y) / d1 / d2;
}

__device__
float2 calculateAcceleration(
	uint    index,
	float2  pos,
	float2  vel,
	float2* oldPos,
	float2* oldVel,
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
			if (neightbourGrid.x == -1) { neightbourGrid.x = params.gridSize.x - 1; offset.x -= params.width; }
			else if (neightbourGrid.x == params.gridSize.x) { neightbourGrid.x = 0; offset.x += params.width; }
			if (neightbourGrid.y == -1) { neightbourGrid.y = params.gridSize.y - 1; offset.y -= params.height; }
			else if (neightbourGrid.y == params.gridSize.y) { neightbourGrid.y = 0; offset.y += params.height; }

			uint gridHash = calcGridHash(neightbourGrid);

			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)          
			{
				uint endIndex = cellEnd[gridHash];

				for (uint j = startIndex; j < endIndex; j++)
				{
					if (j == index) continue;              

					float2 pos2 = make_float2(oldPos[j].x, oldPos[j].y) + offset;
					float2 toVec = pos2 - pos;
					float dist = length(toVec);
					float r = params.particleRadius;

					if (r < dist) continue;

					float cosVision = cosf(params.visionAngle * CUDART_PI_F / 180.0);
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
		sep_acc = setLength(sep_acc, params.maxSpeed);
		sep_acc = setLength(sep_acc, params.maxAcceleration) * params.separationFactor;
	}

	float2 alg_acc = make_float2(0, 0);
	if (alg_n)
	{
		alg_acc = alg_sum / alg_n;
		alg_acc = setLength(alg_acc, params.maxSpeed);
		alg_acc = setLength(alg_acc, params.maxAcceleration) * params.alignmentFactor;
	}


	float2 coh_acc = make_float2(0, 0);
	if (coh_n)
	{
		coh_acc = coh_sum / coh_n - pos;
		coh_acc = setLength(coh_acc, params.maxSpeed);
		coh_acc = setLength(coh_acc, params.maxAcceleration) * params.cohesionFactor;
	}

	float2 repl_acc = make_float2(0, 0);
	float2 toMouse = params.mousePos - pos;

	float cosVision = cosf(params.visionAngle * CUDART_PI_F / 180.0);
	float cosToVec = angleBetween(toMouse, vel);

	if (cosVision <= cosToVec && length(toMouse) < params.mouseRadius)
	{
		repl_acc = setLength(-toMouse, params.maxSpeed);
		repl_acc = setLength(repl_acc, params.maxAcceleration) * params.mouseFactor;
	}

	return sep_acc + alg_acc + coh_acc + repl_acc;
}

__global__
void run(float2* newVel,               
	float2* oldPos,               
	float2* oldVel,               
	uint* gridParticleIndex,    
	uint* cellStart,
	uint* cellEnd,
	uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	
	float2 pos = make_float2(oldPos[index].x, oldPos[index].y);
	float2 vel = make_float2(oldVel[index].x, oldVel[index].y);

	
	float2 acc = calculateAcceleration(index, pos, vel, oldPos, oldVel, cellStart, cellEnd);

	
	uint originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = setLength(vel + acc, params.maxSpeed);
}

#endif
