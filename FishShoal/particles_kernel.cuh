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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
	float particleRadius;

	uint2 gridSize;
	//uint numCells;
	float2 worldOrigin;
	float2 cellSize;

	uint numBodies;
	uint maxParticlesPerCell;

	float separationFactor;
	float separationRadius;
	float alignmentFactor;
	float alignmentRadius;
	float cohesionFactor;
	float cohesionRadius;
	float visionAngle;

	float mouseFactor;
	float mouseRadius;
	float2 mousePos;

	float maxSpeed;
	float maxAcceleration;

	float squareSize;


	float2 test;
};

#endif
