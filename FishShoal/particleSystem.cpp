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

 // OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint2 gridSize) :
	m_bInitialized(false),
	m_numParticles(numParticles),
	m_hPos(0),
	m_hVel(0),
	m_dPos(0),
	m_dVel(0),
	m_gridSize(gridSize),
	m_timer(NULL)
{
	m_numGridCells = m_gridSize.x * m_gridSize.y;

	m_gridSortBits = 18;    // increase this for larger grids

	// set simulation parameters
	params.gridSize = m_gridSize;
	params.numCells = m_numGridCells;
	params.numBodies = m_numParticles;

	params.particleRadius = 1.0f/64;

	//use for mouse
	params.mousePos = make_float2(-1.2f, -0.8f);
	params.mouseRadius = 0.2f;

	//params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	params.worldOrigin = make_float2(-100, -100);
	float cellSize = params.particleRadius * 2.0f;  // cell size equal to particle diameter
	params.cellSize = make_float2(cellSize, cellSize);

	params.separationFactor = 1.f;
	params.separationRadius = 1.f;
	params.alignmentFactor = 1.f;
	params.alignmentRadius = 1.f;
	params.cohesionFactor = 1.f;
	params.cohesionRadius = 1.f;
	params.visionAngle = 180.f;
	params.mouseFactor = 10.f;
	params.mouseRadius = 1.f;

	_initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
	_finalize();
	m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

inline float lerp(float a, float b, float t)
{
	return a + t * (b - a);
}

// create a color ramp
void colorRamp(float t, float* r)
{
	const int ncolors = 7;
	float c[ncolors][3] =
	{
		{ 1.0, 0.0, 0.0, },
		{ 1.0, 0.5, 0.0, },
		{ 1.0, 1.0, 0.0, },
		{ 0.0, 1.0, 0.0, },
		{ 0.0, 1.0, 1.0, },
		{ 0.0, 0.0, 1.0, },
		{ 1.0, 0.0, 1.0, },
	};
	t = t * (ncolors - 1);
	int i = (int)t;
	float u = t - floor(t);
	r[0] = lerp(c[i][0], c[i + 1][0], u);
	r[1] = lerp(c[i][1], c[i + 1][1], u);
	r[2] = lerp(c[i][2], c[i + 1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
	assert(!m_bInitialized);

	m_numParticles = numParticles;

	// allocate host storage
	m_hPos = new float[m_numParticles * 4];
	m_hVel = new float[m_numParticles * 4];
	memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

	// allocate GPU data
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;

	m_posVbo = createVBO(memSize);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);


	allocateArray((void**)&m_dVel, memSize);

	allocateArray((void**)&m_dSortedPos, memSize);
	allocateArray((void**)&m_dSortedVel, memSize);

	allocateArray((void**)&m_dGridParticleHash, m_numParticles * sizeof(uint));
	allocateArray((void**)&m_dGridParticleIndex, m_numParticles * sizeof(uint));

	allocateArray((void**)&m_dCellStart, m_numGridCells * sizeof(uint));
	allocateArray((void**)&m_dCellEnd, m_numGridCells * sizeof(uint));


	m_colorVBO = createVBO(m_numParticles * 4 * sizeof(float));
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

	// fill color buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
	float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float* ptr = data;

	for (uint i = 0; i < m_numParticles; i++)
	{
		float t = i / (float)m_numParticles;
#if 0
		* ptr++ = rand() / (float)RAND_MAX;
		*ptr++ = rand() / (float)RAND_MAX;
		*ptr++ = rand() / (float)RAND_MAX;
#else
		colorRamp(t, ptr);
		ptr += 3;
#endif
		* ptr++ = 1.0f;
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);


	sdkCreateTimer(&m_timer);

	setParameters(&params);

	m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
	assert(m_bInitialized);

	delete[] m_hPos;
	delete[] m_hVel;
	delete[] m_hCellStart;
	delete[] m_hCellEnd;

	freeArray(m_dVel);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);

	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);

	unregisterGLBufferObject(m_cuda_colorvbo_resource);
	unregisterGLBufferObject(m_cuda_posvbo_resource);
	glDeleteBuffers(1, (const GLuint*)&m_posVbo);
	glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
	assert(m_bInitialized);

	float* dPos;

	dPos = (float*)mapGLBufferObject(&m_cuda_posvbo_resource);

	// update constants
	setParameters(&params);

	// integrate
	integrateSystem(
		dPos,
		m_dVel,
		deltaTime,
		m_numParticles);

	// calculate grid hash
	calcHash(
		m_dGridParticleHash,
		m_dGridParticleIndex,
		dPos,
		m_numParticles);

	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(
		m_dCellStart,
		m_dCellEnd,
		m_dSortedPos,
		m_dSortedVel,
		m_dGridParticleHash,
		m_dGridParticleIndex,
		dPos,
		m_dVel,
		m_numParticles,
		m_numGridCells);

	// process collisions
	collide(
		m_dVel,
		m_dSortedPos,
		m_dSortedVel,
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_numGridCells);

	// note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	unmapGLBufferObject(m_cuda_posvbo_resource);
}

float*
ParticleSystem::getArray(ParticleArray array)
{
	assert(m_bInitialized);

	float* hdata = 0;
	float* ddata = 0;
	struct cudaGraphicsResource* cuda_vbo_resource = 0;

	switch (array)
	{
	default:
	case POSITION:
		hdata = m_hPos;
		ddata = m_dPos;
		cuda_vbo_resource = m_cuda_posvbo_resource;
		break;

	case VELOCITY:
		hdata = m_hVel;
		ddata = m_dVel;
		break;
	}

	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
	return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float* data, int start, int count)
{
	assert(m_bInitialized);

	switch (array)
	{
	default:
	case POSITION:
	{
		unregisterGLBufferObject(m_cuda_posvbo_resource);
		glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
		glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
	}
	break;

	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	}
}

inline float frand()
{
	return rand() / (float)RAND_MAX;
}

void
ParticleSystem::initGrid(uint* size, uint numParticles)
{
	srand(1973);
	int p = 0, v = 0;

	for (uint i = 0; i < m_numParticles; i++)
	{
		float px = frand();
		float py = frand();
		m_hPos[p++] = 200 * (px - 0.5f);
		m_hPos[p++] = 200 * (py - 0.5f);
		m_hPos[p++] = 0;
		m_hPos[p++] = 1.0f; // radius

		m_hVel[v++] = 0.0f;
		m_hVel[v++] = 0.0f;
		m_hVel[v++] = 0.0f;
		m_hVel[v++] = 0.0f;
	}
}

void
ParticleSystem::reset()
{
	uint s = (int)ceilf(sqrtf(m_numParticles));
	uint gridSize[3];
	gridSize[0] = gridSize[1] = s;
	gridSize[2] = 1;
	initGrid(gridSize, m_numParticles);

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
}
