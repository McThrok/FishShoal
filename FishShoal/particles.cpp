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
	 Particle system example with collisions using uniform grid

	 CUDA 2.1 SDK release 12/2008
	 - removed atomic grid method, some optimization, added demo mode.

	 CUDA 2.2 release 3/2009
	 - replaced sort function with latest radix sort, now disables v-sync.
	 - added support for automated testing and comparison to a reference value.
 */

 // OpenGL Graphics includes
#include <helper_gl.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   1600;

const uint width = 640, height = 480;

// view params
int ox, oy;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

uint numParticles = 0;
uint3 gridSize;
int numIterations = 0; // run until exit

// simulation parameters
float timestep = 0.5f;
float damping = 1.0f;
int iterations = 1;
int ballr = 10;

float collideSpring = 0.5f;;
float collideDamping = 0.02f;;
float collideShear = 0.1f;
float collideAttraction = 0.0f;

float separationFactor = 1.f;
float separationRadius = 1.f;
float alignmentFactor = 1.f;
float alignmentRadius = 1.f;
float cohesionFactor = 1.f;
float cohesionRadius = 1.f;
float angle = 180.f;

float mouseFactor = 10.f;
float mouseRadius = 1.f;

ParticleSystem* psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface* timer = NULL;
ParticleRenderer* renderer = 0;

float modelView[16];

ParamListGL* params;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
const char* sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char** argv);
extern "C" void cudaGLInit(int argc, char** argv);
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);

// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize)
{
	psystem = new ParticleSystem(numParticles, gridSize);
	psystem->reset();

	renderer = new ParticleRenderer;
	renderer->setParticleRadius(psystem->m_params.particleRadius);
	renderer->setColorBuffer(psystem->getColorBuffer());

	sdkCreateTimer(&timer);
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (psystem)
	{
		delete psystem;
	}
	return;
}

// initialize OpenGL
void initGL(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA Particles");

	if (!isGLVersionSupported(2, 0) ||
		!areGLExtensionsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
	{
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(EXIT_FAILURE);
	}

#if defined (WIN32)

	if (wglewIsSupported("WGL_EXT_swap_control"))
	{
		// disable vertical sync
		wglSwapIntervalEXT(0);
	}

#endif

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.1, 0.1, 0.1, 1.0);

	glutReportErrors();
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "CUDA Particles (%d particles): %3.1f fps", numParticles, ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

void display()
{
	sdkStartTimer(&timer);

	// update the simulation
	psystem->setIterations(iterations);
	psystem->m_params.globalDamping = damping;
	psystem->m_params.spring = collideSpring;
	psystem->m_params.damping = collideDamping;
	psystem->m_params.shear = collideShear;
	psystem->m_params.attraction = collideAttraction;

	psystem->update(timestep);

	if (renderer)
	{
		renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
	}

	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// view transform
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0, 0, -3);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

	// cube
	glColor3f(1.0, 1.0, 1.0);
	glutWireCube(2.0);

	if (renderer)
	{
		renderer->display(displayMode);
	}

	//sliders
	glDisable(GL_DEPTH_TEST);
	glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
	glEnable(GL_BLEND);
	params->Render(0, 5);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	sdkStopTimer(&timer);

	glutSwapBuffers();
	glutReportErrors();

	computeFPS();
}

inline float frand()
{
	return rand() / (float)RAND_MAX;
}

void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(60.0, (float)w / (float)h, 0.1, 100.0);
	//gluOrtho2D(-1, 0, -1, 0);
	//gluOrtho2D(-1000, w, -1000, h);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	if (renderer)
	{
		renderer->setWindowSize(w, h);
		renderer->setFOV(60.0);
	}
}

void mouse(int button, int state, int x, int y)
{
	ox = x;
	oy = y;

	//sliders
	if (params->Mouse(x, y, button, state))
	{
		glutPostRedisplay();
		return;
	}

	glutPostRedisplay();
}

// transform vector by matrix
void xform(float* v, float* r, GLfloat* m)
{
	r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + m[12];
	r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + m[13];
	r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float* v, float* r, GLfloat* m)
{
	r[0] = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
	r[1] = v[0] * m[4] + v[1] * m[5] + v[2] * m[6];
	r[2] = v[0] * m[8] + v[1] * m[9] + v[2] * m[10];
}

void ixformPoint(float* v, float* r, GLfloat* m)
{
	float x[4];
	x[0] = v[0] - m[12];
	x[1] = v[1] - m[13];
	x[2] = v[2] - m[14];
	x[3] = 1.0f;
	ixform(x, r, m);
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	//sliders
	if (params->Motion(x, y))
	{
		ox = x;
		oy = y;
		glutPostRedisplay();
		return;
	}

	ox = x;
	oy = y;

	glutPostRedisplay();
}

void special(int k, int x, int y)
{
	params->Special(k, x, y);
}

void idle(void)
{
	glutPostRedisplay();
}

void initParams()
{
	// create a new parameter list
	params = new ParamListGL("misc");
	params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.01f, &timestep));
	params->AddParam(new Param<float>("damping", damping, 0.0f, 1.0f, 0.001f, &damping));
	params->AddParam(new Param<int>("ball radius", ballr, 1, 20, 1, &ballr));

	params->AddParam(new Param<float>("collide spring", collideSpring, 0.0f, 1.0f, 0.001f, &collideSpring));
	params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 0.1f, 0.001f, &collideDamping));
	params->AddParam(new Param<float>("collide shear", collideShear, 0.0f, 0.1f, 0.001f, &collideShear));
	params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));
}

int main(int argc, char** argv)
{
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s Starting...\n\n", sSDKsample);

	numParticles = NUM_PARTICLES;
	uint gridDim = GRID_SIZE;
	numIterations = 0;

	gridSize.x = gridSize.y = gridSize.z = gridDim;
	printf("grid: %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.x * gridSize.y );
	printf("particles: %d\n", numParticles);

	initGL(&argc, argv);
	cudaInit(argc, argv);

	initParticleSystem(numParticles, gridSize);
	initParams();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutSpecialFunc(special);
	glutIdleFunc(idle);

	glutCloseFunc(cleanup);

	glutMainLoop();

	if (psystem)
	{
		delete psystem;
	}

	exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
