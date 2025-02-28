
 //compute_30, sm_30
 //compute_35, sm_35
 //compute_37, sm_37
 //compute_50, sm_50
 //compute_52, sm_52
 //compute_60, sm_60
 //compute_61, sm_61
 //compute_70, sm_70
 //compute_75, sm_75
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

//#define GRID_SIZE 16
#define RADIUS 30
#define NUM_PARTICLES 50000;

uint width = 1800, height = 1000;

// view params
int ox, oy;

//uint numParticles = 0;

// simulation parameters
float timestep = 1.f;

float separationFactor = 1.f;
float alignmentFactor = 0.75f;
float cohesionFactor = 0.5f;
float separationRadius = 0.15f;
float alignmentRadius = 0.75f;
float cohesionRadius = 1.f;
float visionAngle = 180.f;
float mouseFactor = 20.f;
float mouseRadius = 40.f;

float maxSpeed = 1.6f;
float maxAcceleration = 0.4f;

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
void initParticleSystem(int numParticles, uint2 gridSize)
{
	psystem = new ParticleSystem(numParticles, gridSize, width, height, RADIUS);
	psystem->reset();

	renderer = new ParticleRenderer;
	renderer->setParticleRadius(psystem->params.particleRadius);
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
		sprintf(fps, "CUDA Particles (%d particles): %3.1f fps", psystem->getNumParticles(), ifps);

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
	psystem->params.separationFactor = separationFactor;
	psystem->params.alignmentFactor = alignmentFactor;
	psystem->params.cohesionFactor = cohesionFactor;
	psystem->params.separationRadius = separationRadius;
	psystem->params.alignmentRadius = alignmentRadius;
	psystem->params.cohesionRadius = cohesionRadius;
	psystem->params.visionAngle = visionAngle;
	psystem->params.mouseFactor = mouseFactor;
	psystem->params.mouseRadius = mouseRadius;
	psystem->params.maxSpeed = maxSpeed;
	psystem->params.maxAcceleration = maxAcceleration;

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
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

	// mouse
	float2 mousePos = psystem->params.mousePos;
	glTranslatef(mousePos.x, mousePos.y, 0);
	glColor3f(1.0, 1.0, 1.0);
	glColor3f(1.0, 0, 0);
	glutWireCylinder(psystem->params.mouseRadius, 0, 100, 0);

	// radius
	float2 pos = psystem->getFirstPosition();
	glLoadIdentity();
	glTranslatef(pos.x, pos.y, 0);

	float r = psystem->params.particleRadius;
	glColor3f(1.0, 0, 0);
	glutWireCylinder(r * psystem->params.separationRadius, 0, 100, 0);
	glColor3f(0, 1.0, 0);
	glutWireCylinder(r * psystem->params.alignmentRadius, 0, 100, 0);
	glColor3f(0, 0, 1.0);
	glutWireCylinder(r * psystem->params.cohesionRadius, 0, 100, 0);


	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	if (renderer)
	{
		renderer->display();
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

void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(-GLdouble(psystem->params.width) / 2, GLdouble(psystem->params.width) / 2, -GLdouble(psystem->params.height) / 2, GLdouble(psystem->params.height) / 2, -1000, 1000);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	if (renderer)
	{
		renderer->setWindowSize(w, h);
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
	else
	{
		psystem->params.mousePos = make_float2(x - GLdouble(psystem->params.width) / 2, GLdouble(psystem->params.height) / 2 - y);
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
	params->AddParam(new Param<float>("time step", timestep, 0.0f, 5.0f, 0.01f, &timestep));

	params->AddParam(new Param<float>("separation factor", separationFactor, 0.0f, 1.0f, 0.001f, &separationFactor));
	params->AddParam(new Param<float>("alignment factor", alignmentFactor, 0.0f, 1.0f, 0.001f, &alignmentFactor));
	params->AddParam(new Param<float>("cohesion factor", cohesionFactor, 0.0f, 1.0f, 0.001f, &cohesionFactor));
	params->AddParam(new Param<float>("separation radius", separationRadius, 0.0f, 1.0f, 0.001f, &separationRadius));
	params->AddParam(new Param<float>("alignment radius", alignmentRadius, 0.0f, 1.0f, 0.001f, &alignmentRadius));
	params->AddParam(new Param<float>("cohesion radius", cohesionRadius, 0.0f, 1.0f, 0.001f, &cohesionRadius));

	params->AddParam(new Param<float>("mouse radius", mouseRadius, 0.0f, 100.0f, 0.001f, &mouseRadius));
	params->AddParam(new Param<float>("mouse factor", mouseFactor, 0.0f, 100.0f, 0.001f, &mouseFactor));

	params->AddParam(new Param<float>("vision angle", visionAngle, 0.0f, 180.0f, 1.f, &visionAngle));

	params->AddParam(new Param<float>("max speed", maxSpeed, 0.0f, 5.0f, 0.001f, &maxSpeed));
	params->AddParam(new Param<float>("max acceleration", maxAcceleration, 0.0f, 2.0f, 0.001f, &maxAcceleration));
}

int main(int argc, char** argv)
{
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s Starting...\n\n", sSDKsample);

	uint numParticles = NUM_PARTICLES;
	uint2 gridSize;
	gridSize.x = width / RADIUS;
	gridSize.y = height / RADIUS;
	width = gridSize.x * RADIUS;
	height = gridSize.y * RADIUS;

	printf("grid: %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.x * gridSize.y);
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
