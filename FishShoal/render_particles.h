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

#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer
{
    public:
        ParticleRenderer();
        ~ParticleRenderer();

        void setPositions(float *pos, int numParticles);
        void setVertexBuffer(unsigned int vbo, int numParticles);
        void setColorBuffer(unsigned int vbo)
        {
            m_colorVBO = vbo;
        }
        void display();
        void displayGrid();

        void setPointSize(float size)
        {
            m_pointSize = size;
        }
        void setParticleRadius(float r)
        {
            m_particleRadius = r;
        }
        void setWindowSize(int w, int h)
        {
            m_window_w = w;
            m_window_h = h;
        }

    protected: // methods
        void _initGL();
        void _drawPoints();
        GLuint _compileProgram(const char *vsource, const char *fsource);

    protected: // data
        float *m_pos;
        int m_numParticles;

        float m_pointSize;
        float m_particleRadius;
        int m_window_w, m_window_h;

        GLuint m_program;

        GLuint m_vbo;
        GLuint m_colorVBO;
};

#endif //__ RENDER_PARTICLES__
