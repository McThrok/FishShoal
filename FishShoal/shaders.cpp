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

#define STRINGIFY(A) #A

const char *vertexShader = STRINGIFY(
                               uniform float pointRadius;  
                               uniform float pointScale;   
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    
    gl_PointSize = pointRadius * pointScale;

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xy,0, 1.0);

    gl_FrontColor = gl_Color;
}
                           );

const char *spherePixelShader = STRINGIFY(
                                    void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   

    N.z = sqrt(1.0-mag);

    
    float diffuse = max(0.0, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;
}
                                );
