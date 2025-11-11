// mexhat.cpp — SDL2 + OpenGL ES 2.0 spinning Mexican-hat surface (z = sin(r)/r)
// No SDL_test_common.h dependency.
//
// Build (Ubuntu):
//   sudo apt-get update && sudo apt-get install -y libsdl2-dev libgles2-mesa-dev
//   g++ -O2 mexhat.cpp $(sdl2-config --cflags --libs) -lGLESv2 -o mexhat
//
// Run notes (UserLAnd/VNC):
//   - Start a VNC server (e.g. :1), then: export DISPLAY=:1
//   - If you see XDG warnings: export XDG_RUNTIME_DIR=/tmp/runtime-$USER; mkdir -p $XDG_RUNTIME_DIR
//   - SDL will use EGL over X11 (Mesa llvmpipe is fine)

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengles2.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static const char* VS_SRC = R"(#version 100
attribute vec3 aPos;
attribute vec3 aCol;
uniform mat4 uMVP;
varying vec3 vCol;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vCol = aCol;
}
)";

static const char* FS_SRC = R"(#version 100
precision mediump float;
varying vec3 vCol;
void main() {
    gl_FragColor = vec4(vCol, 1.0);
}
)";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei n=0; glGetShaderInfoLog(s, 1024, &n, log);
        fprintf(stderr,"Shader compile error:\n%.*s\n", n, log);
        exit(1);
    }
    return s;
}

static GLuint linkProgram(const char* vs, const char* fs) {
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glBindAttribLocation(p, 0, "aPos");
    glBindAttribLocation(p, 1, "aCol");
    glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei n=0; glGetProgramInfoLog(p, 1024, &n, log);
        fprintf(stderr,"Program link error:\n%.*s\n", n, log);
        exit(1);
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// simple column-major mat4 helpers
struct Mat4 {
    float m[16];
    static Mat4 identity() {
        Mat4 A{}; for(int i=0;i<16;++i) A.m[i]=(i%5==0)?1.f:0.f; return A;
    }
};
static Mat4 mul(const Mat4& A, const Mat4& B){
    Mat4 R{}; 
    for(int r=0;r<4;++r) for(int c=0;c<4;++c){
        R.m[c*4+r] = A.m[0*4+r]*B.m[c*4+0] + A.m[1*4+r]*B.m[c*4+1]
                   + A.m[2*4+r]*B.m[c*4+2] + A.m[3*4+r]*B.m[c*4+3];
    }
    return R;
}
static Mat4 perspective(float fovy, float aspect, float znear, float zfar){
    float f = 1.0f / tanf(fovy*0.5f);
    Mat4 P{}; 
    P.m[0]=f/aspect; P.m[5]=f; P.m[10]=(zfar+znear)/(znear-zfar); P.m[11]=-1.0f;
    P.m[14]=(2.0f*zfar*znear)/(znear-zfar);
    return P;
}
static Mat4 rotateY(float a){
    Mat4 R=Mat4::identity(); float c=cosf(a), s=sinf(a);
    R.m[0]=c; R.m[2]=s; R.m[8]=-s; R.m[10]=c; return R;
}
static Mat4 rotateX(float a){
    Mat4 R=Mat4::identity(); float c=cosf(a), s=sinf(a);
    R.m[5]=c; R.m[6]=s; R.m[9]=-s; R.m[10]=c; return R;
}
static Mat4 translate(float x,float y,float z){
    Mat4 T=Mat4::identity(); T.m[12]=x; T.m[13]=y; T.m[14]=z; return T;
}

struct Mesh {
    GLuint vbo=0, cbo=0, ibo=0;
    GLsizei indexCount=0;
};
static Mesh makeSombrero(int N=128, float radius=6.0f, float zscale=1.0f, float freq=1.0f) {
    if (N<3) N=3;
    const int V = N*N;
    std::vector<float> pos; pos.reserve(V*3);
    std::vector<float> col; col.reserve(V*3);
    const float xmin=-radius, xmax=radius, ymin=-radius, ymax=radius;

    // compute positions and z-range
    std::vector<float> zs; zs.reserve(V);
    float zmin=1e9f, zmax=-1e9f;
    for (int j=0;j<N;++j){
        float ty = float(j)/(N-1);
        float y = ymin + ty*(ymax-ymin);
        for (int i=0;i<N;++i){
            float tx = float(i)/(N-1);
            float x = xmin + tx*(xmax-xmin);
            float r = sqrtf(x*x + y*y);
            if (r < 1e-4f) r = 1e-4f;
            float z = zscale * (sinf(freq*r)/r);
            pos.push_back((x/radius)*1.5f);
            pos.push_back((y/radius)*1.5f);
            pos.push_back(z);
            zs.push_back(z);
            if (z<zmin) zmin=z; if (z>zmax) zmax=z;
        }
    }
    float range = (zmax - zmin); if (range < 1e-6f) range = 1.0f;

    // colors by height
    for (int v=0; v<V; ++v){
        float t = (zs[v]-zmin)/range; // 0..1
        float r,g,b;
        if (t < 0.25f) { float k=t/0.25f; r=0.0f; g=k;   b=1.0f; }
        else if (t < 0.50f) { float k=(t-0.25f)/0.25f; r=0.0f; g=1.0f; b=1.0f-k; }
        else if (t < 0.75f) { float k=(t-0.50f)/0.25f; r=k;   g=1.0f; b=0.0f; }
        else { float k=(t-0.75f)/0.25f; r=1.0f; g=1.0f-k; b=0.0f; }
        col.push_back(r); col.push_back(g); col.push_back(b);
    }

    // indices (GLushort, two tris per cell)
    std::vector<GLushort> idx; idx.reserve((N-1)*(N-1)*6);
    for (int j=0;j<N-1;++j){
        for (int i=0;i<N-1;++i){
            GLushort a = j*N + i;
            GLushort b = j*N + (i+1);
            GLushort c = (j+1)*N + i;
            GLushort d = (j+1)*N + (i+1);
            idx.push_back(a); idx.push_back(c); idx.push_back(b);
            idx.push_back(b); idx.push_back(c); idx.push_back(d);
        }
    }

    Mesh m{};
    glGenBuffers(1,&m.vbo); glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER, pos.size()*sizeof(float), pos.data(), GL_STATIC_DRAW);
    glGenBuffers(1,&m.cbo); glBindBuffer(GL_ARRAY_BUFFER,m.cbo);
    glBufferData(GL_ARRAY_BUFFER, col.size()*sizeof(float), col.data(), GL_STATIC_DRAW);
    glGenBuffers(1,&m.ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(GLushort), idx.data(), GL_STATIC_DRAW);
    m.indexCount = (GLsizei)idx.size();
    return m;
}

int main(int argc, char** argv){
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr,"SDL_Init: %s\n", SDL_GetError()); return 1;
    }

    // Request GLES 2.0 context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* win = SDL_CreateWindow(
        "Spinning Sombrero (SDL2 + GLES2)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        900, 700, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!win){ fprintf(stderr,"SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }

    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx){ fprintf(stderr,"SDL_GL_CreateContext: %s\n", SDL_GetError()); return 1; }
    SDL_GL_SetSwapInterval(1);

    GLuint prog = linkProgram(VS_SRC, FS_SRC);
    GLint locPos = 0; // bound via glBindAttribLocation
    GLint locCol = 1;
    GLint locMVP = glGetUniformLocation(prog, "uMVP");

    Mesh mesh = makeSombrero(128, 6.0f, 1.0f, 1.0f);

    int w=900,h=700;
    glViewport(0,0,w,h);
    glEnable(GL_DEPTH_TEST);

    bool quit=false;
    float ang=0.0f;

    while(!quit){
        SDL_Event e;
        while(SDL_PollEvent(&e)){
            if (e.type==SDL_QUIT) quit=true;
            if (e.type==SDL_WINDOWEVENT && e.window.event==SDL_WINDOWEVENT_SIZE_CHANGED){
                w=e.window.data1; h=e.window.data2;
                glViewport(0,0,w,h);
            }
        }

        glClearColor(0.02f,0.02f,0.03f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (h>0) ? (float)w/(float)h : 1.0f;
        Mat4 P = perspective(60.0f*(3.1415926f/180.0f), aspect, 0.1f, 50.0f);
        Mat4 V = translate(0.0f, 0.0f, -4.5f);
        Mat4 R = mul(rotateY(ang*0.9f), rotateX(ang*0.5f));
        Mat4 MVP = mul(P, mul(V, R));

        glUseProgram(prog);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, MVP.m);

        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glVertexAttribPointer(locPos, 3, GL_FLOAT, GL_FALSE, 0, (const void*)0);
        glEnableVertexAttribArray(locPos);

        glBindBuffer(GL_ARRAY_BUFFER, mesh.cbo);
        glVertexAttribPointer(locCol, 3, GL_FLOAT, GL_FALSE, 0, (const void*)0);
        glEnableVertexAttribArray(locCol);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ibo);
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_SHORT, 0);

        SDL_GL_SwapWindow(win);
        ang += 0.02f;
    }

    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}