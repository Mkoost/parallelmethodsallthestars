#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <ostream>
#include <string>
#include <random>
#include <vector>


using real = double;
using vec3 = real[3];

constexpr int BS = 256;


struct Body {
    real mass;
    vec3 r;
    vec3 v;
    vec3 a;
};

struct BodyTMP {
    real mass;
    vec3 r;
    vec3 a;
};


struct RungeData {
    vec3 r;
    vec3 v;
    vec3 vnew;
    vec3 a;
};

struct NBodyNBINIT_DDEVC_t {
    Body* devBodies;
    RungeData* devYn;
    int size;
    real tau;
};

__constant__ real G = 6.67e-11;

__constant__ NBodyNBINIT_DDEVC_t NBINIT_DDEVC;
NBodyNBINIT_DDEVC_t NBINIT_DHOST{ nullptr, nullptr, 0, 0. };



__host__ void NBodyInit(int size, real tau) {

    NBINIT_DHOST.size = size;
    NBINIT_DHOST.tau = tau;

    cudaMalloc((void**)(&NBINIT_DHOST.devBodies), size * sizeof(Body));
    cudaMalloc((void**)(&NBINIT_DHOST.devYn), size * sizeof(RungeData));

    cudaMemcpyToSymbol(NBINIT_DDEVC, &NBINIT_DHOST, sizeof(NBodyNBINIT_DDEVC_t));

}

__host__ void NBodyFinilize() {
    cudaFree(NBINIT_DHOST.devBodies);
    cudaFree(NBINIT_DHOST.devYn);

    NBINIT_DHOST.devBodies = nullptr;
    NBINIT_DHOST.devYn = nullptr;
    NBINIT_DHOST.size = 0;
    NBINIT_DHOST.tau = 0;

    cudaMemcpyToSymbol(&NBINIT_DDEVC, &NBINIT_DHOST, sizeof(NBodyNBINIT_DDEVC_t));
}


__global__
void CalculateAccelerationCUDA() {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ BodyTMP bodies[2 * BS];

    if (idx >= NBINIT_DDEVC.size) return;
    //printf("bl: %d, idth: %d, sz: %d\n", blockIdx.x, threadIdx.x, NBINIT_DDEVC.size);
    

    bodies[threadIdx.x].mass = NBINIT_DDEVC.devBodies[idx].mass;
    for (int k = 0; k < 3; ++k)
        bodies[threadIdx.x].r[k] = NBINIT_DDEVC.devBodies[idx].r[k];

    for (int k = 0; k < 3; ++k)
        bodies[threadIdx.x].a[k] = 0;

    __syncthreads();
    
    for (int i = 0, n = NBINIT_DDEVC.size; i < n; i += BS) {
      if(i + threadIdx.x >= NBINIT_DDEVC.size) return;
        
        bodies[BS + threadIdx.x].mass = NBINIT_DDEVC.devBodies[i + threadIdx.x].mass;

        for (int k = 0; k < 3; ++k)
            bodies[BS + threadIdx.x].r[k] = NBINIT_DDEVC.devBodies[i + threadIdx.x].r[k];

        __syncthreads();

        for(int j = 0; j < BS; ++j)
        {
            int jj =  j;
            if(i + jj >= NBINIT_DDEVC.size) continue;
            jj += BS;

            vec3 r;
            

            for (int k = 0; k < 3; ++k)
              r[k] = bodies[threadIdx.x].r[k] - bodies[jj].r[k];

            real r3 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            r3 *= sqrt(r3);
            r3 = max(r3, 1e-10);

            for (int k = 0; k < 3; ++k)
                bodies[threadIdx.x].a[k] -=  (bodies[jj].mass) * r[k] / r3;
            
        }
        __syncthreads();
    }

    for (int k = 0; k < 3; ++k)
        NBINIT_DDEVC.devBodies[idx].a[k] = G * bodies[threadIdx.x].a[k];
}


__global__
void RK4step1CUDA() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= NBINIT_DDEVC.size) return;

    for (int k = 0; k < 3; ++k)
    {
        NBINIT_DDEVC.devYn[idx].r[k] = NBINIT_DDEVC.devBodies[idx].r[k];
        NBINIT_DDEVC.devYn[idx].v[k] = NBINIT_DDEVC.devBodies[idx].v[k];
    }

    for (int k = 0; k < 3; ++k)
    {
        NBINIT_DDEVC.devYn[idx].vnew[k] = NBINIT_DDEVC.devBodies[idx].v[k];
        NBINIT_DDEVC.devYn[idx].a[k] =  NBINIT_DDEVC.devBodies[idx].a[k];
    }

    for (int k = 0; k < 3; ++k)
    {
        NBINIT_DDEVC.devBodies[idx].v[k] += 0.5 * NBINIT_DDEVC.tau * NBINIT_DDEVC.devBodies[idx].a[k];
        NBINIT_DDEVC.devBodies[idx].r[k] += 0.5 * NBINIT_DDEVC.tau * NBINIT_DDEVC.devBodies[idx].v[k];
    }
}

__global__
void RK4step2CUDA() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (idx >= NBINIT_DDEVC.size) return;

    for (int k = 0; k < 3; ++k) {
        NBINIT_DDEVC.devYn[idx].a[k] += 2. * NBINIT_DDEVC.devBodies[idx].a[k];
        NBINIT_DDEVC.devYn[idx].vnew[k] += 2 * NBINIT_DDEVC.devBodies[idx].v[k];
    }

    for (int k = 0; k < 3; ++k) {
        NBINIT_DDEVC.devBodies[idx].r[k] = NBINIT_DDEVC.devYn[idx].r[k] + 0.5 * NBINIT_DDEVC.tau * NBINIT_DDEVC.devBodies[idx].v[k];
        NBINIT_DDEVC.devBodies[idx].v[k] = NBINIT_DDEVC.devYn[idx].v[k] + 0.5 * NBINIT_DDEVC.tau * NBINIT_DDEVC.devBodies[idx].a[k];
    }
}

__global__
void RK4step3CUDA() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (idx >= NBINIT_DDEVC.size) return;

    for (int k = 0; k < 3; ++k) {
        NBINIT_DDEVC.devYn[idx].a[k] += 2. * NBINIT_DDEVC.devBodies[idx].a[k];
        NBINIT_DDEVC.devYn[idx].vnew[k] += 2 * NBINIT_DDEVC.devBodies[idx].v[k];
    }

    for (int k = 0; k < 3; ++k) {
        NBINIT_DDEVC.devBodies[idx].r[k] = NBINIT_DDEVC.devYn[idx].r[k] + NBINIT_DDEVC.tau * NBINIT_DDEVC.devBodies[idx].v[k];
        NBINIT_DDEVC.devBodies[idx].v[k] = NBINIT_DDEVC.devYn[idx].v[k] + NBINIT_DDEVC.tau * NBINIT_DDEVC.devBodies[idx].a[k];
    }
}

__global__
void RK4step4CUDA() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= NBINIT_DDEVC.size) return;

    for (int k = 0; k < 3; ++k) {
        NBINIT_DDEVC.devBodies[idx].r[k] =
            NBINIT_DDEVC.devYn[idx].r[k] +
            NBINIT_DDEVC.tau * (NBINIT_DDEVC.devYn[idx].vnew[k] + NBINIT_DDEVC.devBodies[idx].v[k]) / 6.;
        NBINIT_DDEVC.devBodies[idx].v[k] =
            NBINIT_DDEVC.devYn[idx].v[k] + NBINIT_DDEVC.tau * (NBINIT_DDEVC.devYn[idx].a[k] + NBINIT_DDEVC.devBodies[idx].a[k]) / 6.;
    }
}

__host__
void input_from_file(const std::string &src, Body* &bodies, int &size) {
    std::ifstream fin(src);

    fin >> size;
    bodies = new Body[size];
    Body tmp_bod;
    for (int i = 0; i < size; ++i) {
      fin >> tmp_bod.mass >> tmp_bod.r[0] >> tmp_bod.r[1] >> tmp_bod.r[2] >>
          tmp_bod.v[0] >> tmp_bod.v[1] >> tmp_bod.v[2];
      bodies[i] = tmp_bod;
    }

    fin.close();
}


template<class U>
__host__ void output_to_file(const U& src, Body* bodies, int size, real t) {
  std::ofstream outFile(src, std::ios::app);

  for (size_t i = 0; i < size; ++i)
      outFile << t << " " << std::setprecision(14) << bodies[i].r[0] << " " << bodies[i].r[1] << std::endl;

  outFile.close();

}


__host__
void nbody_step(int num_step, int size, int block_size=BS) {
    int grid = (size + BS - 1) / BS;
    for (int step = 0; step < num_step; step++) {
        CalculateAccelerationCUDA<<<grid, BS>>>();

        RK4step1CUDA<<<grid, BS>>>();

        CalculateAccelerationCUDA<<<grid, BS>>>();

        RK4step2CUDA<<<grid, BS>>>();

        CalculateAccelerationCUDA<<<grid, BS >>>();

        RK4step3CUDA<<<grid, block_size>>>();

        CalculateAccelerationCUDA<<<grid, BS >>>();
        
        RK4step4CUDA<<<grid, BS >>>();
        
    }
}

__host__
void start_program(int argc, char** argv) {
    int size = 1024;
    int block = BS;
    real tau = 0.01;
    real T = 20;

    

    Body* bodies;

    
    input_from_file(argv[1], bodies, size);
    

    NBodyInit(size, tau);


    cudaMemcpy(NBINIT_DHOST.devBodies, bodies, sizeof(Body) * size, cudaMemcpyHostToDevice);
    
    nbody_step(T / tau, size, block);

    cudaMemcpy(bodies, NBINIT_DHOST.devBodies, sizeof(Body) * size, cudaMemcpyDeviceToHost);

    output_to_file(argv[2], bodies, size, T);

    delete[] bodies;

    NBodyFinilize();
}

int main(int argc, char** argv) {
    if (argc != 3) {std::cout << "It must be 3 arguments!" << std::endl; return 0;}

    start_program(argc, argv);

    return 0;
}