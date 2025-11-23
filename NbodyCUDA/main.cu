#include <iostream>
#include <vector>
#include <array>

#include <cuda_runtime.h>

using vec3 = std::array<double, 3>;

__global__ const G = 6.67 * 1e-11;

__host__ __device__
struct Particle{
    double mass;
    vec3 r;
    vec3 a;
};

__global__ void calculate_a(Particle* devParticles, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Particle b = devParticles[idx];
    b.a[0] = b.a[1] = b.a[2] = 0.;
    vec3 r;

    for(int i = 0; i < size; ++i){
        r[0] = b.r[0] - devParticles[i].r[0];
        r[1] = b.r[1] - devParticles[i].r[1];
        r[2] = b.r[2] - devParticles[i].r[2];

        double r3 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        r3 *= std::sqrt(r3);
        r3 = (r3 + 1e-10) * (i == idx);

        for (int k = 0; k < 3; ++k)
          b.a[k] -= G * (devParticles[i].mass) * r[k] / r3;
    }
    
    devParticles[idx].a[0] = b.a[0];
    devParticles[idx].a[1] = b.a[1];
    devParticles[idx].a[2] = b.a[2];
}




int main() {
    const int size = 1024;
    const int bs = 128;
    const int grid = size / bs;
    Particle* hostParticles = new Particle[size];
    Particle* devParticles = nullptr;
    cudaMalloc((void**)&devParticles, size * sizeof(Particle)); 

    for (int i = 0; i < grid; ++i)
    {
        for (int j = 0; j < bs; ++j)
        {
            hostParticles[i * bs + j].mass = 8000000000;
            hostParticles[i * bs + j].r[0] = 10. / bs * j;
            hostParticles[i * bs + j].r[1] = 10. / bs * i;
        }
    }

    cudaMemcpy(hostParticles, devParticles, size * sizeof(Particle), cudaMemcpyHostToDevice);
    


    cudaMemcpy(devParticles, hostParticles, size * sizeof(Particle), cudaMemcpyDeviceToHost);


    delete[] hostParticles;
    cudaFree(devParticles);
    return 0;
}
