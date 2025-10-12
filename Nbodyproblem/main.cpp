#include <iostream>
#include <vector>
#include <cmath>
using vec3 = double[3];

constexpr const double G = 6.67 * 1e-11;
constexpr const vec3 zerovec{0, 0, 0};
struct body{
    vec3 r;
    vec3 v;
    vec3 a;
    double mass;
}


struct Nbody{
    std::vector<double> bodies;

    double calculate_forces(){
        for(int i = 0, int n = bodies.size(); i < n; ++i){
            bodies[i] = zerovec;
            for(int j = 0; j < n; ++j){
                if(i == j) continue;

                double r2 = (bodies[i].r[0] - bodies[j].r[0]) * (bodies[i].r[0] - bodies[j].r[0]) 
                            + (bodies[i].r[1] - bodies[j].r[1]) * (bodies[i].r[1] - bodies[j].r[1])
                            + (bodies[i].r[2] - bodies[j].r[2]) * (bodies[i].r[2] - bodies[j].r[2]);

                double r = std::sqrt(r2);
                double tmp = std::max(r * r2, 1e-9);

                for(int k = 0; k < 3; ++k)
                    bodies[i].a[k] += G * (bodies[i].r[k] - bodies[j].r[k]) * bodies[j].m / tmp;

            }
        }
    }
}




int main(){

    
    

    return 0;
}