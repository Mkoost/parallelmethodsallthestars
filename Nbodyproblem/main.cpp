#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using vec3 = double[3];
using vec2 = double[2];

constexpr const double G = 6.67 * 1e-11;
constexpr const vec3 zerovec{0, 0, 0};
struct body{
    vec3 r;
    vec3 v;
    vec3 a;
    double mass;

    body() : r{0,0,0}, v{0,0,0}, a{0,0,0}, mass(0.0) {}
};

struct Nbody{
    std::vector<body> bodies;
    size_t n = 0;

    // 4
    Nbody() {
        std::ifstream fin("4body.txt");
        
        fin >> n;
        body tmp_bod;

        for (int i = 0; i < n; ++i) {
            fin >> tmp_bod.mass
                >> tmp_bod.r[0] >> tmp_bod.r[1] >> tmp_bod.r[2]
                >> tmp_bod.v[0] >> tmp_bod.v[1] >> tmp_bod.v[2];
            bodies.emplace_back(tmp_bod);
        }

        fin.close();


        calculate_a(bodies);
     }

    void a_reset(std::vector<body> &bodies_) {
        for (int i = 0; i < n; ++i)
            for(int k = 0; k < 3; ++k)
                bodies_[i].a[k] = 0;
    }


    void calculate_a(std::vector<body> &bodies_){
        a_reset(bodies_);
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; ++j){
                if(i == j) continue;

                double r2 = (bodies_[i].r[0] - bodies_[j].r[0]) * (bodies_[i].r[0] - bodies_[j].r[0]) 
                            + (bodies_[i].r[1] - bodies_[j].r[1]) * (bodies_[i].r[1] - bodies_[j].r[1])
                            + (bodies_[i].r[2] - bodies_[j].r[2]) * (bodies_[i].r[2] - bodies_[j].r[2]);

                double r = std::sqrt(r2);
                double tmp = std::max(r2 * r, eps * eps * eps);

                for(int k = 0; k < 3; ++k)
                    bodies_[i].a[k] -= G * (bodies_[i].r[k] - bodies_[j].r[k]) * bodies_[j].mass / tmp;

            }
        }
    }

    void parallel_calculate(std::vector<body> &bodies_){
        a_reset(bodies_);
        #pragma omp parallel for num_threads(nthreads) if(nthreads < (n - i))
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; ++j){
                if(i == j) continue;

                double r2 = (bodies_[i].r[0] - bodies_[j].r[0]) * (bodies_[i].r[0] - bodies_[j].r[0]) 
                            + (bodies_[i].r[1] - bodies_[j].r[1]) * (bodies_[i].r[1] - bodies_[j].r[1])
                            + (bodies_[i].r[2] - bodies_[j].r[2]) * (bodies_[i].r[2] - bodies_[j].r[2]);

                double r = std::sqrt(r2);
                double tmp = std::max(r2 * r, eps * eps * eps);

                for(int k = 0; k < 3; ++k)
                    bodies_[i].a[k] -= G * (bodies_[i].r[k] - bodies_[j].r[k]) * bodies_[j].mass / tmp;

            }
        }
    }
    
    void logs_solve(double step = 0.00001, double end_t = 20) {
        for (int i = 1; i < 5; ++i) {
            std::ofstream file("nt" + std::to_string(i) + ".txt", std::ios::trunc);
            file.close();
        }

        std::vector<std::ofstream> files;
        for (int i = 1; i < 5; ++i) 
            files.push_back(std::ofstream("nt" + std::to_string(i) + ".txt", std::ios::app));

        vec2 k1, k2, k3, k4;
        double remainder;
        size_t tmp;

        // снимем копию с bodies
        std::vector<body> tmp_bodies;
        std::vector<body> y_n;
        // TODO: возвращать независимый вектор ускорений?
        double t = -step;

        for (size_t i = 0; i <= std::floor(end_t / step)+1; i++) {
            t += step;
            remainder = std::fmod(t, 0.1);
            if (i % 10000 == 0) {
                for (int i = 0; i < 4; ++i) {
                    files[i] << std::setprecision(6) << t << " ";
                    files[i] << std::setprecision(16) << bodies[i].r[0] << " " << bodies[i].r[1] << " " << bodies[i].r[2] << "\n";
                }

            }
            
            ///////////////////////////// rk4
            /////////////////////////////
            tmp_bodies = bodies;
            y_n = bodies;
            calculate_a(tmp_bodies);
            tmp = 0;
            for (body &bod : tmp_bodies) {
                for (int i = 0; i < 3 ; i++) {
                    // 0 <-> dr, 1 <-> dv
                    k1[0] = bod.v[i];
                    k1[1] = bod.a[i];

                    bodies[tmp].r[i] += step/6. * k1[0];
                    bodies[tmp].v[i] += step/6. * k1[1];

                    bod.r[i] = y_n[tmp].r[i] + step/2. * k1[0];
                    bod.v[i] = y_n[tmp].v[i] + step/2. * k1[1];
                }

                tmp++;
            }
            calculate_a(tmp_bodies);

            tmp = 0;
            for (body &bod : tmp_bodies) {
                for (int i = 0; i < 3 ; i++) {
                    // 0 <-> dr, 1 <-> dv
                    k2[0] = bod.v[i];
                    k2[1] = bod.a[i];

                    bodies[tmp].r[i] += step / 3. * k2[0];
                    bodies[tmp].v[i] += step / 3. * k2[1];

                    bod.r[i] = y_n[tmp].r[i] + step / 2. * k2[0];
                    bod.v[i] = y_n[tmp].v[i] + step / 2. * k2[1];
                }

                tmp++;
            }
            calculate_a(tmp_bodies);

            tmp = 0;
            for (body &bod : tmp_bodies) {
                for (int i = 0; i < 3 ; i++) {
                    // 0 <-> dr, 1 <-> dv
                    k3[0] = bod.v[i];
                    k3[1] = bod.a[i];

                    bodies[tmp].r[i] += step / 3. * k3[0];
                    bodies[tmp].v[i] += step / 3. * k3[1];
                    
                    bod.r[i] = y_n[tmp].r[i] + step * k3[0];
                    bod.v[i] = y_n[tmp].v[i] + step * k3[1];
                }

                tmp++;
            }
            calculate_a(tmp_bodies);

            tmp = 0;
            for (body &bod : tmp_bodies) {
                for (int i = 0; i < 3 ; i++) {
                    // 0 <-> dr, 1 <-> dv
                    k4[0] = bod.v[i];
                    k4[1] = bod.a[i];
                    bodies[tmp].r[i] += step / 6. * k4[0];
                    bodies[tmp].v[i] += step / 6. * k4[1];
                }

                tmp++;
            }
/*
            for (body &bod : bodies) {
                for (int i = 0; i < 3 ; i++) {
                    // 0 <-> dr, 1 <-> dv

                    k1[0] = bod.v[i];
                    k1[1] = bod.a[i];

                    k2[0] = bod.v[i] + step/2 * k1[0];
                    k2[1] = bod.a[i] + step/2 * k1[1];

                    k3[0] = bod.v[i] + step/2 * k2[0];
                    k3[1] = bod.a[i] + step/2 * k2[1];

                    k4[0] = bod.v[i] + step * k3[0];
                    k4[1] = bod.a[i] + step * k3[1];

                    bod.r[i] += step/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);                                        
                    bod.v[i] += step/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
                }
            }
*/
        }

        for (std::ofstream& file : files)
            file.close();
    }
};




int main(){
    Nbody N1;
    N1.logs_solve();

    std::cout << N1.bodies[0].r[0] << " " << N1.bodies[2].v[0] << "\n";

    return 0;
}