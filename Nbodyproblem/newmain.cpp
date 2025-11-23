#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <ostream>
#include <string>
#include <random>
#include <vector>

using vec3 = double[3];
using vec2 = double[2];

constexpr const double G = 6.67e-11;
constexpr const vec3 zerovec{0, 0, 0};

struct NBodyProblem {
  struct Body {
    double mass;
    vec3 r;
    vec3 v;
    vec3 a;

    Body() : r{0., 0., 0.}, v{0., 0., 0.}, a{0., 0., 0.}, mass(0.) {}
  };

  std::vector<Body> bodies;
  NBodyProblem(int n_) : bodies(n_) {}

  void calculate_a() {
    int n = bodies.size();
#pragma omp for
    for (int i = 0; i < n; ++i) {
      Body b = bodies[i];
      b.a[0] = b.a[1] = b.a[2] = 0.;
      for (int j = 0; j < n; ++j) {
        if (i == j)
          continue;

        vec3 r;
        r[0] = b.r[0] - bodies[j].r[0];
        r[1] = b.r[1] - bodies[j].r[1];
        r[2] = b.r[2] - bodies[j].r[2];

        double r3 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        r3 *= std::sqrt(r3);
        r3 = std::max(r3, 1e-10);

        for (int k = 0; k < 3; ++k)
          b.a[k] -= (bodies[j].mass) * r[k] / r3;
      }

      for (int k = 0; k < 3; ++k)
        bodies[i].a[k] = G * b.a[k];
    }
  }
};

struct RK4Integrator {
  struct RKdata {
    vec3 r;
    vec3 v;
    vec3 a;
    vec3 vnew;
  };

  double tau;
  int n;

  std::vector<RKdata> yn;

  RK4Integrator(int n_, double tau_) : tau(tau_), yn(n_) {}

  void step(double num, NBodyProblem &nbodies) {

    int n = yn.size();
#pragma omp parallel shared(num, n, nbodies) \
    num_threads(omp_get_max_threads()) if (omp_get_max_threads() > 1)
    {
      for (int ii = 0; ii < num; ++ii) {

        nbodies.calculate_a();

#pragma omp for 
        for (int i = 0; i < n; ++i)
          for (int k = 0; k < 3; ++k) {
            yn[i].r[k] = nbodies.bodies[i].r[k];
            yn[i].v[k] = nbodies.bodies[i].v[k];
            yn[i].a[k] = nbodies.bodies[i].a[k];
            yn[i].vnew[k] = nbodies.bodies[i].v[k];
          }

#pragma omp for
        for (int i = 0; i < n; ++i) 
          for (int k = 0; k < 3; ++k) {
            nbodies.bodies[i].r[k] += 0.5 * tau * nbodies.bodies[i].v[k];
            nbodies.bodies[i].v[k] += 0.5 * tau * nbodies.bodies[i].a[k];
          }

        nbodies.calculate_a();

#pragma omp for
        for (int i = 0; i < n; ++i) {
          for (int k = 0; k < 3; ++k) {
            yn[i].a[k] += 2. * nbodies.bodies[i].a[k];
            yn[i].vnew[k] += 2 * nbodies.bodies[i].v[k];
          }

          for (int k = 0; k < 3; ++k) {
            nbodies.bodies[i].r[k] =
                yn[i].r[k] + 0.5 * tau * nbodies.bodies[i].v[k];
            nbodies.bodies[i].v[k] =
                yn[i].v[k] + 0.5 * tau * nbodies.bodies[i].a[k];
          }
        }

        nbodies.calculate_a();

#pragma omp for
        for (int i = 0; i < n; ++i) {
          for (int k = 0; k < 3; ++k) {
            yn[i].a[k] += 2. * nbodies.bodies[i].a[k];
            yn[i].vnew[k] += 2 * nbodies.bodies[i].v[k];
          }

          for (int k = 0; k < 3; ++k) {
            nbodies.bodies[i].r[k] = yn[i].r[k] + tau * nbodies.bodies[i].v[k];
            nbodies.bodies[i].v[k] = yn[i].v[k] + tau * nbodies.bodies[i].a[k];
          }
        }

        nbodies.calculate_a();
#pragma omp for
        for (int i = 0; i < n; ++i) {
          for (int k = 0; k < 3; ++k) {
            nbodies.bodies[i].r[k] =
                yn[i].r[k] +
                tau * (yn[i].vnew[k] + nbodies.bodies[i].v[k]) / 6.;
            nbodies.bodies[i].v[k] =
                yn[i].v[k] + tau * (yn[i].a[k] + nbodies.bodies[i].a[k]) / 6.;
          }
        }
      }
    }
  }
};


struct Interface {
  NBodyProblem nbodies;
  RK4Integrator integrator;
  int n;
  double t;
  Interface(int n_, double tau_)
      : nbodies(n_), integrator(n_, tau_), n(n_), t(0) {};

  Interface(const std::string &path, double tau_)
      : nbodies(0), integrator(0, tau_), n(0), t(0) {

    std::ifstream fin(path);

    fin >> n;
    NBodyProblem::Body tmp_bod;
    resize(n);

    for (int i = 0; i < n; ++i) {
      fin >> tmp_bod.mass >> tmp_bod.r[0] >> tmp_bod.r[1] >> tmp_bod.r[2] >>
          tmp_bod.v[0] >> tmp_bod.v[1] >> tmp_bod.v[2];
      nbodies.bodies[i] = tmp_bod;
    }

    fin.close();
  };

  void step(double num) {
    t += num * integrator.tau;
    integrator.step(num, nbodies);
  }

  void reserve(int num) {
    integrator.yn.reserve(num);
    nbodies.bodies.reserve(num);
  }

  void resize(int num) {
    integrator.yn.resize(num);
    nbodies.bodies.resize(num);
  }

  // WARNING: no third coordinate
  template <class U> void save_state(const U &path) {
    std::ofstream outFile(path, std::ios::app);

    for (size_t i = 0; i < n; ++i)
      outFile << t << " " << std::setprecision(14) << nbodies.bodies[i].r[0] << " " << nbodies.bodies[i].r[1] << std::endl;

    outFile.close();
  }
};



void time_test(double tau = 0.001, double t = 20.) {
  std::cout << "MY N-BODY PROBLEM" << std::endl;
  std::cout << "Threads number: " << omp_get_max_threads() << std::endl;
  Interface nbod("init.txt", tau);
  double t1 = -omp_get_wtime();
  nbod.step(int(t / tau));
  t1 += omp_get_wtime();
  std::cout << "Time: " << t1 << "s" << std::endl;
  nbod.save_state("res.txt");
}

double AbsErr() {
  std::ifstream file1("errtest.txt"); // файл с 4 телами
  std::ifstream bodies[4] = {
        std::ifstream("traj1_new.txt"),
        std::ifstream("traj2_new.txt"),
        std::ifstream("traj3_new.txt"),
        std::ifstream("traj4_new.txt")
    };

  double maxDiff = 0.;
  double tmp1 = 0., tmp2 = 0.;

  while (file1) {

    for (int i = 0; i < 4; ++i) {
      file1 >> tmp1;
      bodies[i] >> tmp2;

      // if (flag) std::cout << tmp1 << " " << tmp2 << std::endl;

      for (int j = 0; j < 2; ++j) {
        file1 >> tmp1;
        bodies[i] >> tmp2;
        maxDiff = std::max(maxDiff, std::fabs(tmp1 - tmp2));
      }
    }
  }
  

  return maxDiff;
}

void err_test(double tau = 0.01) {
  std::cout << "MY N-BODY PROBLEM" << std::endl;
  Interface nbod("4body.txt", tau);
  std::ofstream outFile("errtest.txt", std::ios::trunc);
  outFile.close();
  nbod.save_state("errtest.txt");

  for (int i = 0; i < 200; ++i) {
    nbod.step(int(0.1 / tau));
    nbod.save_state("errtest.txt");
  }

  std::cout << "MaxErr = " << AbsErr() << std::endl;
}

void generate_nbody_file(
    const std::string &filename,
    int n,
    double mass_min, double mass_max,
    double pos_min, double pos_max,
    double vel_min, double vel_max,
    unsigned long long seed = std::random_device{}()
) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> mass_d(mass_min, mass_max);
    std::uniform_real_distribution<double> pos_d(pos_min, pos_max);
    std::uniform_real_distribution<double> vel_d(vel_min, vel_max);

    std::ofstream fout(filename);
    if (!fout) return; // можно бросать исключение или вернуть ошибку по желанию

    fout << n << '\n' << std::fixed << std::setprecision(8);
    for (int i = 0; i < n; ++i) {
        double m  = mass_d(rng);
        double x  = pos_d(rng), y  = pos_d(rng), z  = pos_d(rng);
        double vx = vel_d(rng), vy = vel_d(rng), vz = vel_d(rng);
        fout << m << ' ' << x << ' ' << y << ' ' << z << ' '
             << vx << ' ' << vy << ' ' << vz << '\n';
    }
}

int main(int argc, char **argv) {
  // сгенерировать 100 тел: масса 1..10, позиции -100..100, скорости -5..5, seed=42
  generate_nbody_file("init.txt", 200, 1.0, 10.0, -100.0, 100.0, -5.0, 5.0, 42ULL);
  time_test();
}