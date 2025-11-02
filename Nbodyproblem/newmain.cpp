#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <string>
#include <vector>

using vec3 = double[3];
using vec2 = double[2];

constexpr const double G = 6.67 * 1e-11;
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
  int n;
  NBodyProblem(int n_) : bodies(n_), n(n_) {}

  void calculate_a() {
#pragma omp for
    for (int i = 0; i < n; ++i) {
      Body b = bodies[i];
      b.a[0] = b.a[1] = b.a[2] = 0.;

      for (int j = 0; j < n; ++i) {
        if (i == j)
          continue;

        vec3 r;
        r[0] = b.r[0] - bodies[j].r[0];
        r[1] = b.r[0] - bodies[j].r[0];
        r[2] = b.r[0] - bodies[j].r[0];

        double r3 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        r3 *= std::sqrt(r3);
        r3 = std::max(r3, 1e-10);

        for (int k = 0; k < 3; ++k)
          b.a[k] -= r[k] * bodies[j].mass / r3;
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

  RK4Integrator(int n_, double tau_) : tau(tau_), yn(n_), n(n_) {}

  void step(double num, NBodyProblem &nbodies) {
#pragma omp parallel num_threads(                                              \
        omp_get_max_threads()) if (omp_get_max_threads() > 1)
    {
      for (int ii = 0; ii < num; ++ii) {
        nbodies.calculate_a();

#pragma omp single
        for (int i = 0; i < n; ++i) {
          for (int k = 0; k < 3; ++k) {
            yn[i].r[k] = nbodies.bodies[i].r[k];
            yn[i].v[k] = nbodies.bodies[i].v[k];
            yn[i].a[k] = nbodies.bodies[i].a[k];
            yn[i].vnew[k] = nbodies.bodies[i].v[k];
          }

          for (int k = 0; k < 3; ++k) {
            nbodies.bodies[i].r[k] += 0.5 * tau * nbodies.bodies[i].v[k];
            nbodies.bodies[i].v[k] += 0.5 * tau * nbodies.bodies[i].a[k];
          }
        }

        nbodies.calculate_a();
#pragma omp single
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
#pragma omp single
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
#pragma omp single
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
  Interface(int n_, double tau_) : nbodies(n_), integrator(n_, tau_), n(n_) {};

  Interface(std::string &path,  double tau_) : nbodies(0), integrator(0, tau_), n(0) {

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

  void step(double num) { integrator.step(n, nbodies); }

  void reserve(int num) {
    integrator.yn.reserve(num);
    nbodies.bodies.reserve(num);
  }

  void resize(int num) {
    integrator.yn.resize(num);
    nbodies.bodies.resize(num);
  }


  template <class U> void save_state(const U &path) {
    std::ofstream outFile(path, std::ios::app);

    for (size_t i = 0; i < n; ++i)
      outFile << nbodies.bodies[i].r[0] << nbodies.bodies[i].r[1]
              << nbodies.bodies[i].r[2];

    outFile.close();
  }
};