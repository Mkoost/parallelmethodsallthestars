#include <cmath>
#include <vector>
#include <omp.h>
#include "HelmholtzSolver.h"



#ifndef HELMHOLTZ_MPI__
#define HELMHOLTZ_MPI__

struct HelmholtzGridSchemeMPI {
    double h;
    double k;
    int n;
    std::vector<double> grid;
    HelmholtzGridSchemeMPI(int n_, int m_, double k_) : h(1. / (n_ - 1)), grid(n_* m_ + 1, 0), n(n_), k(k_) {}

    double& operator()(int i, int j) { return grid[i * n + j]; }

    void approximate_node(int i, int j) {
        grid[i * n + j] = (h * h * f(h * j, h * i, k) + grid[i * n + (j - 1)] + grid[i * n + (j + 1)] + grid[(i - 1) * n + j] + grid[(i + 1) * n + j]) / (4. + k * k * h * h);
    }

    void approximate_node(int i, int j, std::vector<double>& vec) {
        grid[i * n + j] = (h * h * f(h * j, h * i, k) + vec[i * n + (j - 1)] + vec[i * n + (j + 1)] + vec[(i - 1) * n + j] + vec[(i + 1) * n + j]) / (4. + k * k * h * h);
    }

    void print() {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
                std::cout << grid[i * n + j] << " ";
            std::cout << std::endl;
        }
    }
};

struct JacobyIterationMPI_v1 {
    double err = 0.0;
    std::vector<double> old;
    JacobyIterationMPI_v1(int n) : old(n* n) {};
    double get_err() const { return err; }

    void step(HelmholtzGridScheme& matrix) {
        const int n = matrix.n;


#pragma omp single
        old.swap(matrix.grid);

        double local_err = 0.0;
        // double err_n = err / omp_get_num_threads();

#pragma omp for nowait 
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                matrix.approximate_node(i, j, old);
                local_err += std::fabs(matrix(i, j) - old[i * n + j]);

            }
        }

        local_err /= n;

#pragma omp single 
        {
            err = 0;
        }

#pragma omp critical 
        {
            err += local_err;
        }

#pragma omp barrier
    }
};

#endif