#include <cmath>
#include <vector>
#include <omp.h>
#include "HelmholtzSolver.h"



#ifndef HELMHOLTZ_MPI__
#define HELMHOLTZ_MPI__

struct GridBlock {
    double h;
    double k;
    int n;
    int m;
    std::vector<double> grid;
    GridBlock(int n_, int m_, double k_) : h(1. / (n_ - 1)), grid(n_* m_, 0), n(n_), m(m_), k(k_) {}

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

class RedNBlackIterationsBlock {
    double err = 0;
public:
    double get_err() { return err; };
    RedNBlackIterationsBlock(int n, int m) {}
    RedNBlackIterationsBlock() {}
    void step(GridBlock& matrix) {
        int n = matrix.n;
        int m = matrix.m;


        double err_ = 0;
        double err_n = err / omp_get_num_threads();
#pragma omp for
        for (int i = 0; i < m - 2; ++i)
            for (int j = i % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

#pragma omp for nowait
        for (int i = 0; i < n - 2; ++i)
            for (int j = (i + 1) % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }


#pragma omp single
        {
            err = 0;
        }

#pragma omp critical
        {
            err += err_ / m;
        }

        

#pragma omp barrier

    }


    template<class Solver>
    class InterfaceMPI {

        GridBlock matrix;
        Solver solver;
        int rank;
        int size;
    public:
        InterfaceMPI(int n, double k): solver(n, n / size + 1) {

            MPI_Comm_size(MPI_COMM_WORLD, &size);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            if (rank < n % size)
            {
                matrix.grid.resize(n, n / size + 1);
                matrix.k = k;
                matrix.h = 1. / (n - 1);
                matrix.n = n;
                matrix.m = n / size + 1;
                
            }
            else
            {
                matrix.grid.resize(n, n / size);
                matrix.k = k;
                matrix.h = 1. / (n - 1);
                matrix.n = n;
                matrix.m = n / size;

            }
            
        };
        
       void data_synchronize_v1() {
           int n = matrix.n;
            switch (rank) {
            case 0:
                MPI_Recv(&matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD);
                MPI_Send(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD);
                break;
            case size - 1:
                MPI_Send(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD);
                MPI_Recv(&matrix(0, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD);
                break;
             default:
                MPI_Send(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD);
                MPI_Recv(&matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD);

                MPI_Send(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD);
                MPI_Recv(&matrix(0, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD);
                break;
            }

        }

        int solve(double err) {
            int i = 0;
//#pragma omp parallel shared(matrix, solver, i)  num_threads(omp_get_max_threads()) if (omp_get_max_threads() > 1)
            do {

#pragma omp master
                {
                    ++i;
                }

                solver.step(matrix);

            } while (solver.get_err() > err);
            return i;
        }
        double get_node(int i, int j) { return matrix(i, j); };
        double get_err() { return solver.get_err(); }
        void print() { matrix.print(); }
        /*Какие-то еще доп. функции*/
    };

};


#endif