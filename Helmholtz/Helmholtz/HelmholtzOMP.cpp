#include "HelmholtzSolver.h"



#ifndef HELMHOLTZ_OMP__
#define HELMHOLTZ_OMP__

struct Grid {
    double h;
    double k;
    int n;
    std::vector<double> grid;
    Grid(int n_, double k_) : h(1. / (n_ - 1)), grid(n_* n_, 0), n(n_), k(k_) {}

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

struct JacobyIteration {
    double err = 0.0;
    std::vector<double> old;
    JacobyIteration(int n) : old(n* n) {};
    double get_err() const { return err; }

    void step(Grid& matrix) {
        const int n = matrix.n;


//#pragma omp single
        old.swap(matrix.grid);

        double local_err = 0.0;
        double err_n = err / omp_get_num_threads();

//#pragma omp for nowait 
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                matrix.approximate_node(i, j, old);
                local_err += std::fabs(matrix(i, j) - old[i * n + j]);

            }
        }

        local_err /= n;



//#pragma omp critical 
        {
            err -= err_n;
            err += local_err;
        }

//#pragma omp barrier
    }
};

class RedNBlackIterations {
    double err = 0;
public:
    double get_err() { return err; };
    RedNBlackIterations(int n) {}
    void step(Grid& matrix) {
        int n = matrix.n;
        int nn = n - 2;



        double err_ = 0;
        double err_n = err / omp_get_num_threads();
//#pragma omp for
        for (int i = 0; i < nn; ++i)
            for (int j = i & 0x1; j < nn; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                
                err_ += std::fabs(matrix(i + 1, j + 1) - tmp);
            }

//#pragma omp for nowait
        for (int i = 0; i < nn; ++i)
            for (int j = (i + 1) & 0x1; j < nn; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                err_ += std::fabs( matrix(i + 1, j + 1) - tmp);
            }

        err_ /= n;



//#pragma omp critical 
        {
            err -= err_n;
            err += err_;
        }

//#pragma omp barrier

    }

};


template<class Solver>
class Interface {

    Grid matrix;
    Solver solver;
public:
    Interface(int n, double k) : matrix(n, k), solver(n) {};
    int solve(double err) {
        int i = 0;
//#pragma omp parallel shared(matrix, solver, i)  num_threads(omp_get_max_threads()) if (omp_get_max_threads() > 1)
        {
            do {

//#pragma omp master
                {
                    ++i;
                }

                solver.step(matrix);

            } while (solver.get_err() > err);
        }
        return i;

    }
    double get_node(int i, int j) { return matrix(i, j); };
    double get_err() { return solver.get_err(); }
    void print() { matrix.print(); }
    /*Какие-то еще доп. функции*/
};
#endif
