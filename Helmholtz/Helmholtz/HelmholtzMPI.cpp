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

    int global_i;

    std::vector<double> grid;
    GridBlock() : h(0), k(0), n(0), m(0), grid(0), global_i(0){}
    GridBlock(int n_, int m_, double k_) : h(1. / (n_ - 1)), grid(n_* m_, 0), n(n_), m(m_), k(k_), global_i(0) {}

    double& operator()(int i, int j) { return grid[i * n + j]; }

    void approximate_node(int i, int j) {
        grid[i * n + j] = (h * h * f(h * j, h * (i + global_i), k) + grid[i * n + (j - 1)] + grid[i * n + (j + 1)] + grid[(i - 1) * n + j] + grid[(i + 1) * n + j]) / (4. + k * k * h * h);
    }

    void approximate_node(int i, int j, std::vector<double>& vec) {
        grid[i * n + j] = (h * h * f(h * j, h * (i + global_i), k) + vec[i * n + (j - 1)] + vec[i * n + (j + 1)] + vec[(i - 1) * n + j] + vec[(i + 1) * n + j]) / (4. + k * k * h * h);
    }

    void print() {
        std::string out = "";
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                out += std::to_string(grid[i * n + j]); 
                out += " ";
            }

            out += "\n";
        }
        std::cout << out;
    }
};

class RedNBlackIterationsBlock {
public:
    double err;
    int rank;
    int size;

    double get_err() { return err; };
    RedNBlackIterationsBlock(int n, int m) : err(0) {

    }
    RedNBlackIterationsBlock() : err(0){
    }
    void step(GridBlock& matrix) {
        int n = matrix.n;
        int m = matrix.m;


        double err_ = 0;
        double err_n = err / omp_get_num_threads();

        for (int i = 0; i < m - 2; ++i)
            for (int j = (i + matrix.global_i) % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

        for (int i = 0; i < m - 2; ++i)
            for (int j = (i + matrix.global_i + 1) % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

            err = 0;
            err += err_ / n;
    }
};

template<class Solver>
class InterfaceMPI {
public:
    GridBlock matrix;
    Solver solver;
    int rank;
    int size;

    InterfaceMPI(int n, double k) : solver(n, n / size + 1 + 2) {

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);


        int m = n / size + 2;
        int global_i = n / size * rank - 1;
        if (rank < n % size)
        {
            global_i += rank;
            ++m;
        }
        else
            global_i += n % size;

        if (rank == 0)
        {
            ++global_i;
            --m;
        }
        if (rank == size - 1)
            --m;

        //std::cout << "process: " << rank << " global i: " << global_i  << " m: " << m << " n: " << n << "\n";

        matrix.grid.resize(n * m);
        matrix.k = k;
        matrix.h = 1. / (n - 1);
        matrix.n = n;
        matrix.m = m;
        matrix.global_i = global_i;
    }

    void data_synchronize_v1() {
        int n = matrix.n;
        MPI_Status st;
        if (size == 1) return;
        if (rank == 0)
        {
            MPI_Recv(&matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &st);
            MPI_Send(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD);
        }
        else if (rank == size - 1)
        {
            MPI_Send(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD);
            MPI_Recv(&matrix(0, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &st);
        }
        else
        {
            MPI_Send(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD);
            MPI_Recv(&matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &st);

            MPI_Send(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD);
            MPI_Recv(&matrix(0, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &st);
        }


        double sm = 0;
        MPI_Allreduce(&solver.err, &sm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //sm /= matrix.n;
        solver.err = sm;
    }

    int solve(double err) {
        int i = 0;
        do {
            //std::cout << matrix.m << " " << matrix.n << " " << matrix.grid.size() << std::endl;

            ++i;


            solver.step(matrix);

            data_synchronize_v1();

        } while (solver.get_err() > err);
        return i;
    }


    double get_node(int i, int j) { return matrix(i, j); };
    double get_err() { return solver.get_err(); }
    void print() { matrix.print(); }
    /*Какие-то еще доп. функции*/
};




#endif