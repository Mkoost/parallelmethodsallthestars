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


    double get_err() { return err; };
    RedNBlackIterationsBlock(int n, int m) : err(0) {

    }
    RedNBlackIterationsBlock() : err(0) {
    }
    void init(int n, int m) {}

    void step(GridBlock& matrix) {
        int n = matrix.n;
        int m = matrix.m;


        double err_ = 0;

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

        err = err_ / n;
    }

    void step_nonblock(GridBlock& matrix, MPI_Request* req, MPI_Status* sts, const int& num, const int& rank, const int& size) {
        int n = matrix.n;
        int m = matrix.m;


        double err_ = 0;

        for (int i = 2; i < m - 4; ++i)
            for (int j = (i + matrix.global_i) % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

        MPI_Waitall(num, req, sts);

        for (int i = 0; i < 2; ++i)
            for (int j = (i + matrix.global_i) % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

        for (int i = m - 4; i < m - 2; ++i)
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

        err = err_ / n;
    }
};

struct JacobyIterationBlock {
    double err = 0.0;
    std::vector<double> old;
    JacobyIterationBlock(int n, int m) : old(n * m) {};
    JacobyIterationBlock() = default;
    void init(int n, int m) { old.resize(n * m); }

    double get_err() const { return err; }

    void step(GridBlock& matrix) {
        int n = matrix.n;
        int m = matrix.m;

        old.swap(matrix.grid);

        double local_err = 0.0;
        double err_n = err / omp_get_num_threads();

        for (int i = 1; i < m - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                matrix.approximate_node(i, j, old);
                local_err += std::fabs(matrix(i, j) - old[i * n + j]);

            }
        }


        err = 0;
        err += local_err / n;

    }

    void step_nonblock(GridBlock& matrix, MPI_Request* req, MPI_Status* sts, const int& num, const int& rank, const int& size) {
        int n = matrix.n;
        int m = matrix.m;

        old.swap(matrix.grid);

        double local_err = 0.0;
        double err_n = err / omp_get_num_threads();

        for (int i = 2; i < m - 2; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                matrix.approximate_node(i, j, old);
                local_err += std::fabs(matrix(i, j) - old[i * n + j]);

            }
        }

        MPI_Waitall(num, req, sts);

            for (int j = 1; j < n - 1; ++j) {
                matrix.approximate_node(1, j, old);
                local_err += std::fabs(matrix(1, j) - old[1 * n + j]);

            }

            for (int j = 1; j < n - 1; ++j) {
                matrix.approximate_node(m - 2, j, old);
                local_err += std::fabs(matrix(m - 2, j) - old[(m - 2) * n + j]);

            }

        err = 0;
        err = local_err / n;
    
    
    }
};

struct ForwardingTypes {
    class SendRecv {};

    class SendAndRecv {};

    class ISendAndIRecv {};
};

template<class Solver, class Forwarding = ForwardingTypes::SendAndRecv>
class InterfaceMPI {
public:
    

    GridBlock matrix;
    Solver solver;
    int rank;
    int size;

    InterfaceMPI(int n, double k) {

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

        solver.init(n, m);
    }


    int solve(double err) {
        if constexpr (std::is_same<Forwarding, ForwardingTypes::ISendAndIRecv>())
            return solve_nonblock(err);
        else
            return solve_block(err);
    }

    int solve_block(double err) {
        int i = 0;
        
            do {
                //std::cout << matrix.m << " " << matrix.n << " " << matrix.grid.size() << std::endl;

                ++i;


                solver.step(matrix);
                if constexpr (std::is_same<Forwarding, ForwardingTypes::SendRecv>())
                    data_synchronize_v2();
                else
                    data_synchronize_v1();

            } while (solver.get_err() > err);
            return i;
        
    }

    int solve_nonblock(double err) {
        int i = 0;
        MPI_Request req[4]{ MPI_REQUEST_NULL , MPI_REQUEST_NULL , MPI_REQUEST_NULL , MPI_REQUEST_NULL};
        MPI_Status sts[4];

        int num = (rank != 0) && (rank != size - 1) ? 4 : 2;

        do {
            //std::cout << matrix.m << " " << matrix.n << " " << matrix.grid.size() << std::endl;

            ++i;


            solver.step_nonblock(matrix, req, sts, num, rank, size);

            MPI_Waitall(num, req, sts);
            data_synchronize_v3(req);

        } while (solver.get_err() > err);
        MPI_Waitall(num, req, sts);

        return i;

    }


    double get_node(int i, int j) { return matrix(i, j); };
    double get_err() { return solver.get_err(); }
    void print() { matrix.print(); }

private:
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


        err_synchronize();
    }

    void data_synchronize_v2() {
        int n = matrix.n;
        MPI_Status st;

        if (size == 1) return;
        if (rank == 0)
        {
            
            MPI_Sendrecv(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 10, &matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &st);
        }
        else if (rank == size - 1)
        {
            MPI_Sendrecv(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, &matrix(0, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &st);
        }
        else
        {
            MPI_Sendrecv(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, &matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &st);

            MPI_Sendrecv(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 10, &matrix(0, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &st);
        }


        err_synchronize();
    }

    void data_synchronize_v3(MPI_Request* req) {
        int n = matrix.n;

        err_synchronize();

        if (size == 1) return;
        if (rank == 0)
        {
            MPI_Irecv(&matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, req);
            MPI_Isend(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 20, MPI_COMM_WORLD, req + 1);
        }
        else if (rank == size - 1)
        {
            MPI_Isend(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, req);
            MPI_Irecv(&matrix(0, 0), n, MPI_DOUBLE, rank - 1, 20, MPI_COMM_WORLD, req + 1);
        }
        else
        {
            MPI_Isend(&matrix(1, 0), n, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, req);
            MPI_Irecv(&matrix(matrix.m - 1, 0), n, MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, req + 1);

            MPI_Isend(&matrix(matrix.m - 2, 0), n, MPI_DOUBLE, rank + 1, 20, MPI_COMM_WORLD, req + 2);
            MPI_Irecv(&matrix(0, 0), n, MPI_DOUBLE, rank - 1, 20, MPI_COMM_WORLD, req + 3);
        }


        
    }

    void err_synchronize() {

        double sm = 0;
        MPI_Allreduce(&solver.err, &sm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        solver.err = sm;
    }
};




#endif