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

    void data_synchronize_v2() {
        if (size <= 1) return;

        int n = matrix.n;

        // Только внутренние границы обмениваем
        if (rank > 0) {
            // Отправляем свою верхнюю строку вверх + получаем ghost сверху
            MPI_Sendrecv(
                &matrix(1, 0),          n, MPI_DOUBLE, rank-1, 100,
                &matrix(0, 0),          n, MPI_DOUBLE, rank-1, 200,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // rank == 0: верхняя ghost-строка всегда 0 — НЕ получаем ничего!
            std::fill(&matrix(0, 0), &matrix(0, n), 0.0);
        }

        if (rank < size - 1) {
            // Отправляем нижнюю строку вниз + получаем ghost снизу
            MPI_Sendrecv(
                &matrix(matrix.m-2, 0), n, MPI_DOUBLE, rank+1, 200,
                &matrix(matrix.m-1, 0), n, MPI_DOUBLE, rank+1, 100,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // rank == size-1: нижняя ghost-строка всегда 0
            std::fill(&matrix(matrix.m-1, 0), &matrix(matrix.m-1, n), 0.0);
        }

        // Ошибка — как и раньше
        double global_err = 0.0;
        MPI_Allreduce(&solver.err, &global_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        solver.err = global_err;
    }

    int solve(double err) {
        int i = 0;
        do {
            //std::cout << matrix.m << " " << matrix.n << " " << matrix.grid.size() << std::endl;

            ++i;


            solver.step(matrix);

            data_synchronize_v2();

        } while (solver.get_err() > err);
        return i;
    }

    int solve_nonblock(double err) {
        int iter = 0;
        MPI_Request requests[4];
        int req_cnt = 0;
        int n = matrix.n;
        int m = matrix.m;

        // === Первая итерация: просто считаем без перекрытия ===
        solver.step(matrix);

        while (true) {
            ++iter;

            // 1. Запускаем неблокирующий обмен границами ПЕРЕД вычислениями
            req_cnt = 0;
            if (size > 1) {
                if (rank > 0) {
                    MPI_Isend(&matrix(1, 0),     n, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &requests[req_cnt++]);
                    MPI_Irecv(&matrix(0, 0),     n, MPI_DOUBLE, rank-1, 200, MPI_COMM_WORLD, &requests[req_cnt++]);
                }
                if (rank < size-1) {
                    MPI_Isend(&matrix(m-2, 0),   n, MPI_DOUBLE, rank+1, 200, MPI_COMM_WORLD, &requests[req_cnt++]);
                    MPI_Irecv(&matrix(m-1, 0),   n, MPI_DOUBLE, rank+1, 100, MPI_COMM_WORLD, &requests[req_cnt++]);
                }
            }

            // 2. СЧИТАЕМ НОВЫЕ ЗНАЧЕНИЯ — коммуникации идут параллельно!
            solver.step(matrix);

            // 3. Только теперь ждём, когда придут ghost-значения от предыдущей итерации
            if (req_cnt > 0) {
                MPI_Waitall(req_cnt, requests, MPI_STATUSES_IGNORE);
            }

            // 4. Принудительно зануляем внешние ГУ (важно!)
            if (rank == 0)       std::fill(&matrix(0,0),     &matrix(0,n),     0.0);
            if (rank == size-1)  std::fill(&matrix(m-1,0),   &matrix(m-1,n),   0.0);

            // 5. Собираем ошибку
            double global_err = 0.0;
            MPI_Allreduce(&solver.err, &global_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            solver.err = global_err;

            if (solver.err <= err) break;
        }

        return iter;
    }


    double get_node(int i, int j) { return matrix(i, j); };
    double get_err() { return solver.get_err(); }
    void print() { matrix.print(); }
};




#endif