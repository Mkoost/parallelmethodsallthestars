#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>

constexpr const double pi = 3.141'592'653'589'793;
double f(double x, double y, double k) {
    return 2. * std::sin(pi * y) + k * k * (1. - x) * x * std::sin(pi * y) + pi * pi * (1. - x) * x * std::sin(pi * y);
}

double u(double x, double y) { return (1. - x) * x * std::sin(pi * y); }

struct HelmholtzGridScheme {
    double h;
    double k;
    int n;
    std::vector<double> grid;
    HelmholtzGridScheme(int n_, double k_) : h(1. / (n_ - 1)), grid(n_* n_, 0), n(n_), k(k_) {}

    double& operator()(int i, int j) { return grid[i * n + j]; }
    void approximate_node(int i, int j) { grid[i * n + j] = (h * h * f(h * j, h * i, k) + grid[i * n + (j - 1)] + grid[i * n + (j + 1)] + grid[(i - 1) * n + j] + grid[(i + 1) * n + j]) / (4. + k * k * h * h); }

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
    double err = 0;
    /*Какие-то еще переменные*/
    void step(HelmholtzGridScheme& matrix);

};

class RedNBlackIterations {
    double err = 0;
public:
    double get_err() { return err; };

    void step(HelmholtzGridScheme& matrix) {
        int n = matrix.n;


        double err_ = 0;
        double err_n = err / omp_get_num_threads();
#pragma omp for
        for (int i = 0; i < n - 2; ++i)
            for (int j = i % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

#pragma omp for
        for (int i = 0; i < n - 2; ++i)
            for (int j = (i + 1) % 2; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

        err_ /= n;

        
#pragma omp critical 
        {
            err += err_ - err_n;
        }

#pragma omp barrier

    }

};

template<class Solver>
class Interface{

    HelmholtzGridScheme matrix;
    Solver solver;
public:
    Interface(int n, double k) : matrix(n, k) {};
    int solve(double err) {
        int i = 0;
#pragma omp parallel shared(matrix, solver, i)  num_threads(omp_get_max_threads()) if (omp_get_max_threads() > 1)
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



int main()
{
    int n = 10000;
    double h = 1. / (n-1);
    Interface<RedNBlackIterations> interface(n, n / (h * h));
    double t = -omp_get_wtime();
    int iter_num = interface.solve(0.01);
    t += omp_get_wtime();

    std::cout << "error: " << interface.get_err() << " iteration num: " << iter_num << "\n\n";
    //interface.print();

    double realErr = 0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            realErr += std::fabs(interface.get_node(i, j) - u(j * h, i * h));
    std::cout << "Real error: " << realErr / n << " time: " << t << "s";

}

