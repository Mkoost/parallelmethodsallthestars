#include <iostream>
#include <cmath>
#include <vector>

constexpr const double pi = 3.141'592'653'589'793;
double f(double x, double y, double k) {
    return 2. * std::sin(pi * y) + k * k * (1. - x) * x * std::sin(pi * y) + pi * pi * (1. - x) * x * std::sin(pi * y);
}


struct HelmholtzGridScheme {
    double h;
    double k;
    int n;
    std::vector<double> grid;
    HelmholtzGridScheme(int n_, double k_) : h(1. / (n_ - 1)), grid(n_* n_, 0), n(n_), k(k_) {}

    double& operator()(int i, int j) { return grid[i * n + j]; }
    void approximate_node(int i, int j) { grid[i * n + j] = (h * h * f(h * j, h * i, k) + grid[i * n + (j - 1)] + grid[i * n + (j + 1)] + grid[(i - 1) * n + j] + grid[(i + 1) * n + j]) / (4. + k * k * h * h); }

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
            for (int j = i % 2 + 1; j < n - 2; j += 2)
            {
                double tmp = matrix(i + 1, j + 1);
                matrix.approximate_node(i + 1, j + 1);
                tmp = std::fabs(tmp - matrix(i + 1, j + 1));
                err_ += tmp;
            }

        err_ /= n;

#pragma omp atomic nowait
        err += err_;

    
    }

};

template<class Solver>
class Interface{
    HelmholtzGridScheme matrix;
    Solver solver;
public:
    int solve(double err) {
        int i = 0;
        do {
            ++i;
            solver.step();
        } while (solver.get_err() > err);
        return i;
    }
    /*Какие-то еще доп. функции*/
};


int main()
{
    std::cout << "Hello World!\n";
}

