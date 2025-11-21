#include "HelmholtzSolver.h"


template<class Solver>
void time_test(int n = 1000){
    double h = 1. / (n-1);
    Interface<Solver> interface(n, 1/(h));
    double t = -omp_get_wtime();
    int iter_num = interface.solve(0.00001);
    t += omp_get_wtime();

    std::cout << "Error: " << interface.get_err() << " iteration num: " << iter_num << "\n";
    //interface.print();

    double realErr = 0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            realErr += std::fabs(interface.get_node(i, j) - u(j * h, i * h));
    std::cout << "Real error: " << realErr / n << " time: " << t << "s" << std::endl;
}



int main(int argc, char** argv)
{
    /*std::cout << "RedNBlackIterations" << std::endl;
    time_test<RedNBlackIterations>();

    std::cout << std::endl << "JacobyIteration" << std::endl;
    time_test<JacobyIteration>();
    std::cout << std::endl;*/

    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &size);

    std::cout << "process " << rank << std::endl;

    MPI_Finalize();
}
