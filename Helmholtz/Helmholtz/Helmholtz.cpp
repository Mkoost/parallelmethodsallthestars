#include "HelmholtzSolver.h"


template<class Solver>
void time_test(int n = 20){
    double h = 1. / (n-1);
    Interface<Solver> interface(n, 1/(h));
    double t = -omp_get_wtime();
    int iter_num = interface.solve(0.0000000001);
    t += omp_get_wtime();

    std::cout << "Error: " << interface.get_err() << " iteration: " << iter_num << "\n";
    //interface.print();

    double realErr = 0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            realErr += std::fabs(interface.get_node(i, j) - u(j * h, i * h));
    std::cout << "Real error: " << realErr / n << " time: " << t << "s" << std::endl;
}

template<class Solver>
void time_test2(int n = 1003) {
    int rank;
    int size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double h = 1. / (n - 1);
    InterfaceMPI<Solver> interface(n, 1 / (h));
    double t = -omp_get_wtime();
    int iter_num = interface.solve(0.00000001);
    t += omp_get_wtime();

    std::cout << "Error: " << interface.get_err() << ", iteration: " << iter_num << ", process: " << rank << ", global_i: " << interface.matrix.global_i << "\n";
    //interface.print();
    //interface.print();

    double realErr = 0;
    int ii = 0;
    int mm = 0;
    if (rank == 0)
        ++ii;
    else if (rank == size - 1)
        --mm;
    else
    {
        ++ii;
        --mm;
    }
    
    for (int i = ii; i < interface.matrix.m + mm; ++i)
        for (int j = 0; j < n; ++j)
            realErr += std::fabs(interface.get_node(i, j) - u(j * h, (i + interface.matrix.global_i) * h));
    double sumerr;
    MPI_Allreduce(&realErr, &sumerr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0)
    {
        std::string out = "";
        std::cout << "Real error: " << sumerr / n << std::endl;
        
    }

    
    
}

int main(int argc, char** argv)
{
    /*std::cout << "RedNBlackIterations" << std::endl;
    time_test<RedNBlackIterations>();

    std::cout << std::endl << "JacobyIteration" << std::endl;
    time_test<JacobyIteration>();
    std::cout << std::endl;*/

    //time_test2<RedNBlackIterationsBlock>();
    
    
    MPI_Init(&argc, &argv);




    time_test2<RedNBlackIterationsBlock>();


    MPI_Finalize();
}
