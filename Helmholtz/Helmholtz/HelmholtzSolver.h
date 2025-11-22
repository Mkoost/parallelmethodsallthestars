#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <string>

#ifndef MYMATHEMATICS__
#define MYMATHEMATICS__
constexpr const double pi = 3.141'592'653'589'793;
double f(double x, double y, double k) {
    return 2. * std::sin(pi * y) + k * k * (1. - x) * x * std::sin(pi * y) + pi * pi * (1. - x) * x * std::sin(pi * y);
}

double u(double x, double y) { return (1. - x) * x * std::sin(pi * y); }
#endif 




#include "HelmholtzOMP.cpp"
#include "HelmholtzMPI.cpp"