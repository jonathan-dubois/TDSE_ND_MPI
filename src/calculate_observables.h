
#ifndef CALCULATE_OBSERVABLES_H
#define CALCULATE_OBSERVABLES_H

#include "meshgrid.h"
#include <complex.h>
#include <mpi.h>

/* Structure definition */
typedef struct {
    /** Metadata for Calculation/Output **/
    int total_count; // Total number of requested observables (e.g., 4)
    int total_values; // Total size needed for contiguous buffer (e.g., 1+dim+1 = 5)
    
    /** Lookup Arrays (Index is the observable request number i) **/
    char** names; // Array of string names ("Lz", "dipole", ...)
    int* dimensions; // Array storing the size of each result (1, P->dim, 1, ...)
    int* offsets; // Array of starting indices in the contiguous buffer (0, 1, 1+dim, ...)
    
    /** Result Buffers (Used for MPI communication) **/
    double complex* local_results; // Contiguous buffer for local rank summation
    double complex* global_results; // Contiguous buffer for final, summed results
    double* values; // Current values of the observables
} Observables;

/* Function prototypes */
/** Global **/
void Calculation_global_energies(double *Ej_global, const int N, const double complex *Psi, const MeshGrid *grid, MPI_Comm comm);
void Calculation_global_observables(const double complex *Psi, int dim, const MeshGrid *grid, const Observables *Obs, double *alpha, MPI_Comm comm);

/** Local **/
void Calculation_local_Lz(double complex *Lz, const double complex *Psi, const MeshGrid *grid, double *alpha, MPI_Comm comm);
void Calculation_local_energies(double complex *Ej_local, const int N, const double complex *Psi, const MeshGrid *grid, MPI_Comm comm);
void Calculation_local_dipole_r(double complex *dipole, int dim, const double complex *Psi, const MeshGrid *grid);
void Calculation_local_dipole_v(double complex *dipole, int dim, const double complex *Psi, double *alpha, const MeshGrid *grid);

/** Utility functions **/
void Initialize_observables(Observables* Obs, char **observables_name, int observables_number, int dim, MPI_Comm comm);
void Finalize_observables(Observables* Obs);

#endif /* CALCULATE_OBSERVABLES_H */
