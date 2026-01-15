
#ifndef IO_FUNCTIONS_H
#define IO_FUNCTIONS_H

#include "parameters.h"
#include "meshgrid.h"
#include "calculate_observables.h"
#include <mpi.h>
#include <complex.h>

/* Function prototype */
/** Writing **/
void Write_wavefunction_to_file(const char* filename, double complex* Psi, const Parameters* P, const MeshGrid* grid, MPI_Comm comm);
void Write_output_imag(const char* filename, double complex** Psi, double** Ej_global, const Parameters* P, const MeshGrid* grid, MPI_Comm comm);
void Write_observables(FILE *file_ptr, double time, const Observables* Obs, MPI_Comm comm);
void Write_final_observables(double complex *Psi, const Parameters *P, MeshGrid *grid, const Observables *Obs, ptrdiff_t *local_start, MPI_Comm comm);

/** Reading **/
void Read_wavefunction_MPI(const Parameters *P, const Parameters *P_imag, const MeshGrid *grid, double complex *Psi, ptrdiff_t *local_start_real, MPI_Comm comm);

#endif /* IO_FUNCTIONS_H */
