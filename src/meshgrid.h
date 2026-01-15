
#ifndef MESHGRID_H
#define MESHGRID_H

#include <complex.h>
#include <stddef.h> // For ptrdiff_t
#include <mpi.h>
#include <fftw3-mpi.h> // fftw_complex and fftw_plan
#include "parameters.h"

/* Data structures used by all files */
typedef struct { // Meshgrid of the system
    /* Grid in position and momentum */
    double *r; // Position coordinates along each direction
    double *pr; // Momentum coordinates along each direction
    ptrdiff_t *index; // Index to retrieve the position and momentum coordinates from the global index (size [dim*dim_HS])
    double *Vpot; // Potential function evaluated on the grid
    double *Vcap; // Complex absorbing potential
    double *Kin; // Kinetic energy evaluated on the grid
    
    /* Fourier transforms between position and momentum representations */
    double fft_scaling;
    fftw_complex *in; // Input in the fftw plan
    fftw_complex *out; // Output of the fftw plan
    fftw_plan plan_forward; // Plan forward
    fftw_plan plan_backward; // Plan backward
    
    /* Parameters related to the local system: Only Nr[0] becomes local_n0 */
    double dNr; // Infinitesimal volume in position
    double dNp; // Infinitesimal volume in momentum
    ptrdiff_t dim_HS; // Dimension of the Hilbert space: dim_HS=local_n0*Nr[1]*...*Nr[dim-1]
    ptrdiff_t offset_HS; // Where does the rank locates in the 0 to global size dimension
    ptrdiff_t *Nr_local; // Number of grid points along each direction
    ptrdiff_t *Nc; // Cumulative number of grid points: Nc={0,local_n0,local_n0+Nr[1],...,local_n0+...+Nr[dim-1]} which has a dimension dim+1
    ptrdiff_t *Nm; // Multiplicative number of grid points: Nm={1,local_n0,local_n0*Nr[1],...,local_n0*...*Nr[dim-1]} which has a dimension dim+1
} MeshGrid;

/* Function prototype */
void Initialize_meshgrid(const Parameters *P, MeshGrid *grid, const ptrdiff_t *local_n, const ptrdiff_t *local_start, ptrdiff_t alloc_local, MPI_Comm comm);
void Finalize_meshgrid(MeshGrid *grid);
void Recover_indices(ptrdiff_t *index, ptrdiff_t N, const int dim, const ptrdiff_t *Nr);

#endif /* MESHGRID_H */
