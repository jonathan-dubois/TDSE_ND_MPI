
#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <stddef.h>         // For ptrdiff_t (used in function prototypes).
#include <complex.h>        // For double complex (used in function prototypes).
#include <mpi.h>            // For MPI_Comm
#include "parameters.h"     // For the Parameters and Field structs.
#include "meshgrid.h"       // For the MeshGrid struct.

/* Constant definitions */
#define MAX_EXP_ARG             700.    // Largest argument that can be given in the argument of exp() functions
#define MAX_INTEGRATOR_STEPS    10      // Maximum number of integration steps in the integrator
#define MAX_INTEGRATOR_CHAR     10      // Maximum number of characters in the integrator's name

/* Data structures used by all files */
typedef struct { // Parameters of the integrator
    int s_integrator; // Integrator steps (the number of coefficients is 2*s_integrator)
    double as[2*MAX_INTEGRATOR_STEPS]; // Coefficients of the integrator
    double as_scaled[2*MAX_INTEGRATOR_STEPS]; // Coefficients of the integrator
} Integrator;

/* Function prototype */
void Initialize_integrator(const char* integrator_name, double dt, Integrator* Int, MPI_Comm comm);
void Initialize_field_vectors(const Parameters* P, Field* F, const Integrator* Int, MPI_Comm comm); // Initialize field vectors accordingly to the integrator step size
void Integration_step_imag(double complex *Psi, const MeshGrid *grid, const Integrator* Int);
void Integration_step_length(double complex *Psi, const double *Et, const int dim, const MeshGrid *grid, const Integrator* Int);
void Integration_step_velocity(double complex *Psi, const double *At, const int dim, const MeshGrid *grid, const Integrator* Int);
void Finalize_integrator(Integrator* Int);

#endif /* INTEGRATORS_H */
