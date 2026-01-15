
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <stddef.h>     // For ptrdiff_t
#include <complex.h>    // For double complex
#include <mpi.h>        // For MPI_Comm
#include <stdbool.h>    // For booleans
#include <stdio.h>      // For fopen, fscanf, fprintf, printf, perror, snprintf

/* Constant definitions */
/** Related to the parameters **/
#define MAX_INTEGRATOR_CHAR     10 // Maximum number of characters in the integrator's name
#define MAX_POTENTIAL_CHAR      20 // Maximum number of characters in the potential's name
#define MAX_OUTPUT_FILES_CHAR   30 // Maximum number of characters in the output file's name
#define MAX_OBSERVABLE_CHAR     32 // Maximum number of characters in the observables's name

/** Related to the laser electric field **/
#define MAX_GAUGE_CHAR          10 // Maximum number of characters in the gauge's name
#define MAX_ENVELOPE_CHAR       10 // Maximum number of characters in the envelope's name

/** Reasonable max length for parameter names to be read **/
#define MAX_PARAM_NAME_LEN      128

/* Data structures used by all files */
/** Other structure declaration **/
typedef struct { // Parameters of the system read out from the file
    /** Parameters related to the global system **/
    int dim; // Dimension of the configuration space
    ptrdiff_t *Nr_global; // Number of grid points along each direction (array of size dim)
    double *dr; // Step size of the grid along each direction (array of size dim)
   
    /** Initial conditions **/
    int N_states; // Number of states to compute or to be superposed in the initial condition
    int *j_states; // Which states from the imaginary time propagation to propagate and to superpose to the initial wavefunction
    double complex *coeffs_states; // Coefficients in the superposition of the initial state
    
    /** Time-integration parameters **/
    double dt; // Time-step of the imaginary-time integration
    double Tintegration; // Duration of the imaginary-time integration
    char integrator[MAX_INTEGRATOR_CHAR]; // Name of the integrator
    
    /** Output files to be read or written **/
    char output_imag[MAX_OUTPUT_FILES_CHAR]; // Name of the file in which the output is saved
    
    /** Potential energy **/
    char potential[MAX_POTENTIAL_CHAR]; // Name of the potential
    double cap[2]; // Complex absorbing potential: cap[0] strength, cap[1] length
    
    /** Parameters computed from the options **/
    double *time; // Vector saving the values of time
    ptrdiff_t N_time; // Length of the time vector, i.e. number of integration steps
    
    /** Parameters for the observables **/
    double dt_save; // The time step at which the observables must be computed
    int N_observables; // Number of observables to compute
    char** observables; // Array of string pointers (char*) for the name of the observables
    
    /** Parameters related to the calculation of the PMD & PED **/
    double *pr_max; // Maximum momentum coordinates to use and save (array of size dim)
    double R_mask[2]; // Mask for removing the bound states (R_mask[0]=Rmin & R_mask[1]=Rmax)
    int Ne; // Number of energy steps for the PED
    double de; // Energy step for the PED
} Parameters;

typedef struct { // Parameters of the field
    /** Global parameters **/
    int N_modes; // Number of independent laser modes (N)
    char gauge[MAX_GAUGE_CHAR]; // Name of the gauge laser
    
    /** Parameters specific to the real-time propagation **/
    double *intensity; // Intensities of the pulses
    double *lambda; // Frequencies of the pulses
    double *cep; // Carrier-envelope phases of the pulses
    double *ellipticity; // Ellipticities of the pulses
    double *duration; // Durations of the pulses
    double *delay; // Delays of the pulses
    char **envelope; // Name of the envelope laser
    
    /** Parameters computed from the options **/
    double *frequency; // Wavelength of the laser
    double *amplitude; // Amplitude of the laser
    
    /** Pre-calculated field arrays **/
    double *E; // Values of the laser electric field used throughout the propagation
    double *A; // Values of the laser vector potential used throughout the propagation
} Field;

/* Function prototype */
/** Utility functions **/
double complex parse_complex(const char* str);
int Expected_arguments(FILE *file_ptr, const char *parameter_name, int expected_number, const char *KnownKeys[], MPI_Comm comm);
int Find_dimension_in_file_MPI(const char* filename, MPI_Comm comm);
void Allocate_dim_arrays(Parameters* P, int dim, MPI_Comm comm);
bool Is_known_key(const char *token, const char **key_list);

/** General parameters read from files **/
void Initialize_parameters_imag(const char* filename, Parameters* P, int argc, char *argv[], MPI_Comm comm);
void Initialize_parameters_real(const char* filename, Parameters* P, Parameters* P_imag, Field* F, int argc, char *argv[], MPI_Comm comm);
void Finalize_parameters(Parameters* P);

/** Laser electric field and vector potential **/
void Field_vectors(const Parameters* P, const Field* F, const double t, double* A, double *E, MPI_Comm comm);
void Finalize_field_vectors(Field* F);

#endif /* PARAMETERS_H */
