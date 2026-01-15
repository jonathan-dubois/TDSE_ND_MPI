
# QuantumND: Parallel Attosecond Quantum Dynamics Solver

README.md
Version: v1.0 
Author: Jonathan DUBOIS

Copyright (C) 2025 Jonathan Dubois
 
Paris, December 11, 2025.
 
Contact information:
Jonathan DUBOIS
Laboratoire de Chimie Physique-Matière et Rayonnement (LCPMR) UMR7614
Sorbonne Université
4 place Jussieu
75252 Paris, France
E-mail: jonathan.dubois@cnrs.fr

This project is open-source. Feel free to submit pull requests or open issues on GitHub.
This project benefited from collaborative development and debugging sessions with Google Gemini 2.5 Flash.

## Project Overview ====================================

QuantumND is a highly efficient, parallel C program designed to solve the time-dependent Schrödinger equation (TDSE) for a quantum system in N spatial dimensions (ND) subject to external fields (e.g., perturbative electric fields, strong laser pulses).
The code leverages MPI (Message Passing Interface) for distributing the computational grid and FFTW (Fastest Fourier Transform in the West) for efficient propagation in both position and momentum spaces. It also utilizes shared-memory parallelism via OpenMP for optimizing local computations.
This project was developed in collaboration with the Google Gemini 2.5 Flash model.

## Features ============================================

- Parallel Execution: Hybrid parallelization using MPI (domain decomposition) and OpenMP (local threading).
- Multi-Dimensional: Supports N-dimensional grids.
- High-order symplectic integrators (Verlet, BM4, BM6, RKN4)
- Arbitrary potential (Ne_2D, H_1D, H_2D, H_3D,...)
- Multiple eigenstates via Gram-Schmidt orthonormalization
- Multi-Mode Fields: Handles arbitrary configurations of multiple laser modes, each with independent parameters (Intensity, Wavelength, Phase, etc.).
- Robust I/O: Uses high-speed binary file format for wavefunction output (PMD) and complex ASCII/binary structures for configuration.
- Flexible Configuration: Parameters can be set via a primary input file (params.real) and overridden via command-line arguments.

## Compilation and Setup ===============================

Prerequisites:
- C Compiler (e.g., GCC/Clang)
- MPI Library (e.g., Open MPI, MPICH)
- FFTW Library (Parallel MPI-enabled version required)
- OpenMP (usually bundled with the compiler)

### Example compilation command 

Adjust flags and paths as necessary for your system in the Makefile:

CFLAGS = -Wall -O3 -march=native -ffast-math -fopenmp -c 
LDFLAGS = -fopenmp -lfftw3_mpi -lfftw3 -lfftw3_threads -lpthread -lm
MPI_INC = -I/opt/homebrew/opt/fftw/include
MPI_LIB = -L/opt/homebrew/opt/fftw/lib

These are the typical flags and paths on macOS for FFTW installed with homebrew.

Then run:

>> make 

### Execution
The program is designed to be executed using mpirun to manage the distributed ranks and OMP_NUM_THREADS to manage local threads.

1) Basic Execution (Using Default Parameters)
Execute the program using the parameters defined entirely within the configuration file (params.real).

>> OMP_NUM_THREADS=2 mpirun -np 4 ./QuantumND_imag params.imag
>> OMP_NUM_THREADS=2 mpirun -np 4 ./QuantumND_real params.real

2) Parameter Overrides (Recommended for Sweeps)
Individual parameters can be overridden directly from the command line. This is crucial for running automated parameter sweeps (e.g., via cluster job scripts).

Syntax: KEY VALUE [KEY VALUE]...

Example: Run with 8 MPI ranks, 2 OpenMP threads, overriding the CEP and Intensity of the first mode (only mode 0 can be overridden).

>> OMP_NUM_THREADS=2 mpirun -np 8 ./QuantumND_real params.real CEP 0.5 INTENSITY 1.0e14

## Configuration file ==================================
The primary input file defines the grid structure, time integration, and default field parameters. It is essential that all array parameters have the exact number of required arguments.

### Parameters of the imaginary-time propagation:

All the parameters have default values, except the dimension (DIMENSION). It must be specified in params.imag and passed at the execution.

Parameter       Type            Required count  Example         Default values  Description
OUTPUT_IMAG     String          1               output.imag     output.imag     % specifies the name of the output file in which the results are saved
DIMENSION       Integer         1               2               [NO DEFAULT]    % dimension of the configuration space (1D, 2D, ...). The dimension should be compatible with the number of NR and DR.
N_STATES        Integer         1               4               1               % number of states to compute in the imaginary-time propagation
POTENTIAL       String          1               Ne_2D           H_3D            % name of the potential used. Already some are implemented (H_1D, H_2D, H_3D, Ne_2D, Ne_short_2D, Ar_2D, Kr_2D, Xe_2D)
DR              Double          DIMENSION       0.2     0.2     0.1             % step size in all directions (as many numbers as the value of DIMENSION)
NR              Long Integer    DIMENSION       512     512     256             % number of grid points in all directions (as many numbers as the value of DIMENSION)
INTEGRATOR      String          1               BM4             BM4             % integrator for the imaginary-time propagation. Already some are implemented (Verlet, BM4, BM6, RKN4)
DT              Double          1               0.1             0.1             % step size of the imaginary-time propagation
TINTEGRATION    Double          1               1.0e2           100.0           % integration time of the imaginary-time propagation

### Parameters of the real-time propagation:

All the parameters have default values, except the output file of the imaginary-time propagation (OUTPUT_IMAG), the number of initial states. They must be specified in params.imag and passed at the execution.
Note that the default value for the number of observables is 0 and the default observables are "none". It means that if they are not specified, the program would run, but no observables will be computed.

Parameter       Type            Required count  Example         Default values  Description
OUTPUT_IMAG     string          1               output.imag     [NO DEFAULT]    % specifies the name of the output file of the imaginary-time propagation to read 
NR              Long Integer    DIMENSION       4096    4096    256             % number of grid points in all directions (as many numbers as the value of DIMENSION and larger than NR of the imaginary-time propagation)
N_STATES        Integer         1               2               1               % number of states to consider in the superposition for the initial wave function
J_STATES        Integer         N_STATES        2       3       1               % indices of the states to consider from the states of the imaginary-time propagation
COEFFS_STATES   Double Complex  N_STATES        1.0     1.0j    1.0             % coefficients of the states to consider from the states of the imaginary-time propagation (in the form a+bj)
CAP             Double          2               0.1     20.0    0.0     0.0     % coefficients (a,R) of the complex absorbing potential V(r)=-I*a*(r-R)^2 (a>0 and R>0)
GAUGE           String          1               length          velocity        % gauge in which the calculations are performed
N_MODES         Integer         1               2               1               % number of field modes
INTENSITY       Double          N_MODES         1.0e14  1.0e13  1.0e14          % intensity of the laser electric field
LAMBDA          Double          N_MODES         800.0   400.0   800.0           % wavelength of the laser electric field
ELLIPTICITY     Double          N_MODES         1.0     0.0     0.0             % ellipticity of the laser electric field
CEP             Double          N_MODES         0.0     0.0     0.0             % carrier-envelope phase of the laser electric field
DELAY           Double          N_MODES         0.0     100.0   0.0             % delay of the laser electric field (must be positive)
ENVELOPE        String          N_MODES         cos4    cos4    cos4            % envelope of the laser electric field (constant, cos2, cos4)
DURATION        Double          N_MODES         100.0   150.0   220.638         % duration of the laser electric field
INTEGRATOR      String          1               BM4             BM4             % integrator for the real-time propagation (Verlet, BM4, BM6, RKN4)
DT              Double          1               0.1             0.1             % step size of the real-time propagation
TINTEGRATION    Double          1               200.0           220.638         % integration time of the real-time propagation
DT_SAVE         Double          1               0.1             1.0             % time step at which the time-dependent observables are computed
N_OBSERVABLES   Integer         1               2               0               % number of observables
OBSERVABLES     String          N_OBSERVABLES   PED     PMD     none            % observables (PMD, PED, wavefunction, Lz, energy, dipole_r, dipole_v)
P_MAX           Double          DIMENSION       2.0     2.0     1.0             % maximum momentum in the calculation of the PED and PMD
R_MASK          Double          2               20.0    30.0    20.0    30.0    % start and end radii of the mask function for the calculation of the PMD (Rmin and Rmax)
DE              Double          1               0.01            0.01            % energy step for the energy grid in the PED
NE              Integer         1               100             100             % number of points in the energy grid

## Output Files ========================================

The programs generates highly efficient binary output files for the wavefunctions and photoelectron momentum distribution (PMD). The headers can be read by humans.

### Output file for the imaginary-time propagation:

output.imag: Contains the essential parameters of the imaginary-time propagation, the potential, the energies of the converged states and the error on the energy.

### Output file for the real-time propagation:

electric_field.real: Contains the electric field as a function of time used for the calculation.
vector_potential.real: Contains the vector potential as a function of time used for the calculation.
observables.real: Contains the time-dependent observables and time.
PMD.real: If "PMD" is an observable.
PED.real: If "PED" is an observable.
wavefunction.real: If "wavefunction" is an observable.

### Read the output files from Matlab codes:
Matlab codes are also available in order to read the eigenstates ("read_data_imag.m"), the wavefunction ("read_wavefunction.m") and the PMD ("read_PMD.m").
