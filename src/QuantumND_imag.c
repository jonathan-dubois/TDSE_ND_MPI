/**--------------------------------------------------------------------------------
 QuantumND_imag.c, v1.0 21 November, 2025.
 Author: Jonathan DUBOIS
 
 This code uses the imaginary-time propagation to compute the first eigenstates of the Hamilton operator H = - Delta/2 + V(r), with Delta the Laplacian operator in N-dimensions. The potential energy is arbitrary and can have any symmetry.
 The exponential of the time-evolution operator is computed using the split-operator method.
 Different orders of the split-operator are available and can be chosen as parameters.
 Multiple states can be considered and are computed using the Gram-Schmidt orthonormalization method.
 The FFTW is parallelized on different threads and with MPI.
 
 Copyright (C) 2025 Jonathan Dubois
 
 Paris, September 2025.
 
 Contact information:
 Jonathan DUBOIS
 Laboratoire de Chimie Physique-Matière et Rayonnement (LCPMR) UMR7614
 Sorbonne Université
 4 place Jussieu
 75252 Paris, France
 E-mail: jonathan.dubois@cnrs.fr
 --------------------------------------------------------------------------------**/

/* Include files */
/** Text and print **/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>

/** FFTW & MPI Framework **/
#include <fftw3.h>                  // Included before fftw3-mpi.h for complex type compatibility
#include <mpi.h>
#include <fftw3-mpi.h>

/** Local structure definition & prototypes */
#include "parameters.h"             // Defines Parameters struct, Field struct, and prototypes for Initialize/Finalize_parameters
#include "meshgrid.h"               // Defines MeshGrid struct, and prototypes for Initialize/Finalize_meshgrid, and Recover_indices
#include "integrators.h"            // Defines Integrator struct, and prototypes for Initialize/Finalize_integrators, and Integration_step_imag
#include "calculate_observables.h"  // Prototypes for Calculation_global_energies
#include "io_functions.h"           // Prototypes for writing data

/*######################################*/
// DEFINITION OF FUNCTIONS
/*######################################*/
void Gram_Schmidt_orthonormalization(const int N, double complex **Psi, const MeshGrid *grid, MPI_Comm comm);

//*######################################*/
// MAIN PROGRAM
/*######################################*/
int main(int argc, char* argv[]){
    /* Initialization of the MPI framework */
    MPI_Init(&argc,&argv);
    fftw_mpi_init();
    MPI_Comm comm=MPI_COMM_WORLD; // Obtain the default communicator (the group of ALL processes started)
    int rank, size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    
    if (argc<2){ // Ensure the parameter file name is provided
        fprintf(stderr,"Usage: %s <param_file> [KEY VALUE ...]\n",argv[0]);
        MPI_Abort(comm,1);
    }
    
    if (rank==0){
        printf("\n##############################################\n");
        printf("\n##    PROGRAM IMAGINARY-TIME PROPAGATION    ##\n");
        printf("\n##############################################\n");
    }
    
    /* Initialization */
    ptrdiff_t i; // Declaration of a large integer (loop over dim_HS)
    int j,k;
    double radius;
    
    /** Declaration of all the structures that can be called in other functions **/
    Parameters P_imag;
    Field F;
    Observables Obs;
    memset(&P_imag,0,sizeof(Parameters));
    memset(&F,0,sizeof(Field));
    memset(&Obs,0,sizeof(Observables));
    
    /** Parameters **/
    if (rank==0){printf("\n# Parameters read and broadcasted to all ranks\n\n");}
    Parameters P; // Declare the structure of parameters
    memset(&P,0,sizeof(Parameters)); // Initialize it to zero
    Initialize_parameters_imag(argv[1],&P,argc,argv,comm); // Initialize the parameters
    
    /** Initialize integrator **/
    Integrator Int;
    memset(&Int,0,sizeof(Integrator)); // Initialize it to zero
    Initialize_integrator(P.integrator,P.dt,&Int,comm);
    
    /** Initialization of the MPI variables **/
    ptrdiff_t *local_n=NULL, *local_start=NULL;
    ptrdiff_t alloc_local;
    local_n=malloc(P.dim*sizeof(ptrdiff_t));
    local_start=malloc(P.dim*sizeof(ptrdiff_t));
    if (local_n==NULL || local_start==NULL){
        if (local_n) free(local_n);
        if (local_start) free(local_start);
        fprintf(stderr, "ERROR: memory allocation failed for local_n and local_start.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    if (P.dim==1){alloc_local=fftw_mpi_local_size_1d(P.Nr_global[0],comm,FFTW_FORWARD,0,local_n,local_start,local_n,local_start);}
    else {alloc_local=fftw_mpi_local_size_many(P.dim,P.Nr_global,1,FFTW_MPI_DEFAULT_BLOCK,comm,local_n,local_start);}
    // Every MPI rank knows its number of points (local_n) and location in the local grid (local_start)
    
    /** Meshgrid and Fourier transform **/
    MeshGrid grid; // Declare the structure of meshgrid
    memset(&grid,0,sizeof(MeshGrid)); // Initialize it to zero
    Initialize_meshgrid(&P,&grid,local_n,local_start,alloc_local,comm); // Initialize the grid
    if (grid.dim_HS!=alloc_local){
        fprintf(stderr,"ERROR: Problem with the dimension of the local Hilbert space and the local allocation.\n");
        fprintf(stderr," dim_HS = %td \n alloc_local = %td\n",grid.dim_HS,alloc_local);
        perror("fftw MPI");
        MPI_Abort(comm,1);
    }
    
    /** Wavefunction: Memory allocation & initialization **/
    double complex **Psi=NULL;
    Psi=malloc(P.N_states*sizeof(double complex*));
    if (!Psi) {perror("malloc with Psi_local"); MPI_Abort(comm,1);}
    for (j=0;j<P.N_states;j++){
        Psi[j]=fftw_alloc_complex(grid.dim_HS);
        if (!Psi[j]) {perror("fftw_alloc_complex with Psi_local"); MPI_Abort(comm,1);}
        memset(Psi[j],0,grid.dim_HS*sizeof(double complex)); // Initialization of the wave functions
    }
    
    /** Parameter for the length vs velocity gauge calculation of the observables (here 0) **/
    double *alpha=NULL;
    alpha=(double*) malloc(P.dim*sizeof(double));
    if (alpha==NULL){
        fprintf(stderr, "ERROR: memory allocation failed for alpha\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (j=0;j<P.dim;j++){alpha[j]=0.;}
    
    /* Propagation */
    if (rank==0){printf("\n# Imaginary-time propagation\n");}
    clock_t start;
    struct timespec start_real,end_real;
    double time;
    
    double **Ej_global=NULL;
    Ej_global=(double**) malloc(P.N_states*sizeof(double*));
    if (Ej_global==NULL){
        fprintf(stderr, "ERROR: memory allocation failed for Ej_global.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (j=0;j<P.N_states;j++){
        Ej_global[j]=(double*) malloc(2*sizeof(double));
        if (Ej_global[j]==NULL){
            for (k=0;k<j;k++){free(Ej_global[k]);}
            free(Ej_global);
            fprintf(stderr, "ERROR: memory allocation failed for Ej_global.\n");
            perror("malloc");
            MPI_Abort(comm,1);
        }
    }
    
    for (j=0;j<P.N_states;j++){
        start=clock();
        clock_gettime(CLOCK_MONOTONIC,&start_real);
        
        /** Initialization of the state **/
        for (i=0;i<grid.dim_HS;i++){
            radius=0;
            for (k=0;k<P.dim;k++){radius+=grid.r[grid.index[k*grid.dim_HS+i]]*grid.r[grid.index[k*grid.dim_HS+i]];} // Recover the associated position coordinate and compute its norm
            radius=sqrt(radius);
            Psi[j][i]=(radius+(i%3))*exp(-radius)+I*0.; // Initialization of the wavefunction with no symmetry
        }
        
        /** Imaginary-time propagation **/
        for (time=0.;time<P.Tintegration;time+=P.dt){
            Integration_step_imag(Psi[j],&grid,&Int); // Integration of the jth state using the split-operator method
            Gram_Schmidt_orthonormalization(j,Psi,&grid,comm); // Gram-Schmidt orthonormalization procedure with respect to all the states already computed
        }
        
        /** Properties of the converged state **/
        Calculation_global_energies(Ej_global[j],2,Psi[j],&grid,comm); // Calculation of the 2 energy moment E and E^2 of the jth eigenstate
        clock_gettime(CLOCK_MONOTONIC,&end_real);
        
        char *name_Lz="Lz";
        Initialize_observables(&Obs,&name_Lz,1,P.dim,comm);
        Calculation_global_observables(Psi[j],P.dim,&grid,&Obs,alpha,comm);
        if (rank==0){ // Pint only if rank 0
            printf("\n> eigenfunction %d:\n",j+1);
            printf("  E\t\t\t%16.15f\n",Ej_global[j][0]);
            printf("  DE\t\t\t%5.4e \n",Ej_global[j][1]-Ej_global[j][0]*Ej_global[j][0]);
            printf("  Lz\t\t\t%5.4e \n",Obs.values[0]);
            printf("  Elapsed CPU time\t%16.15f seconds\n",(double)(clock()-start)/CLOCKS_PER_SEC);
            printf("  Elapsed real time\t%16.15f seconds\n",(double)(end_real.tv_sec-start_real.tv_sec)+(end_real.tv_nsec-start_real.tv_nsec)/1e9);
        }
    }
    printf("\n");
    
    /* Make all the eigenfunctions reals by removing the arbitrary global phase */
    double rho_local=0.;
    double phase_local=0.;
    double mag,phase;
    double complex factor;
    struct{
        double value; // rho
        int rank; // rank number
    } in, out;
    
    for (j=0;j<P.N_states;j++){
        rho_local=0.;
        phase_local=0.;
        
        /** Calculation of the local maximum **/
        for (i=0;i<grid.dim_HS;i++){
            mag=cabs(Psi[j][i]);
            if (mag>rho_local){
                rho_local=mag;
                phase_local=carg(Psi[j][i]); // phase at the local max
            }
        }
        
        /** Reduce the maximum magnitude + the rank where it occurs **/
        in.value=rho_local;
        in.rank=rank;
        MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT,MPI_MAXLOC,comm);
        
        /** Broadcast the phase from the winning rank **/
        phase=(rank==out.rank)?phase_local:0.;
        MPI_Bcast(&phase,1,MPI_DOUBLE,out.rank,comm);
        
        /** Remove the global phase **/
        factor=cexp(-I*phase);
        for (i=0;i<grid.dim_HS;i++){Psi[j][i]*=factor;}
    }
    
    /* Save data */
    Write_output_imag(P.output_imag,Psi,Ej_global,&P,&grid,comm); // Save the data in the output file to be read out by QuantumND_real.c
    
    /* Finalization */
    if (rank==0){printf("# The code ended successfully\n\n");}
    free(alpha);
    free(local_n);
    free(local_start);
    for (j=0;j<P.N_states;j++){free(Ej_global[j]);}
    free(Ej_global);
    for (j=0;j<P.N_states;j++){fftw_free(Psi[j]);}
    free(Psi);
    Finalize_meshgrid(&grid);
    Finalize_integrator(&Int);
    Finalize_parameters(&P);
    Finalize_parameters(&P_imag);
    Finalize_field_vectors(&F);
    Finalize_observables(&Obs);
    fftw_cleanup(); // Finish with FFTW
    MPI_Finalize(); // Finish with MPI
    return 0;
}

void Gram_Schmidt_orthonormalization(const int N, double complex **Psi, const MeshGrid *grid, MPI_Comm comm){ // Gram-Schmidt orthonormalization procedure for the N's state in Psi. It is therefore computed with respect to all states smaller than N. It gives the Nth wavefunction normalized and orthonormal with respect to the others already computed. The code assumes that the states 1,...,N-1 are already normalized and orthogonal with each other.
    ptrdiff_t i;
    int j;
    
    /* Memory allocation for the Nth vector & initialization */
    double complex *v=NULL;
    v=(double complex*) malloc(grid->dim_HS*sizeof(double complex));
    if (v==NULL){
        fprintf(stderr, "ERROR: memory allocation failed for v in Gram_Schmidt_orthonormalization.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (i=0;i<grid->dim_HS;i++){v[i]=Psi[N][i];}
    
    /* Modified Gram-Schmidt: subtract projections (more stable since it does not assume perfect orthogonality between the other states) */
    double complex scalar_local;
    double complex scalar_global;
    for (j=0;j<N;j++){
        scalar_local=0.+0.*I;
        for (i=0;i<grid->dim_HS;i++){scalar_local+=conj(Psi[j][i])*v[i]*grid->dNr;} // Local scalar product calculation
        MPI_Allreduce(&scalar_local,&scalar_global,1,MPI_C_DOUBLE_COMPLEX,MPI_SUM,comm); // Global summation
        for (i=0;i<grid->dim_HS;i++){v[i]-=scalar_global*Psi[j][i];}
    }

    /* Normalization */
    double tol=1.e-12;
    double norm_local=0.;
    double norm_global;
    for (i=0;i<grid->dim_HS;i++){norm_local+=cabs(v[i])*cabs(v[i])*grid->dNr;} // Local norm calculation
    MPI_Allreduce(&norm_local,&norm_global,1,MPI_DOUBLE,MPI_SUM,comm); // Global summation
    
    if (norm_global<tol) { // Use a small tolerance
        fprintf(stderr,"ERROR: Wavefunction norm decayed to zero during Gram-Schmidt.\n");
        MPI_Abort(comm,1);
    }
    
    double invnorm=1./(double)sqrt(norm_global);
    for (i=0;i<grid->dim_HS;i++){Psi[N][i]=v[i]*invnorm;}
    
    /* Finalization */
    free(v);
}
