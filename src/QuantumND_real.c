/**--------------------------------------------------------------------------------
 QuantumND_real.c, v1.0 25 November, 2025.
 Author: Jonathan DUBOIS
 
 This code uses the real-time propagation to compute the propagation of a wavefunction through the Hamilton operator H = - [p+A(t)]^2/2 + V(r), where p=-i\nabla is the momentum operator.
 The exponential of the time-evolution operator is computed using the split-operator method.
 Different orders of the split-operator are available and can be chosen as parameters.
 
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
#include <fftw3.h>
#include <time.h>

/** FFTW & MPI Framework **/
#include <fftw3.h>
#include <mpi.h>
#include <fftw3-mpi.h>

/** Local structure definition & prototypes */
#include "parameters.h"             // Defines Parameters struct, Field struct, and prototypes for Initialize/Finalize_parameters
#include "meshgrid.h"               // Defines MeshGrid struct, and prototypes for Initialize/Finalize_meshgrid, and Recover_indices
#include "integrators.h"            // Defines Integrator struct, and prototypes for Initialize/Finalize_integrators, and Integration_step_imag
#include "calculate_observables.h"  // Prototypes for Calculation_global_energies
#include "io_functions.h"           // Prototypes for writing data

//*######################################*/
// MAIN PROGRAM
/*######################################*/
int main(int argc, char* argv[]){
    /* Initialization of the MPI framework */
    MPI_Init(&argc,&argv);
    fftw_mpi_init();
    MPI_Comm comm=MPI_COMM_WORLD; // Obtain the default communicator (the group of ALL processes started)
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); // Initialize the rank of MPI
    MPI_Comm_size(MPI_COMM_WORLD,&size); // Initialize the size of each rank
    
    if (argc<2){ // Ensure the parameter file name is provided
        fprintf(stderr,"Usage: %s <param_file> [KEY VALUE ...]\n",argv[0]);
        MPI_Abort(comm,1);
    }
    
    if (rank==0){
        printf("\n##############################################\n");
        printf("\n##      PROGRAM REAL-TIME PROPAGATION       ##\n");
        printf("\n##############################################\n");
    }
    
    /* Initialization */
    ptrdiff_t i; // Declaration of a large integer (loop over dim_HS)
    int j,k;
    
    /** Parameters & Field **/
    if (rank==0){printf("\n# Parameters read and broadcasted to all ranks\n\n");}
    Field F; // Declare the structure of field parameters
    Parameters P; // Declare the structure of parameters
    Parameters P_imag; // Declare the structure of parameters of the output of the imaginary time propagation
    memset(&F,0,sizeof(Field));
    memset(&P,0,sizeof(Parameters));
    memset(&P_imag,0,sizeof(Parameters));
    Initialize_parameters_real(argv[1],&P,&P_imag,&F,argc,argv,comm); // Initialize the parameters
    
    /** Initialize integrator & field vectors **/
    Integrator Int;
    memset(&Int,0,sizeof(Integrator)); // Initialize it to zero
    Initialize_integrator(P.integrator,P.dt,&Int,comm);
    Initialize_field_vectors(&P,&F,&Int,comm);
    
    /** Initialize observables **/
    Observables Obs;
    memset(&Obs,0,sizeof(Observables)); // Initialize it to zero
    Initialize_observables(&Obs,P.observables,P.N_observables,P.dim,comm);
    FILE *file_obs=NULL;
    if (rank==0){
        file_obs=fopen("observables.real","w");
        if (!file_obs){
            perror("Error opening the file of observables");
            MPI_Abort(comm,1);
        }
        
        /** Write down the first line of observables **/
        fprintf(file_obs,"time"); // Write the time
        for (j=0;j<Obs.total_count;j++){for (k=0;k<Obs.dimensions[j];k++){if (Obs.dimensions[k]!=0){fprintf(file_obs,"\t%s",Obs.names[j]);}}} // Write the other observables
        fprintf(file_obs,"\n"); // Write the time
    }
    
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
    
    //** Meshgrid and Fourier transform **/
    MeshGrid grid; // Declare the structure of meshgrid
    memset(&grid,0,sizeof(MeshGrid)); // Initialize it to zero
    Initialize_meshgrid(&P,&grid,local_n,local_start,alloc_local,comm); // Initialize the grid
    if (grid.dim_HS!=alloc_local){
        fprintf(stderr,"ERROR: Problem with the dimension of the local Hilbert space and the local allocation.\n");
        fprintf(stderr,"dim_HS = %td \nalloc_local = %td\n",grid.dim_HS,alloc_local);
        perror("fftw MPI");
        MPI_Abort(comm,1);
    }
    
    /** Wavefunction: Memory allocation & initialization **/
    if (rank==0){printf("\n# Initialization of the wave function\n");}
    double complex *Psi=NULL;
    Psi=fftw_alloc_complex(grid.dim_HS);
    if (Psi==NULL){
        fprintf(stderr,"ERROR: memory allocation failed for Psi.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    memset(Psi,0,grid.dim_HS*sizeof(double complex));
    Read_wavefunction_MPI(&P,&P_imag,&grid,Psi,local_start,comm);
      
    /** Memory allocation and initialization for the length vs velocity gauge calculation of the observables **/
    double *alpha=NULL, *alpha_buffer=NULL;
    alpha=(double*) malloc(P.dim*sizeof(double));
    alpha_buffer=(double*) malloc(P.dim*sizeof(double));
    if (alpha==NULL || alpha_buffer==NULL){
        if (alpha) free(alpha);
        if (alpha_buffer) free(alpha_buffer);
        fprintf(stderr, "ERROR: memory allocation failed for alpha or alpha_buffer\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (j=0;j<P.dim;j++){
        alpha[j]=0.; // Initialization to 0
        alpha_buffer[j]=0.; // Initialization to 0
    }
    
    /* Propagation */
    if (rank==0){printf("\n# Real-time propagation\n");}
    clock_t start=clock();
    struct timespec start_real,end_real;
    clock_gettime(CLOCK_MONOTONIC,&start_real);
    
    double dt_cum=P.dt_save;
    int percent=0;
    if(strcmp(F.gauge,"velocity")==0){ // Integration of the wavefunction using the split-operator method in the velocity gauge
        for (ptrdiff_t itime=0;itime<P.N_time;itime++){
            /** Calculation of the desired observables & writing **/
            dt_cum+=P.dt;
            if (dt_cum>=P.dt_save){
                Field_vectors(&P,&F,P.time[itime],alpha,alpha_buffer,comm); // Compute the vector potential at time t
                Calculation_global_observables(Psi,P.dim,&grid,&Obs,alpha,comm); // Compute the observables with At=alpha (velocity gauge)
                Write_observables(file_obs,P.time[itime],&Obs,comm);
            }
            
            /** Integration for one time-step **/
            Integration_step_velocity(Psi,&F.A[itime*Int.s_integrator*P.dim],P.dim,&grid,&Int);
            
            /** Dispaly the advancement **/
            if (percent<100*itime/P.N_time){
                if (rank==0){printf("> %d%%\n",percent);}
                percent+=10;
            }
        }
    }
    else if(strcmp(F.gauge,"length")==0){ // Integration of the wavefunction using the split-operator method in the length gauge
        /** Preparation of the wavefunction in the p-representation **/
        for(i=0;i<grid.dim_HS;i++){grid.in[i]=Psi[i];}
        fftw_execute(grid.plan_forward);
        for(i=0;i<grid.dim_HS;i++){Psi[i]=grid.out[i]*sqrt(grid.fft_scaling);} // Psi in p-representation & rescaled
        
        /** Integration in p-representation **/
        for (ptrdiff_t itime=0;itime<P.N_time;itime++){
            /** Calculation of the desired observables & writing **/
            dt_cum+=P.dt;
            if (dt_cum>=P.dt_save){
                /** Wavefunction to the r-representation **/
                for(i=0;i<grid.dim_HS;i++){grid.in[i]=Psi[i];}
                fftw_execute(grid.plan_backward);
                for(i=0;i<grid.dim_HS;i++){Psi[i]=grid.out[i]*sqrt(grid.fft_scaling);} // Psi in r-representation & rescaled
                
                /** Calculation & writing **/
                for (j=0;j<P.dim;j++){alpha[j]=0.;} // Re-initialization to 0 for sanity check
                Calculation_global_observables(Psi,P.dim,&grid,&Obs,alpha,comm); // Compute the observables with vector potential 0 (length gauge)
                Write_observables(file_obs,P.time[itime],&Obs,comm);
                
                /** Wavefunction back to the p-representation **/
                for(i=0;i<grid.dim_HS;i++){grid.in[i]=Psi[i];}
                fftw_execute(grid.plan_forward);
                for(i=0;i<grid.dim_HS;i++){Psi[i]=grid.out[i]*sqrt(grid.fft_scaling);} // Psi in p-representation & rescaled
            }
            
            /** Integration for one time-step **/
            Integration_step_length(Psi,&F.E[itime*Int.s_integrator*P.dim],P.dim,&grid,&Int);
           
            /** Dispaly the advancement **/
            if (percent<100*itime/P.N_time){
                if (rank==0){printf("> %d%%\n",percent);}
                percent+=10;
            }
        }
        
        /** Wavefunction back to the r-representation **/
        for(i=0;i<grid.dim_HS;i++){grid.in[i]=Psi[i];}
        fftw_execute(grid.plan_backward);
        for(i=0;i<grid.dim_HS;i++){Psi[i]=grid.out[i]*sqrt(grid.fft_scaling);} // Psi in r-representation & rescaled
    }
    
    clock_gettime(CLOCK_MONOTONIC,&end_real);
    if (rank==0){
        printf("  Elapsed time\t\t%16.15f seconds\n",(double)(clock()-start)/CLOCKS_PER_SEC);
        printf("  Elapsed real time\t%16.15f seconds\n",(double)(end_real.tv_sec-start_real.tv_sec)+(end_real.tv_nsec-start_real.tv_nsec)/1e9);
    }
    
    if (rank==0 && file_obs!=NULL){ // Close the file of observables
        fclose(file_obs);
        file_obs=NULL;
    }
    
    /* Save data */
    if (rank==0){printf("\n# Calculation of the final observables\n");}
    for (j=0;j<P.dim;j++){alpha[j]=0.;} // Ensure the alpha is zero
    Write_final_observables(Psi,&P,&grid,&Obs,local_start,comm);
    
    /* Finalization */
    if (rank==0){printf("\n# The code ended successfully\n\n");}
    free(alpha);
    free(alpha_buffer);
    free(local_n);
    free(local_start);
    fftw_free(Psi);
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

