
/* Local Headers */
#include "meshgrid.h"
#include "parameters.h" // Assuming this defines the Parameters struct

/* Standard C/MPI/FFTW Headers */
#include <stdio.h> // For fprintf, printf, perror
#include <stdlib.h> // For exit, malloc, free
#include <string.h> // For strcmp
#include <math.h> // For sqrt, M_PI (if defined here)
#include <omp.h> // To set the number of threads for fftw
// #define M_PI 3.14159265358979323846 // Or ensure <math.h> is included and defines it

/*######################################*/
// INITIALIZE MESHGRID
/*######################################*/
void Initialize_meshgrid(const Parameters *P, MeshGrid *grid, const ptrdiff_t *local_n, const ptrdiff_t *local_start, ptrdiff_t alloc_local, MPI_Comm comm){ // Initialize the grid vectors & the fft plans
    ptrdiff_t i;
    int j;
    
    /* Initialize the relevant local number of grid points */
    /** Infinitesimal volumes **/
    grid->dNr=1.;
    grid->dNp=1.;
    for (j=0;j<P->dim;j++){
        grid->dNr*=P->dr[j]; // Infinitesimal volume in position
        grid->dNp*=2.*M_PI/(double)(P->dr[j]*P->Nr_global[j]); // Infinitesimal volume in momentum
    }
        
    /** Parameters related to the dimension **/
    grid->Nr_local=(ptrdiff_t*) malloc(P->dim*sizeof(ptrdiff_t));
    grid->Nc=(ptrdiff_t*) malloc((P->dim+1)*sizeof(ptrdiff_t));
    grid->Nm=(ptrdiff_t*) malloc((P->dim+1)*sizeof(ptrdiff_t));
    if (grid->Nr_local==NULL || grid->Nc==NULL || grid->Nm==NULL){
        if (grid->Nr_local) free(grid->Nr_local);
        if (grid->Nc) free(grid->Nc);
        if (grid->Nm) free(grid->Nm);
        fprintf(stderr, "ERROR: memory allocation failed in meshgrid.c\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    grid->Nr_local[0]=local_n[0]; // Distributed dimension
    grid->offset_HS=local_start[0];
    for (j=1;j<P->dim;j++){ // Not distributed dimension
        grid->Nr_local[j]=P->Nr_global[j];
        grid->offset_HS*=P->Nr_global[j];
    }
    grid->Nc[0]=0;
    grid->Nm[0]=1;
    for (j=0;j<P->dim;j++){
        grid->Nc[j+1]=grid->Nc[j]+grid->Nr_local[j];
        grid->Nm[j+1]=grid->Nm[j]*grid->Nr_local[j]; // Such that Nm[dim]==dim_HS
    }
    grid->dim_HS=grid->Nm[P->dim]; // Local Hilbert space dimension
    
    /* Grid in position and momentum */
    /** Memory allocation **/
    grid->r=(double*) malloc(grid->Nc[P->dim]*sizeof(double));
    grid->pr=(double*) malloc(grid->Nc[P->dim]*sizeof(double));
    grid->Vpot=(double*) malloc(grid->dim_HS*sizeof(double));
    grid->Vcap=(double*) malloc(grid->dim_HS*sizeof(double));
    grid->Kin=(double*) malloc(grid->dim_HS*sizeof(double));
    if (!grid->r || !grid->pr || !grid->Vpot || !grid->Vcap || !grid->Kin){
        if (grid->r!=NULL) free(grid->r);
        if (grid->pr!=NULL) free(grid->pr);
        if (grid->Vpot!=NULL) free(grid->Vpot);
        if (grid->Vcap!=NULL) free(grid->Vcap);
        if (grid->Kin!=NULL) free(grid->Kin);
        fprintf(stderr, "ERROR: memory allocation failed for the grid in Initialize_meshgrid.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    /** Initialization of the grid **/
    for (j=0;j<P->dim;j++){
        double dr=P->dr[j];
        ptrdiff_t N_local=grid->Nr_local[j];
        ptrdiff_t N_global=P->Nr_global[j];
        double L=dr*N_global;
        double dp=2.*M_PI/L;
        ptrdiff_t start_offset=(j==0)?local_start[0]:0;
        for (i=0;i<N_local;i++){ // Grids along the j-th direction
            ptrdiff_t global_i=i+start_offset; // Define the global index
            grid->r[grid->Nc[j]+i]=-.5*L+global_i*dr; // Position coordinate
            if (global_i<N_global/2){grid->pr[grid->Nc[j]+i]=global_i*dp;} // Momentum coordinate
            else{grid->pr[grid->Nc[j]+i]=(global_i-N_global)*dp;}
        }
    }
    
    /** Define the full grid coordinate and momentum **/
    ptrdiff_t *index;
    index=(ptrdiff_t*) malloc(P->dim*sizeof(ptrdiff_t));
    if (!index){
        fprintf(stderr, "ERROR: memory allocation failed for the grid in Initialize_meshgrid.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    grid->index=(ptrdiff_t*) malloc(P->dim*grid->dim_HS*sizeof(ptrdiff_t));
    if (!grid->index){
        fprintf(stderr, "ERROR: memory allocation failed for the grid in Initialize_meshgrid.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (i=0;i<grid->dim_HS;i++){
        Recover_indices(index,i,P->dim,grid->Nr_local);
        for (j=0;j<P->dim;j++){grid->index[j*grid->dim_HS+i]=grid->Nc[j]+index[j];}
    }
    
    /** Initialization of the physical quantities **/
    double *ri, *pi, r;
    ri=(double*) malloc(P->dim*sizeof(double));
    pi=(double*) malloc(P->dim*sizeof(double));
    if (!ri || !pi){
        if (ri) free(ri);
        if (pi) free(pi);
        fprintf(stderr, "ERROR: memory allocation failed for the grid in Initialize_meshgrid.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    for (i=0;i<grid->dim_HS;i++){
        grid->Kin[i]=0.; // Initialization kinetic energy
        grid->Vcap[i]=0.; // Initialization complex absorbing potential
        r=0.;
        for (j=0;j<P->dim;j++){
            ri[j]=grid->r[grid->index[j*grid->dim_HS+i]];
            pi[j]=grid->pr[grid->index[j*grid->dim_HS+i]];
            grid->Kin[i]+=.5*pi[j]*pi[j]; // Increment of the kinetic energy
            r+=ri[j]*ri[j];
        
            // Initialization of the complex absorbing potential (CAP)
            double L=.5*P->Nr_global[j]*P->dr[j]-P->cap[1];
            if (fabs(ri[j])>=L){grid->Vcap[i]+=P->cap[0]*(fabs(ri[j])-L)*(fabs(ri[j])-L);}
        }
        r=sqrt(r);
        
        // Initialization of the potential
        grid->Vpot[i]=0.;
        if (strcmp(P->potential,"H_1D")==0){grid->Vpot[i]=-1./(double)sqrt(r*r+2.);}
        else if (strcmp(P->potential,"H_3D")==0){grid->Vpot[i]=-1./(double)sqrt(r*r+1e-4);}
        else if (strcmp(P->potential,"H_2D")==0){grid->Vpot[i]=-1./(double)sqrt(r*r+.7396);}
        else if (strcmp(P->potential,"Ne_2D")==0){grid->Vpot[i]=-(1.+9.*exp(-r*r))/(double)sqrt(r*r+2.88172);}
        else if (strcmp(P->potential,"Ar_2D")==0){grid->Vpot[i]=-(1.+17.*exp(-r*r))/(double)sqrt(r*r+.91205);}
        else if (strcmp(P->potential,"Kr_2D")==0){grid->Vpot[i]=-(1.+35.*exp(-r*r))/(double)sqrt(r*r+.7875);}
        else if (strcmp(P->potential,"Xe_2D")==0){grid->Vpot[i]=-(1.+53.*exp(-r*r))/(double)sqrt(r*r+.52592);}
        else if (strcmp(P->potential,"Ne_short_2D")==0){grid->Vpot[i]=-2.7459*exp(-.5*r*r/1.96);}
        else{
            printf("\nUnknown potential: %s\n",P->potential);
            perror("Error with the potential name");
            MPI_Abort(comm,1);
        }
    }
    free(ri);
    free(pi);
        
    /* Fourier transforms between position and momentum representations */
    fftw_init_threads();
    int num_threads=omp_get_max_threads(); // Get the number of maximum threads
    fftw_plan_with_nthreads(num_threads); // Set the environment for parallel FFTW
    grid->fft_scaling=1.;
    for (j=0;j<P->dim;j++){grid->fft_scaling/=(double)P->Nr_global[j];}
    grid->in=fftw_alloc_complex(grid->dim_HS); // Input of the fftw
    grid->out=fftw_alloc_complex(grid->dim_HS); // Output of the fftw
    grid->plan_forward=fftw_mpi_plan_dft(P->dim,P->Nr_global,grid->in,grid->out,comm,FFTW_FORWARD,FFTW_MEASURE); // Forward Fourier transform
    grid->plan_backward=fftw_mpi_plan_dft(P->dim,P->Nr_global,grid->in,grid->out,comm,FFTW_BACKWARD,FFTW_MEASURE); // Backward Fourier transform
    if (!grid->in || !grid->out || !grid->plan_forward || !grid->plan_backward){
        if (grid->in) fftw_free(grid->in);
        if (grid->out) fftw_free(grid->out);
        if (grid->plan_forward) fftw_destroy_plan(grid->plan_forward);
        if (grid->plan_backward) fftw_destroy_plan(grid->plan_backward);
        fprintf(stderr, "ERROR: memory allocation failed for the fftw in Initialize_meshgrid.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    memset(grid->in,0,grid->dim_HS*sizeof(double complex));
    memset(grid->out,0,grid->dim_HS*sizeof(double complex));
}

/*######################################*/
// FINALIZATION / CLEANUP
/*######################################*/
void Finalize_meshgrid(MeshGrid *grid){
    if (grid==NULL) {return;} // Check if grid is allocated
    if (grid->r!=NULL) {free(grid->r); grid->r=NULL;}
    if (grid->pr!=NULL) {free(grid->pr); grid->pr=NULL;}
    if (grid->index!=NULL) {free(grid->index); grid->index=NULL;}
    if (grid->Vpot!=NULL) {free(grid->Vpot); grid->Vpot=NULL;}
    if (grid->Vcap!=NULL) {free(grid->Vcap); grid->Vcap=NULL;}
    if (grid->Kin!=NULL) {free(grid->Kin); grid->Kin=NULL;}
    if (grid->in!=NULL) {fftw_free(grid->in); grid->in=NULL;}
    if (grid->out!=NULL) {fftw_free(grid->out); grid->out=NULL;}
    if (grid->plan_forward!=NULL) {fftw_destroy_plan(grid->plan_forward);}
    if (grid->plan_backward!=NULL) {fftw_destroy_plan(grid->plan_backward);}
    if (grid->Nr_local!=NULL) {free(grid->Nr_local); grid->Nr_local=NULL;}
    if (grid->Nc!=NULL) {free(grid->Nc); grid->Nc=NULL;}
    if (grid->Nm!=NULL) {free(grid->Nm); grid->Nm=NULL;}
}

/*######################################*/
// RECOVER INDICES IN ROW-MAJOR
/*######################################*/
// Recover the indexes for each dimension for an index N<dim_HS and arrange them into an array index of size dim. Knowing that the arrays are arranged as N = index[dim-1] + index[dim-2]*Nr[dim-1] + ... + index[0]*Nr[dim-1]*Nr[dim-2]*...*Nr[1].
void Recover_indices(ptrdiff_t *index, ptrdiff_t N, const int dim, const ptrdiff_t *Nr){
    int j;
    for (j=dim-1;j>=0;j--){ // Recovering row-major indexing (Nr[0] is the slowest-varying index)
        index[j]=N%Nr[j];
        N/=Nr[j];
    }
}
