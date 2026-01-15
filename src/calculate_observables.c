
#include "calculate_observables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3-mpi.h>

/*######################################*/
// INITIALIZATION
/*######################################*/
void Initialize_observables(Observables* Obs, char **observables_name, int observables_number, int dim, MPI_Comm comm){ // Initialize an observable structure with their name and values for dimension dim
    int j,k;
    
    /* Initialize the observables according to the parameters */
    /** Reading **/
    Obs->total_count=observables_number;
    
    /** Memory allocation **/
    Obs->dimensions=(int*) malloc(Obs->total_count*sizeof(int));
    Obs->offsets=(int*) malloc(Obs->total_count*sizeof(int));
    Obs->names=(char**) malloc(Obs->total_count*sizeof(char*));
    if (!Obs->names || !Obs->dimensions || !Obs->offsets){
        if (Obs->names!=NULL) {free(Obs->names);}
        if (Obs->dimensions!=NULL) {free(Obs->dimensions);}
        if (Obs->offsets!=NULL) {free(Obs->offsets);}
        perror("ERROR: Memory allocation failed in Initialize_observables");
        MPI_Abort(comm,1);
    }
    for (j=0;j<Obs->total_count;j++){
        Obs->names[j]=(char*) malloc(MAX_OBSERVABLE_CHAR*sizeof(char));
        if (Obs->names[j]==NULL) {
            perror("ERROR: Memory allocation failed for one of the observable strings");
            for (k=0;k<j;k++){free(Obs->names[k]);}
            free(Obs->names); // Free the array of pointers itself
            MPI_Abort(comm,1);
        }
    }
    
    /** Copy the names of the observables **/
    for (j=0;j<Obs->total_count;j++) {
        if (observables_name[j]==NULL) {continue;} // Check for a valid source pointer (just in case observables_name was poorly initialized)
        strncpy(Obs->names[j],observables_name[j],MAX_OBSERVABLE_CHAR-1); // Copy the string data safely from the source array to the destination buffer
        Obs->names[j][MAX_OBSERVABLE_CHAR-1]='\0'; // Ensure null termination for safety, as strncpy doesn't always null-terminate
    }
    
    /** Initialize the dimension & calculate the total memory required for all observables **/
    for (j=0;j<Obs->total_count;j++){
        if (strcmp(Obs->names[j],"PMD")==0 || strcmp(Obs->names[j],"PED")==0 || strcmp(Obs->names[j],"wavefunction")==0 || strcmp(Obs->names[j],"none")==0){Obs->dimensions[j]=0;} // Takes no room for values
        else if (strcmp(Obs->names[j],"Lz")==0 || strcmp(Obs->names[j],"energy")==0){Obs->dimensions[j]=1;} // Takes 1 value
        else if (strcmp(Obs->names[j],"dipole_r")==0 || strcmp(Obs->names[j],"dipole_v")==0){Obs->dimensions[j]=dim;} // Takes DIMENSION values
        else {
            printf("Unknown observable: %s. Skipping.\n",Obs->names[j]);
            Obs->dimensions[j]=0;
        }
    }
    
    if (Obs->total_count>0){Obs->offsets[0]=0;} // The offset of the first observable (j=0) is always 0
    for (j=1;j<Obs->total_count;j++){Obs->offsets[j]=Obs->offsets[j-1]+Obs->dimensions[j-1];} // Calculate subsequent offsets using the corrected array name
    if (Obs->total_count>0) {Obs->total_values=Obs->offsets[Obs->total_count-1]+Obs->dimensions[Obs->total_count-1];}
    else {Obs->total_values=0;}
    if (Obs->total_values==0){return;}
    
    /* Arrays for the values of observables */
    /** Memory allocation **/
    Obs->local_results=(double complex*) malloc(Obs->total_values*sizeof(double complex));
    Obs->global_results=(double complex*) malloc(Obs->total_values*sizeof(double complex));
    Obs->values=(double*) malloc(Obs->total_values*sizeof(double));
    if (!Obs->local_results || !Obs->global_results || !Obs->values){
        perror("ERROR: Memory allocation failed in Initialized_observables");
        if (Obs->local_results!=NULL) {free(Obs->local_results);}
        if (Obs->global_results!=NULL) {free(Obs->global_results);}
        if (Obs->values!=NULL) {free(Obs->values);}
        MPI_Abort(comm,1);
    }
    
    /** Initialization **/
    for (j=0;j<Obs->total_values;j++){
        Obs->local_results[j]=0.;
        Obs->global_results[j]=0.;
        Obs->values[j]=0.;
    }
}

/*######################################*/
// GLOBAL OBSERVABLES
/*######################################*/
/* Global energy moments */
void Calculation_global_energies(double *Ej_global, const int N, const double complex *Psi, const MeshGrid *grid, MPI_Comm comm){ // Computes the N moments of the energy operator
    int j;
    
    /* Check for trivial case (optional but good practice) */
     if (N<=0) return;
    
    /* Memory allocation */
    double complex *Ej_local=NULL;
    double complex *Ej_sum=NULL;
    Ej_local=(double complex*) malloc(N*sizeof(double complex));
    Ej_sum=(double complex*) malloc(N*sizeof(double complex));
    if (Ej_local==NULL || Ej_sum==NULL){
        if (Ej_local!=NULL) free(Ej_local);
        if (Ej_sum!=NULL) free(Ej_sum);
        fprintf(stderr, "ERROR: memory allocation failed in Calculation_global_energies.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    /* Calculation of the global energies from the local ones */
    Calculation_local_energies(Ej_local,N,Psi,grid,comm); // Calculation of the local energies
    MPI_Allreduce(Ej_local,Ej_sum,N,MPI_C_DOUBLE_COMPLEX,MPI_SUM,comm); // Global summation of E^j all at once
    for (j=0;j<N;j++){Ej_global[j]=creal(Ej_sum[j]);} // Take only the real part
    
    /* Cleanup */
    free(Ej_local);
    free(Ej_sum);
}

/* Global observables from Psi in momentum representation */
void Calculation_global_observables(const double complex *Psi, int dim, const MeshGrid *grid, const Observables *Obs, double *alpha, MPI_Comm comm){
    if (Obs->total_count==0){return;} // Check if it is necessary to run the function
    
    int j;
    int offset;
    
    /* Calculate the local observables */
    for (j=0;j<Obs->total_count;j++){
        offset=Obs->offsets[j];
        if (strcmp(Obs->names[j],"Lz")==0){
            if (dim<=1){Obs->local_results[offset]=0.;}
            else{Calculation_local_Lz(&Obs->local_results[offset],Psi,grid,alpha,comm);}
        }
        else if (strcmp(Obs->names[j],"energy")==0){Calculation_local_energies(&Obs->local_results[offset],1,Psi,grid,comm);}
        else if (strcmp(Obs->names[j],"dipole_r")==0){Calculation_local_dipole_r(&Obs->local_results[offset],dim,Psi,grid);}
        else if (strcmp(Obs->names[j],"dipole_v")==0){Calculation_local_dipole_v(&Obs->local_results[offset],dim,Psi,alpha,grid);}
    }
    
    /* Calculation of the global observables all at one */
    MPI_Allreduce(Obs->local_results,Obs->global_results,Obs->total_values,MPI_C_DOUBLE_COMPLEX,MPI_SUM,comm); // Perform the reduction once on the entire contiguous buffer
    for (j=0;j<Obs->total_values;j++){Obs->values[j]=creal(Obs->global_results[j]);} // Take only the real part for the values (only Hermitian operators)
}

/*######################################*/
// LOCAL OBSERVABLES
/*######################################*/
/* Local energy moments */
void Calculation_local_energies(double complex *Ej_local, const int N, const double complex *Psi, const MeshGrid *grid, MPI_Comm comm){
    ptrdiff_t i;
    int j;
    
    /* Memory allocation */
    double complex *H_Psi=NULL;
    H_Psi=fftw_alloc_complex(grid->dim_HS);
    if (H_Psi==NULL){
        fprintf(stderr, "ERROR: memory allocation failed in Calculation_global_energies.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    /* Calculation of the energy moments */
    for (i=0;i<grid->dim_HS;i++){H_Psi[i]=Psi[i];} // Initialization
    for (j=0;j<N;j++){
        Ej_local[j]=0.+0.*I;
        for (i=0;i<grid->dim_HS;i++){grid->in[i]=H_Psi[i];}
        fftw_execute(grid->plan_forward);
        for (i=0;i<grid->dim_HS;i++){grid->in[i]=grid->Kin[i]*grid->out[i];}
        fftw_execute(grid->plan_backward);
        for (i=0;i<grid->dim_HS;i++){
            H_Psi[i]=grid->out[i]*grid->fft_scaling+grid->Vpot[i]*H_Psi[i];
            Ej_local[j]+=conj(Psi[i])*H_Psi[i]*grid->dNr;
        }
    }
    
    /* Cleanup */
    fftw_free(H_Psi);
}

/* Local angular momentum Lz (ONLY FOR DIMENSION>=2) */
void Calculation_local_Lz(double complex *Lz, const double complex *Psi, const MeshGrid *grid, double *alpha, MPI_Comm comm){ // Calculation of the magnetic quantum number Lz=x*py-y*px of the system
    ptrdiff_t i;
    ptrdiff_t absolute_index;
    const ptrdiff_t X_OFFSET=0*grid->dim_HS;
    const ptrdiff_t Y_OFFSET=1*grid->dim_HS;
    double Ax=alpha[0];
    double Ay=alpha[1];
    
    /* Initialization of the wave function */
    double complex *pPsi=NULL;
    pPsi=fftw_alloc_complex(grid->dim_HS);
    if (!pPsi){
        fprintf(stderr, "ERROR: memory allocation failed in Calculation_local_Lz.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    /* Evaluation of the magnetic quantum number */
    *Lz=0.+I*0.;
    
    /** FFT[Psi] **/
    for (i=0;i<grid->dim_HS;i++){grid->in[i]=Psi[i];}
    fftw_execute(grid->plan_forward);
    for (i=0;i<grid->dim_HS;i++){pPsi[i]=grid->out[i];}
    
    /** Evaluation of <psi|x*(py+Ay[t])|psi> **/
    for (i=0;i<grid->dim_HS;i++){
        absolute_index=grid->index[Y_OFFSET+i]; // (y,py) index
        grid->in[i]=(grid->pr[absolute_index]+Ay)*pPsi[i]; // (py+Ay[t])*FFT[Psi]
    }
    fftw_execute(grid->plan_backward); // grid->out=FFT-1[py*FFT[Psi]]
    for (i=0;i<grid->dim_HS;i++){
        absolute_index=grid->index[X_OFFSET+i]; // (x,px) index
        *Lz+=conj(Psi[i])*grid->r[absolute_index]*grid->out[i];
    }
    
    /** Evaluation of <psi|y*(px+Ax[t])|psi> **/
    for (i=0;i<grid->dim_HS;i++){
        absolute_index=grid->index[X_OFFSET+i]; // (x,px) index
        grid->in[i]=(grid->pr[absolute_index]+Ax)*pPsi[i]; // (px+Ax[t])*FFT[Psi]
    }
    fftw_execute(grid->plan_backward); // grid->out=FFT-1[px*FFT[Psi]]
    for (i=0;i<grid->dim_HS;i++){
        absolute_index=grid->index[Y_OFFSET+i]; // (y,py) index
        *Lz-=conj(Psi[i])*grid->r[absolute_index]*grid->out[i];
    }
    
    /** Apply the scaling **/
    *Lz*=grid->fft_scaling*grid->dNr;
    
    /* Free variables */
    fftw_free(pPsi);
}

/* Local dipole in position */
void Calculation_local_dipole_r(double complex *dipole, int dim, const double complex *Psi, const MeshGrid *grid){ // Calculation of the dipole d=<psi|r|psi> with wave function in position representation
    ptrdiff_t i;
    int j;
    ptrdiff_t absolute_index;
    
    /* Evaluation of the dipole vector */
    for (j=0;j<dim;j++){
        dipole[j]=0.+I*0.;
        for (i=0;i<grid->dim_HS;i++){
            absolute_index=grid->index[j*grid->dim_HS+i];
            dipole[j]+=conj(Psi[i])*grid->r[absolute_index]*Psi[i]*grid->dNr;
        }
    }
}

/* Local dipole in momentum */
void Calculation_local_dipole_v(double complex *dipole, int dim, const double complex *Psi, double *alpha, const MeshGrid *grid){ // Calculation of the dipole d=<psi|(pr+At[t])|psi> with wave function in position representation
    ptrdiff_t i;
    int j;
    ptrdiff_t absolute_index;
    
    /* Wave function to momentum representation */
    for (i=0;i<grid->dim_HS;i++){grid->in[i]=Psi[i];}
    fftw_execute(grid->plan_forward);
    
    /* Evaluation of the dipole vector */
    for (j=0;j<dim;j++){
        dipole[j]=0.+I*0.;
        for (i=0;i<grid->dim_HS;i++){
            absolute_index=grid->index[j*grid->dim_HS+i];
            dipole[j]+=conj(grid->out[i])*(grid->pr[absolute_index]+alpha[j])*grid->out[i]*grid->dNp*grid->fft_scaling;
        }
    }
}

/*######################################*/
// FINALIZATION
/*######################################*/
void Finalize_observables(Observables* Obs){
    if (Obs==NULL) {return;} // Check if Obs is allocated
    if (Obs->dimensions!=NULL) {free(Obs->dimensions); Obs->dimensions=NULL;}
    if (Obs->offsets!=NULL) {free(Obs->offsets); Obs->offsets=NULL;}
    if (Obs->local_results!=NULL) {free(Obs->local_results); Obs->local_results=NULL;}
    if (Obs->global_results!=NULL) {free(Obs->global_results); Obs->global_results=NULL;}
    if (Obs->values!=NULL) {free(Obs->values); Obs->values=NULL;}
    int j;
    if (Obs->names!=NULL) {
        for (j=0;j<Obs->total_count;j++) {
            if (Obs->names[j]!=NULL) {free(Obs->names[j]); Obs->names[j]=NULL;}
        }
    }
    free(Obs->names);
    Obs->names=NULL;
}
