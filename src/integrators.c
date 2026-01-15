#include "integrators.h"    // To confirm prototypes match definitions
#include <complex.h>        // For cexp
#include <fftw3-mpi.h>      // For fftw_execute, fftw_malloc, fftw_plan, and fftw_complex types
#include <stdlib.h>         // For malloc, free, exit
#include <stdio.h>          // For fprintf, perror
#include <string.h>         // To use strncpy
#include <math.h>           // For exp (or cexp implicitly).

/*######################################*/
// INITIALIZATION
/*######################################*/
/* Initialize the integrator structure */
void Initialize_integrator(const char* integrator_name, double dt, Integrator* Int, MPI_Comm comm){ // dt step size of the integration
    int j;
    
    /* Initialization of the integrator coefficients */
    for (j=0;j<2*MAX_INTEGRATOR_STEPS;j++){
        Int->as[j]=0.;
        Int->as_scaled[j]=0.;
    }
    
    if (strcmp(integrator_name,"Verlet")==0){
        Int->s_integrator=1;
        Int->as[0]=.5;
    }
    else if (strcmp(integrator_name,"BM4")==0){
        Int->s_integrator=6;
        Int->as[0]=.0792036964311957;
        Int->as[1]=.1303114101821663;
        Int->as[2]=.2228614958676077;
        Int->as[3]=-.3667132690474257;
        Int->as[4]=.3246481886897062;
        Int->as[5]=.1096884778767498;
    }
    else if (strcmp(integrator_name,"BM6")==0){
        Int->s_integrator=10;
        Int->as[0]=.050262764400392;
        Int->as[1]=.098553683500650;
        Int->as[2]=.314960616927694;
        Int->as[3]=-.447346482695478;
        Int->as[4]=.492426372489876;
        Int->as[5]=-.425118767797691;
        Int->as[6]=.237063913978122;
        Int->as[7]=.195602488600053;
        Int->as[8]=.346358189850727;
        Int->as[9]=-.362762779254345;
    }
    else if (strcmp(integrator_name,"RKN4")==0){
        Int->s_integrator=6;
        Int->as[0]=.082984406417405;
        Int->as[1]=.162314550766866;
        Int->as[2]=.233995250731502;
        Int->as[3]=.370877414979578;
        Int->as[4]=-.409933719901926;
        Int->as[5]=.059762097006575;
    }
    else{
        printf("\nUnknown integrator: %s\n",integrator_name);
        perror("Error with the integrator name");
        MPI_Abort(comm,1);
    }
    
    for (j=0;j<Int->s_integrator;j++){Int->as[Int->s_integrator+j]=Int->as[Int->s_integrator-(j+1)];}
    for (j=0;j<2*Int->s_integrator;j++){Int->as_scaled[j]=Int->as[j]*dt;}
    if (Int->s_integrator>MAX_INTEGRATOR_STEPS){
        printf("\nThe integrator steps is too large compared with the limit\n");
        printf("\nChange the value of MAX_INTEGRATOR_STEPS in integrators.c\n");
        perror("Error with the integrator.");
        MPI_Abort(comm,1);
    }
}

/* Initialization of field vectors */
void Initialize_field_vectors(const Parameters* P, Field* F, const Integrator* Int, MPI_Comm comm){
    ptrdiff_t i;
    int j,k;
    int ind;
    double t;
    
    /* Vecotr potential and electric field */
    F->E=(double*) malloc(P->dim*P->N_time*Int->s_integrator*sizeof(double)); // Note that the size of F->E is N_time*s_integrator because it is accessed in integration_step
    F->A=(double*) malloc(P->dim*P->N_time*Int->s_integrator*sizeof(double)); // Note that the size of F->E is N_time*s_integrator because it is accessed in integration_step
    if (F->E==NULL || F->A==NULL){
        if (F->E) free(F->E);
        if (F->A) free(F->A);
        fprintf(stderr, "ERROR: memory allocation failed for F->E or F->A in Initialize_parameters.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (i=0;i<P->dim*P->N_time*Int->s_integrator;i++){
        F->E[i]=0.;
        F->A[i]=0.;
    }
    for (i=0;i<P->N_time;i++){
        t=P->time[i]+Int->as_scaled[0];
        ind=i*P->dim*Int->s_integrator;
        for (j=0;j<Int->s_integrator;j++){
            Field_vectors(P,F,t,&F->A[ind+j*P->dim],&F->E[ind+j*P->dim],comm); // Initialization of the vectors
            if (j<Int->s_integrator-1){t+=Int->as_scaled[2*j+1]+Int->as_scaled[2*j+2];}
        }
    }
    
    /* Save the laser electric field and vector potential */
    FILE *file_Et=NULL, *file_At=NULL;
    file_Et=fopen("electric_field.real","w");
    file_At=fopen("vector_potential.real","w");
    if (!file_Et || !file_At){
        perror("Error opening the files of the vector potential and the laser electric field.");
        MPI_Abort(comm,1);
    }
    
    for (i=0;i<P->N_time;i++){
        t=P->time[i]+Int->as_scaled[0];
        ind=i*P->dim*Int->s_integrator;
        for (j=0;j<Int->s_integrator;j++){
            fprintf(file_Et,"%16.15f ",t);
            fprintf(file_At,"%16.15f ",t);
            for (k=0;k<P->dim;k++){
                fprintf(file_Et,"%16.15f ",F->E[ind+j*P->dim+k]);
                fprintf(file_At,"%16.15f ",F->A[ind+j*P->dim+k]);
            }
            fprintf(file_Et,"\n");
            fprintf(file_At,"\n");
            if (j<Int->s_integrator-1){t+=Int->as_scaled[2*j+1]+Int->as_scaled[2*j+2];}
        }
    }
    fclose(file_Et);
    fclose(file_At);
}


/*######################################*/
// IMAGINARY-TIME PROP
/*######################################*/
void Integration_step_imag(double complex *Psi, const MeshGrid *grid, const Integrator* Int){ // One integration step of imaginary-time step
    ptrdiff_t i;
    int ia;
    double Arg;
    
    /* Propagation through the split-operator method */
    for (ia=0;ia<Int->s_integrator;ia++){ // chi & chi star applied at each step of the integrator
        // Potential energy
        for(i=0;i<grid->dim_HS;i++){
            Arg=-Int->as_scaled[2*ia]*grid->Vpot[i];
            if (Arg>MAX_EXP_ARG){Arg=MAX_EXP_ARG;}
            grid->in[i]=exp(Arg)*Psi[i];
        }
        fftw_execute(grid->plan_forward);
        // Kinetic energy & Incrementation of time *time+=Int->as_scaled[2*ia]+Int->as_scaled[2*ia+1];
        for(i=0;i<grid->dim_HS;i++){
            Arg=-(Int->as_scaled[2*ia]+Int->as_scaled[2*ia+1])*grid->Kin[i];
            if (Arg>MAX_EXP_ARG){Arg=MAX_EXP_ARG;}
            grid->in[i]=exp(Arg)*grid->out[i];
        }
        fftw_execute(grid->plan_backward);
        // Potential energy
        for(i=0;i<grid->dim_HS;i++){
            Arg=-Int->as_scaled[2*ia+1]*grid->Vpot[i];
            if (Arg>MAX_EXP_ARG){Arg=MAX_EXP_ARG;}
            Psi[i]=exp(Arg)*grid->out[i]*grid->fft_scaling;
        }
    }
}

/*######################################*/
// IMAGINARY-TIME PROP - LENGTH GAUGE
/*######################################*/
// One integration step of time step h in length gauge. Psi is in the p-representation.
// Dimension of the configuration space: dim
void Integration_step_length(double complex *Psi, const double *Et, const int dim, const MeshGrid *grid, const Integrator* Int){
    ptrdiff_t i;
    int ia,j;
    
    /* Propagation through the split-operator method */
    for (ia=0;ia<Int->s_integrator;ia++){ // chi & chi star applied at each step of the integrator
        // Kinetic energy & Incrementation of time *time+=Int->as_scaled[2*ia];
        for (i=0;i<grid->dim_HS;i++){grid->in[i]=cexp(-I*Int->as_scaled[2*ia]*grid->Kin[i])*Psi[i];}
        fftw_execute(grid->plan_backward);
        // Potential energy
        for(i=0;i<grid->dim_HS;i++){
            double Wt=0.;
            for(j=0;j<dim;j++){
                ptrdiff_t absolute_index=grid->index[j*grid->dim_HS+i];
                Wt+=grid->r[absolute_index]*Et[ia*dim+j];
            }
            double Arg=-(Int->as_scaled[2*ia]+Int->as_scaled[2*ia+1])*grid->Vcap[i];
            if (Arg>MAX_EXP_ARG){Arg=MAX_EXP_ARG;}
            grid->in[i]=cexp(Arg-I*(Int->as_scaled[2*ia]+Int->as_scaled[2*ia+1])*(grid->Vpot[i]+Wt))*grid->out[i];
        }
        fftw_execute(grid->plan_forward);
        // Kinetic energy & Incrementation of time *time+=Int->as_scaled[2*ia+1];
        for(i=0;i<grid->dim_HS;i++){Psi[i]=cexp(-I*Int->as_scaled[2*ia+1]*grid->Kin[i])*grid->out[i]*grid->fft_scaling;}
    }
}

/*######################################*/
// IMAGINARY-TIME PROP - VELOCITY GAUGE
/*######################################*/
// One integration step of time step h in velocity gauge. Psi is in the r-representation.
// Dimension of the configuration space: dim
void Integration_step_velocity(double complex *Psi, const double *At, const int dim, const MeshGrid *grid, const Integrator* Int){
    ptrdiff_t i;
    int ia,j;
    double Arg;
    
    /* Propagation through the split-operator method */
    for (ia=0;ia<Int->s_integrator;ia++){ // chi & chi star applied at each step of the integrator
        // Potential energy & Incrementation of time *time+=Int->as_scaled[2*ia];
        for(i=0;i<grid->dim_HS;i++){
            Arg=-Int->as_scaled[2*ia]*grid->Vcap[i];
            if (Arg>MAX_EXP_ARG){Arg=MAX_EXP_ARG;}
            grid->in[i]=cexp(Arg-I*Int->as_scaled[2*ia]*grid->Vpot[i])*Psi[i];
        }
        fftw_execute(grid->plan_forward);
        // Kinetic energy
        for(i=0;i<grid->dim_HS;i++){
            double Wt=0.;
            for(j=0;j<dim;j++){
                ptrdiff_t absolute_index=grid->index[j*grid->dim_HS+i];
                Wt+=(grid->pr[absolute_index]+.5*At[ia*dim+j])*At[ia*dim+j];
            }
            grid->in[i]=cexp(-I*(Int->as_scaled[2*ia]+Int->as_scaled[2*ia+1])*(grid->Kin[i]+Wt))*grid->out[i];
        }
        fftw_execute(grid->plan_backward);
        // Potential energy & Incrementation of time *time+=Int->as_scaled[2*ia+1];
        for(i=0;i<grid->dim_HS;i++){
            Arg=-Int->as_scaled[2*ia+1]*grid->Vcap[i];
            if (Arg>MAX_EXP_ARG){Arg=MAX_EXP_ARG;}
            Psi[i]=cexp(Arg-I*Int->as_scaled[2*ia+1]*grid->Vpot[i])*grid->out[i]*grid->fft_scaling;
        }
    }
}

/*######################################*/
// FINALIZATION
/*######################################*/
void Finalize_integrator(Integrator* Int){
    if (Int==NULL) {return;} // Check if Int is allocated
}
