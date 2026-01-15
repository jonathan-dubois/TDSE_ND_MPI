#include "parameters.h"     // To confirm prototypes match definitions
#include <complex.h>        // For cabs, sqrt
#include <stdlib.h>         // For malloc, free, exit
#include <string.h>         // For strcmp, strchr
#include <math.h>           // For sqrt, fabs, sin, cos, pow, M_PI, and exp

/*######################################*/
// INITIALIZATION
/*######################################*/
/* Initialization of the parameters in the imaginary-time propagation */
void Initialize_parameters_imag(const char* filename, Parameters* P, int argc, char *argv[], MPI_Comm comm) {
    int j;
    char format[20], key[32];
    
    /* List of known parameters for the params.imag */
    const char *KnownKeys_imag[]={
        "OUTPUT_IMAG",
        "DIMENSION",
        "DR","NR",
        "N_STATES",
        "POTENTIAL",
        "INTEGRATOR","DT","TINTEGRATION",
        NULL};
    
    /* Determine the rank & size */
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    
    /* Provide default parameters */
    // Note that only the dimension MUST BE specified
    /** The output imag file **/
    snprintf(P->output_imag,MAX_OUTPUT_FILES_CHAR,"output.imag");
    
    /** Number of states to compute **/
    P->N_states=1;
    
    /** Potential **/
    snprintf(P->potential,MAX_POTENTIAL_CHAR,"H_3D");
    
    /** Integrator **/
    snprintf(P->integrator,MAX_INTEGRATOR_CHAR,"BM4");
    P->dt=.1;
    P->Tintegration=100.;
    
    /* Set the dimension and appropriate memory allocations */
    int dim=Find_dimension_in_file_MPI(filename,comm);
    Allocate_dim_arrays(P,dim,comm);
    
    /* File reading and local check (only on rank 0) */
    FILE* file=NULL;
    int local_error=0;
    if (rank==0){
        file=fopen(filename,"r"); // Open the file
        if (!file){
            perror("Error opening parameter file");
            MPI_Abort(comm,1);
        }
        
        while (fscanf(file,"%31s",key)!=EOF){ // Read the file
            if (strcmp(key,"DIMENSION")==0){
                if (Expected_arguments(file,"DIMENSION",1,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                fscanf(file,"%31s",key);
            }
            else if (strcmp(key,"NR")==0){
                if (Expected_arguments(file,"NR",P->dim,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                for (j=0;j<P->dim;j++){
                    if(fscanf(file,"%td",&P->Nr_global[j])!=1){
                        perror("The number of grid points is not completely determined");
                        MPI_Abort(comm,1);
                    }
                    if ((P->Nr_global[j]!=0 && (P->Nr_global[j] & (P->Nr_global[j]-1))!=0)){
                        printf("\nThe number of points on the grid must be a power of 2\n");
                        perror("Error with grid points number");
                        MPI_Abort(comm,1);
                    }
                }
            }
            else if (strcmp(key,"DR")==0){
                if (Expected_arguments(file,"DR",P->dim,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                for (j=0;j<P->dim;j++){
                    if(fscanf(file,"%lf",&P->dr[j])!=1){
                        perror("The grid spacing is not completely determined");
                        MPI_Abort(comm,1);
                    }
                }
            }
            else if (strcmp(key,"N_STATES")==0){
                if (Expected_arguments(file,"N_STATES",1,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                if (fscanf(file,"%d",&P->N_states)!=1){
                    perror("The number of states is undetermined");
                    MPI_Abort(comm,1);
                }
            }
            else if (strcmp(key,"DT")==0){
                if (Expected_arguments(file,"DT",1,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                if (fscanf(file,"%lf",&P->dt)!=1){
                    perror("The time step is undetermined");
                    MPI_Abort(comm,1);
                }
            }
            else if (strcmp(key,"TINTEGRATION")==0){
                if (Expected_arguments(file,"TINTEGRATION",1,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                if(fscanf(file,"%lf",&P->Tintegration)!=1){
                    perror("The time integration is undetermined");
                    MPI_Abort(comm,1);
                }
            }
            else if (strcmp(key,"POTENTIAL")==0){
                if (Expected_arguments(file,"POTENTIAL",1,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                snprintf(format,sizeof(format),"%%%ds",MAX_POTENTIAL_CHAR-1);
                if (fscanf(file,format,P->potential)!=1){
                    perror("The potential name is undetermined");
                    MPI_Abort(comm,1);
                }
            }
            else if (strcmp(key,"INTEGRATOR")==0){
                if (Expected_arguments(file,"INTEGRATOR",1,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                snprintf(format,sizeof(format),"%%%ds",MAX_INTEGRATOR_CHAR-1);
                if (fscanf(file,format,P->integrator)!=1){
                    perror("The integrator name is undetermined");
                    MPI_Abort(comm,1);
                }
            }
            else if (strcmp(key,"OUTPUT_IMAG")==0){
                if (Expected_arguments(file,"OUTPUT_IMAG",1,KnownKeys_imag,comm)!=0){MPI_Abort(comm,1);}
                snprintf(format,sizeof(format),"%%%ds",MAX_OUTPUT_FILES_CHAR-1);
                if (fscanf(file,format,P->output_imag)!=1){
                    perror("The name of OUTPUT_IMAG is undetermined");
                    MPI_Abort(comm,1);
                }
            }
            else{
                printf("\nUnknown parameter: %s\n",key);
                perror("Error with params.txt");
                MPI_Abort(comm,1);
            }
        }
        fclose(file);
        
        /** Final check on the parameters **/
        if (P->dim==0 || P->Nr_global[0]<=0 || P->N_states<=0){
            fprintf(stderr,"ERROR: Could not read or interpret the file %s properly.\n",filename);
            local_error = 1;
        }
    }
    
    /* Error checks on all ranks */
    int global_error_sum=0; // Global Synchronization and Error Check
    MPI_Allreduce(&local_error,&global_error_sum,1,MPI_INT,MPI_SUM,comm);
    if (global_error_sum>0){
        if (rank==0) {fprintf(stderr, "\n--- FATAL ERROR: Parameter file processing failed. Aborting job. ---\n");}
        MPI_Abort(comm,1);
    }
    
    /* Process command-line overrides, starting at index 2 */
    for (j=2;j<argc;j+=2){
        if (j+1<argc){
            char *key=argv[j];
            char *value_str=argv[j+1];
            
            /** Check for known keys and override the structure member **/
            if (strcmp(key,"DT")==0){
                double value=atof(value_str);
                P->dt=value;
            }
            else if (strcmp(key,"TINTEGRATION")==0){
                double value=atof(value_str);
                P->Tintegration=value;
            }
            else if (strcmp(key,"OUTPUT_IMAG")==0){
                strncpy(P->output_imag,value_str,MAX_OUTPUT_FILES_CHAR);
                P->output_imag[MAX_OUTPUT_FILES_CHAR-1]='\0';
            }
            else {
                fprintf(stderr,"Rank %d ERROR: Unrecognized parameter key '%s'. Check spelling.\n",rank,key);
                MPI_Abort(comm,1);
            }
        }
        else{
            fprintf(stderr,"Rank %d ERROR: Parameter key '%s' found without a corresponding value.\n",rank,key);
            MPI_Abort(comm,1);
        }
    }
    
    /* Broadcast all data (All ranks) */
    MPI_Bcast(P->Nr_global,P->dim,MPI_LONG_LONG_INT,0,comm); // Broadcast Nr_global
    MPI_Bcast(P->dr,P->dim,MPI_DOUBLE,0,comm); // Broadcast dr (use MPI_DOUBLE for double)
    MPI_Bcast(&(P->N_states),1,MPI_INT,0,comm);
    MPI_Bcast(&(P->dt),1,MPI_DOUBLE,0,comm);
    MPI_Bcast(&(P->Tintegration),1,MPI_DOUBLE,0,comm);
    MPI_Bcast(P->potential,MAX_POTENTIAL_CHAR,MPI_CHAR,0,comm);
    MPI_Bcast(P->integrator,MAX_INTEGRATOR_CHAR,MPI_CHAR,0,comm);
    MPI_Bcast(P->output_imag,MAX_OUTPUT_FILES_CHAR,MPI_CHAR,0,comm);
    
    if (rank==0){
        printf("  DIMENSION\t\t%d\n",P->dim);
        printf("  NR\t\t");
        for (j=0;j<P->dim;j++){printf("\t%td",P->Nr_global[j]);}
        printf("\n");
        printf("  DR\t\t");
        for (j=0;j<P->dim;j++){printf("\t%16.15f",P->dr[j]);}
        printf("\n");
        printf("  N_STATES\t\t%d\n",P->N_states);
        printf("  POTENTIAL\t\t%s\n",P->potential);
        printf("  INTEGRATOR\t\t%s\n",P->integrator);
        printf("  DT\t\t\t%16.15f\n",P->dt);
        printf("  TINTEGRATION\t\t%16.15f\n",P->Tintegration);
        printf("  OUTPUT_IMAG\t\t%s\n",P->output_imag);
        fflush(stdout); // CRUCIAL: Force the output buffer to be written immediately
    }
}

/* Initialization of the parameters in the real-time propagation */
void Initialize_parameters_real(const char* filename, Parameters* P, Parameters* P_imag, Field* F, int argc, char *argv[], MPI_Comm comm) {
    int j;
    char format[20], key[32];
    int local_error = 0;
    
    /* List of known parameters for the params.real */
    const char *KnownKeys_real[]={
        "OUTPUT_IMAG","NR",
        "N_STATES","J_STATES","COEFFS_STATES",
        "CAP",
        "GAUGE","N_MODES","INTENSITY","LAMBDA","ELLIPTICITY","CEP","ENVELOPE","DURATION","DELAY",
        "INTEGRATOR","DT","TINTEGRATION",
        "DT_SAVE","N_OBSERVABLES","OBSERVABLES","P_MAX","R_MASK","DE","NE",
        NULL};
    
    /* Determine the rank & size */
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    
    /* Set default parameters */
    // dim and output_imag are set by the files, otherwise the program exits
    /** Initial state **/
    P->N_states=1;
    
    /** Integrator **/
    snprintf(P->integrator,MAX_INTEGRATOR_CHAR,"BM4");
    P->dt=.1;
    P->Tintegration=220.6380;
    memset(P->cap,0,2*sizeof(double));
    
    /** Field **/
    F->N_modes=1;
    snprintf(F->gauge,MAX_GAUGE_CHAR,"velocity");
    
    /** Observables **/
    P->dt_save=1.;
    P->N_observables=0;
    P->R_mask[0]=20.;
    P->R_mask[1]=30.;
    P->Ne=100;
    P->de=.01;
    
    /* Read the parameters file */
    FILE* file=NULL;
    
    /** Find the name of the file of the output of the imaginary time propagation **/
    if (rank==0){
        file=fopen(filename,"r");
        if (!file){
            perror("Error opening parameter file");
            local_error=1;
        }
        int output_found=0;
        int nstates_found=0;
        int nobservables_found=0;
        int nmodes_found=0;
        while (fscanf(file,"%31s",key)!=EOF){
            if (strcmp(key,"OUTPUT_IMAG")==0){ // Look for the OUTPUT_IMAG
                if (Expected_arguments(file,"OUTPUT_IMAG",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                snprintf(format,sizeof(format),"%%%ds",MAX_OUTPUT_FILES_CHAR-1);
                if (fscanf(file,format,P->output_imag)!=1){
                    perror("The name of output file is undetermined");
                    fclose(file);
                    MPI_Abort(comm,1);
                }
                output_found=1;
            }
            else if (strcmp(key,"N_STATES")==0){ // Look for N_STATES
                if (Expected_arguments(file,"N_STATES",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                if (fscanf(file,"%d",&P->N_states)!=1){
                    perror("The number of states is undetermined");
                    fclose(file);
                    MPI_Abort(comm,1);
                }
                nstates_found=1;
            }
            else if (strcmp(key,"N_OBSERVABLES")==0){ // Look for N_OBSERVABLES
                if (Expected_arguments(file,"N_OBSERVABLES",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                if (fscanf(file,"%d",&P->N_observables)!=1){
                    perror("The number of observables is undetermined");
                    fclose(file);
                    MPI_Abort(comm,1);
                }
                nobservables_found=1;
            }
            else if (strcmp(key,"N_MODES")==0){ // Look for N_MODES
                if (Expected_arguments(file,"N_MODES",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                if (fscanf(file,"%d",&F->N_modes)!=1){
                    perror("The number of field modes is undetermined");
                    fclose(file);
                    MPI_Abort(comm,1);
                }
                nmodes_found=1;
            }
            if (output_found && nstates_found && nobservables_found && nmodes_found){break;} // Optimization: If all are found, we can stop reading
        }
        fclose(file);
        
        if (!output_found){
            perror("The output file is not specified");
            local_error=1;
        }
        if (!nstates_found){
            printf("WARNING: The number of states is not specified\n");
            printf("Default value: N_STATES=1\n");
        }
        if (!nobservables_found){
            printf("WARNING: The number of observables is not specified\n");
            printf("Default value: N_OBSERVABLES=0\n");
        }
        if (!nobservables_found){
            printf("WARNING: The number of field modes is not specified\n");
            printf("Default value: N_MODES=1\n");
        }
    }
    
    /** Global check and brodcast critical sizes **/
    int global_error_sum=0;
    MPI_Allreduce(&local_error,&global_error_sum,1,MPI_INT,MPI_SUM,comm);
    if (global_error_sum>0) {
        if (rank==0) {fprintf(stderr,"\n FATAL ERROR: Critical parameters missing. Aborting.\n"); }
        MPI_Abort(comm,1);
    }
    
    /** Broadcast the critical sizes and output filename for ALLOCATION **/
    MPI_Bcast(&(P->N_states),1,MPI_INT,0,comm);
    MPI_Bcast(&(F->N_modes),1,MPI_INT,0,comm);
    MPI_Bcast(&(P->N_observables),1,MPI_INT,0,comm);
    MPI_Bcast(P->output_imag,MAX_OUTPUT_FILES_CHAR,MPI_CHAR,0,comm);
    
    /* Dynamic allocation to all ranks */
    P->j_states=(int*) malloc(P->N_states*sizeof(int));
    P->coeffs_states=(double complex*) malloc(P->N_states*sizeof(double complex));
    P->observables=(char**) malloc(P->N_observables*sizeof(char*));

    F->envelope=(char**) malloc(F->N_modes*sizeof(char*));
    F->intensity=(double*) malloc(F->N_modes*sizeof(double));
    F->lambda=(double*) malloc(F->N_modes*sizeof(double));
    F->ellipticity=(double*) malloc(F->N_modes*sizeof(double));
    F->cep=(double*) malloc(F->N_modes*sizeof(double));
    F->delay=(double*) malloc(F->N_modes*sizeof(double));
    F->duration=(double*) malloc(F->N_modes*sizeof(double));
    F->frequency=(double*) malloc(F->N_modes*sizeof(double));
    F->amplitude=(double*) malloc(F->N_modes*sizeof(double));
    
    if (!P->j_states || !P->coeffs_states || !P->observables || !F->envelope || !F->intensity || !F->lambda || !F->ellipticity || !F->cep || !F->delay || !F->duration || !F->frequency || !F->amplitude){
        if (P->j_states!=NULL){free(P->j_states);}
        if (P->coeffs_states!=NULL){free(P->coeffs_states);}
        if (P->observables!=NULL){free(P->observables);}
        if (F->envelope!=NULL){free(F->envelope);}
        if (F->intensity!=NULL){free(F->intensity);}
        if (F->lambda!=NULL){free(F->lambda);}
        if (F->ellipticity!=NULL){free(F->ellipticity);}
        if (F->cep!=NULL){free(F->cep);}
        if (F->delay!=NULL){free(F->delay);}
        if (F->duration!=NULL){free(F->duration);}
        if (F->frequency!=NULL){free(F->frequency);}
        if (F->amplitude!=NULL){free(F->amplitude);}
        fprintf(stderr, "Rank %d ERROR: Memory allocation failed for main arrays.\n",rank);
        MPI_Abort(comm,1);
    }
    
    for (j=0;j<P->N_observables;j++){ // Allocate observable strings
        P->observables[j]=(char*) malloc(MAX_OBSERVABLE_CHAR*sizeof(char));
        if (P->observables[j]==NULL){
            fprintf(stderr,"Rank %d ERROR: Allocation failed for observable string %d.\n",rank,j);
            MPI_Abort(comm,1);
        }
    }
    
    for (j=0;j<F->N_modes;j++){ // Allocate observable strings
        F->envelope[j]=(char*) malloc(MAX_ENVELOPE_CHAR*sizeof(char));
        if (F->envelope[j]==NULL){
            fprintf(stderr,"Rank %d ERROR: Allocation failed for envelope string %d.\n",rank,j);
            MPI_Abort(comm,1);
        }
    }
    
    /** Default values **/
    for (j=0;j<P->N_states;j++){
        P->j_states[j]=1;
        P->coeffs_states[j]=1.;
    }
    for (j=0;j<P->N_observables;j++){snprintf(P->observables[j],MAX_OBSERVABLE_CHAR,"none");}
    for (j=0;j<F->N_modes;j++){
        snprintf(F->envelope[j],MAX_ENVELOPE_CHAR,"cos4");
        F->intensity[j]=1.e14;
        F->lambda[j]=800.;
        F->ellipticity[j]=1.;
        F->cep[j]=0.;
        F->delay[j]=0.;
        F->duration[j]=220.6380;
    }
    
    /** Set the dimension and appropriate memory allocation **/
    int dim=Find_dimension_in_file_MPI(P->output_imag,comm);
    Allocate_dim_arrays(P,dim,comm);
    Allocate_dim_arrays(P_imag,dim,comm);
      
    /* Read all remaining parameters from the output file of the imaginary time propagation (rank 0 only) */
    local_error=0;
    if (rank==0){
        file=fopen(P->output_imag,"r");
        if (file){
            while (fscanf(file,"%31s",key)!=EOF){
                if (strcmp(key,"PSI")==0){break;} // Stop reading parameters when "PSI" is encountered
                else if (strcmp(key,"DIMENSION")==0){fscanf(file,"%31s",key);}
                else if (strcmp(key,"N_STATES")==0){
                    if(fscanf(file,"%d",&P_imag->N_states)!=1){
                        perror("The number of states of OUTPUT_IMAG is undetermined");
                        local_error=1;
                    }
                    if(P->N_states>P_imag->N_states){printf("WARNING: The number of states for the superposition of the initial wave function is larger than the number of states computed in OUTPUT_IMAG");}
                }
                else if (strcmp(key,"E")==0){for (j=0;j<P_imag->N_states;j++){fscanf(file,"%31s",key);}}
                else if (strcmp(key,"DE")==0){for (j=0;j<P_imag->N_states;j++){fscanf(file,"%31s",key);}}
                else if (strcmp(key,"NR")==0){
                    for (j=0;j<P->dim;j++){
                        if(fscanf(file,"%td",&P_imag->Nr_global[j])!=1){
                            printf("The number of grid points of OUTPUT_IMAG is not completely determined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"DR")==0){
                    for (j=0;j<P->dim;j++){
                        if(fscanf(file,"%lf",&P->dr[j])!=1){
                            perror("The grid space of OUPUT_IMAG is not completely determined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"POTENTIAL")==0){
                    snprintf(format,sizeof(format),"%%%ds",MAX_POTENTIAL_CHAR-1);
                    if(fscanf(file,format,P->potential)!=1){
                        perror("The potential name of OUPUT_IMAG is undetermined");
                        local_error=1;
                    }
                }
                else{
                    printf("\nUnknown parameter: %s\n",key);
                    perror("Error with params.real");
                    local_error=1;
                }
            }
            fclose(file);
        }
        else {
            fprintf(stderr, "Rank 0 Error: Failed to re-open main parameter file for full read.\n");
            local_error=1;
        }
        
        /** Read out of the other parameters **/
        file=fopen(filename,"r");
        if (file){
            while (fscanf(file,"%31s",key)!=EOF){
                if (strcmp(key,"OUTPUT_IMAG")==0){
                    if (Expected_arguments(file,"OUTPUT_IMAG",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    snprintf(format,sizeof(format),"%%%ds",MAX_OUTPUT_FILES_CHAR-1);
                    if(fscanf(file,format,key)!=1){
                        perror("The name of the OUPUT_IMAG is undetermined");
                        MPI_Abort(comm,1);
                    }
                }
                else if (strcmp(key,"N_STATES")==0){
                    if (Expected_arguments(file,"N_STATES",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%d",&j)!=1){
                        perror("The number of states is undetermined");
                        MPI_Abort(comm,1);
                    }
                }
                else if (strcmp(key,"NR")==0){
                    if (Expected_arguments(file,"NR",P->dim,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<P->dim;j++){
                        if(fscanf(file,"%td",&P->Nr_global[j])!=1){
                            perror("The number of grid points is not completely determined");
                            MPI_Abort(comm,1);
                        }
                        if ((P->Nr_global[j]!=0 && (P->Nr_global[j] & (P->Nr_global[j]-1))!=0)){
                            printf("\nThe number of points on the grid must be a power of 2\n");
                            perror("Error with grid points number");
                            MPI_Abort(comm,1);
                        }
                        if (P->Nr_global[j]<P_imag->Nr_global[j]){printf("WARNING: The number of points on the grid is smaller than for the imaginary time propagation\n");}
                    }
                }
                else if (strcmp(key,"J_STATES")==0){
                    if (Expected_arguments(file,"J_STATES",P->N_states,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<P->N_states;j++){
                        if(fscanf(file,"%d",&P->j_states[j])!=1){
                            perror("An ith state is undetermined");
                            local_error=1;
                        }
                        if (P->j_states[j]>P_imag->N_states){
                            printf("\nAn initial eigenstate is not computed in the imaginary time propagation\n");
                            perror("Error with initial eigenstate");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"COEFFS_STATES")==0){
                    if (Expected_arguments(file,"COEFFS_STATES",P->N_states,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<P->N_states;j++){
                        if(fscanf(file,"%31s",key)!=1){
                            perror("The coefficient of a, ith state is undetermined");
                            local_error=1;
                        }
                        P->coeffs_states[j]=parse_complex(key);
                    }
                }
                else if (strcmp(key,"CAP")==0){
                    if (Expected_arguments(file,"CAP",2,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%lf%lf",&P->cap[0],&P->cap[1])!=2){
                        perror("The CAP parameters are undetermined (there must be 2 values)");
                        local_error=1;
                    }
                    P->cap[0]=fabs(P->cap[0]);
                    P->cap[1]=fabs(P->cap[1]);
                } // Parameters of the observables
                else if (strcmp(key,"N_OBSERVABLES")==0){
                    if (Expected_arguments(file,"N_OBSERVABLES",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%d",&j)!=1){
                        perror("The number of observables is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"DT_SAVE")==0){
                    if (Expected_arguments(file,"DT_SAVE",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%lf",&P->dt_save)!=1){
                        perror("The time-step for the observables is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"P_MAX")==0){
                    if (Expected_arguments(file,"P_MAX",P->dim,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<P->dim;j++){
                        if(fscanf(file,"%lf",&P->pr_max[j])!=1){
                            perror("The maximum number of momentum coordinate is undetermined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"R_MASK")==0){
                    if (Expected_arguments(file,"R_MASK",2,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%lf%lf",&P->R_mask[0],&P->R_mask[1])!=2){
                        perror("The parameters of the mask are undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"DE")==0){
                    if (Expected_arguments(file,"DE",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%lf",&P->de)!=1){
                        perror("The energy step for the PED is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"NE")==0){
                    if (Expected_arguments(file,"NE",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%d",&P->Ne)!=1){
                        perror("The number of energy step for the PED is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"OBSERVABLES")==0){
                    if (Expected_arguments(file,"OBSERVABLES",P->N_observables,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<P->N_observables;j++){
                        snprintf(format,sizeof(format),"%%%ds",MAX_OBSERVABLE_CHAR-1);
                        if (fscanf(file,format,P->observables[j])!=1){
                            perror("Not enough observable names found.");
                            local_error=1;
                        }
                    }
                } // Parameters of the integration
                else if (strcmp(key,"DT")==0){
                    if (Expected_arguments(file,"DT",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%lf",&P->dt)!=1){
                        perror("The time step is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"TINTEGRATION")==0){
                    if (Expected_arguments(file,"TINTEGRATION",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if (fscanf(file,"%lf",&P->Tintegration)!=1){
                        perror("The integration time is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"INTEGRATOR")==0){
                    if (Expected_arguments(file,"INTEGRATOR",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    snprintf(format,sizeof(format),"%%%ds",MAX_INTEGRATOR_CHAR-1);
                    if(fscanf(file,format,P->integrator)!=1){
                        perror("The integrator is undetermined");
                        local_error=1;
                    }
                } // Parameters of the field
                else if (strcmp(key,"INTENSITY")==0){
                    if (Expected_arguments(file,"INTENSITY",F->N_modes,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<F->N_modes;j++){
                        if (fscanf(file,"%lf",&F->intensity[j])!=1){
                            perror("One field intensity is undetermined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"N_MODES")==0){
                    if (Expected_arguments(file,"N_MODES",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    if(fscanf(file,"%d",&j)!=1){
                        perror("The number of modes is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"LAMBDA")==0){
                    if (Expected_arguments(file,"LAMBDA",F->N_modes,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<F->N_modes;j++){
                        if(fscanf(file,"%lf",&F->lambda[j])!=1){
                            perror("One field wavelength is undetermined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"ELLIPTICITY")==0){
                    if (Expected_arguments(file,"ELLIPTICITY",F->N_modes,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<F->N_modes;j++){
                        if (fscanf(file,"%lf",&F->ellipticity[j])!=1){
                            perror("One field ellipticity is undetermined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"CEP")==0){
                    if (Expected_arguments(file,"CEP",F->N_modes,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<F->N_modes;j++){
                        if(fscanf(file,"%lf",&F->cep[j])!=1){
                            perror("One field CEP is undetermined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"DELAY")==0){
                    if (Expected_arguments(file,"DELAY",F->N_modes,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<F->N_modes;j++){
                        if(fscanf(file,"%lf",&F->delay[j])!=1){
                            perror("One field delay is undetermined");
                            local_error=1;
                        }
                        if(F->delay[j]<0){
                            perror("Delays must be positive (the time integration starts at 0 always)");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"DURATION")==0){
                    if (Expected_arguments(file,"DURATION",F->N_modes,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    for (j=0;j<F->N_modes;j++){
                        if(fscanf(file,"%lf",&F->duration[j])!=1){
                            perror("One field duration is undetermined");
                            local_error=1;
                        }
                    }
                }
                else if (strcmp(key,"GAUGE")==0){
                    if (Expected_arguments(file,"GAUGE",1,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    snprintf(format,sizeof(format),"%%%ds",MAX_GAUGE_CHAR-1);
                    if (fscanf(file,format,F->gauge)!=1){
                        perror("One field gauge is undetermined");
                        local_error=1;
                    }
                }
                else if (strcmp(key,"ENVELOPE")==0){
                    if (Expected_arguments(file,"ENVELOPE",F->N_modes,KnownKeys_real,comm)!=0){MPI_Abort(comm,1);}
                    snprintf(format,sizeof(format),"%%%ds",MAX_ENVELOPE_CHAR-1);
                    for (j=0;j<F->N_modes;j++){
                        if (fscanf(file,format,F->envelope[j])!=1){
                            perror("One field envelope is undetermined");
                            local_error=1;
                        }
                    }
                }
                else{
                    printf("\nUnknown parameter: %s\n",key);
                    perror("Error with params.real");
                    local_error=1;
                }
            }
            fclose(file);
        }
        else{
            fprintf(stderr,"Rank 0 Error: Failed to open imaginary output file for parameter read.\n");
            local_error=1;
        }
    }
    
    /** Global checks **/
    MPI_Allreduce(&local_error,&global_error_sum,1,MPI_INT,MPI_SUM,comm);
    if (global_error_sum>0){
        if (rank==0) {fprintf(stderr, "\n FATAL ERROR: Data arrays or complex fields failed to read. Aborting.\n"); }
        MPI_Abort(comm,1);
    }
    
    /* Process command-line overrides, starting at index 2 */
    for (j=2;j<argc;j+=2){
        if (j+1<argc){
            char *key=argv[j];
            char *value_str=argv[j+1];
            
            /** Check for known keys and override the structure member **/
            if (strcmp(key,"DT")==0){ // Integrator
                double value=atof(value_str);
                P->dt=value;
            }
            else if (strcmp(key,"TINTEGRATION")==0){
                double value=atof(value_str);
                P->Tintegration=value;
            }
            else if (strcmp(key,"NE")==0){ // Observables
                int value=atoi(value_str);
                P->Ne=value;
            }
            else if (strcmp(key,"DE")==0){
                double value=atof(value_str);
                P->de=value;
            }
            else if (strcmp(key,"INTENSITY")==0){ // Laser
                double value=atof(value_str);
                F->intensity[0]=value;
            }
            else if (strcmp(key,"ELLIPTICITY")==0){
                double value=atof(value_str);
                F->ellipticity[0]=value;
            }
            else if (strcmp(key,"LAMBDA")==0){
                double value=atof(value_str);
                F->lambda[0]=value;
            }
            else if (strcmp(key,"CEP")==0){
                double value=atof(value_str);
                F->cep[0]=value;
            }
            else if (strcmp(key,"DELAY")==0){
                double value=atof(value_str);
                F->delay[0]=value;
            }
            else if (strcmp(key,"DURATION")==0){
                double value=atof(value_str);
                F->duration[0]=value;
            }
            else {
                fprintf(stderr,"Rank %d ERROR: Unrecognized parameter key '%s'. Check spelling.\n",rank,key);
                MPI_Abort(comm,1);
            }
        }
        else{
            fprintf(stderr,"Rank %d ERROR: Parameter key '%s' found without a corresponding value.\n",rank,key);
            MPI_Abort(comm,1);
        }
    }
    
    /** Brodcast all parameters **/
    MPI_Bcast(P->Nr_global,P->dim,MPI_LONG_LONG_INT,0,comm); // Grid parameters
    MPI_Bcast(P->dr,P->dim,MPI_DOUBLE,0,comm);
    MPI_Bcast(&(P->potential),MAX_POTENTIAL_CHAR,MPI_CHAR,0,comm); // Potential
    MPI_Bcast(P->cap,2,MPI_DOUBLE,0,comm);
    MPI_Bcast(P->j_states,P->N_states,MPI_INT,0,comm); // States
    MPI_Bcast(P->coeffs_states,2*P->N_states,MPI_DOUBLE,0,comm);
    for (j=0;j<P->N_observables;j++){MPI_Bcast(P->observables[j],MAX_OBSERVABLE_CHAR,MPI_CHAR,0,comm);} // Observables
    MPI_Bcast(&(P->dt_save),1,MPI_DOUBLE,0,comm);
    MPI_Bcast(P->pr_max,P->dim,MPI_DOUBLE,0,comm);
    MPI_Bcast(P->R_mask,2,MPI_DOUBLE,0,comm);
    MPI_Bcast(&(P->Ne),1,MPI_INT,0,comm);                    
    MPI_Bcast(&(P->de),1,MPI_DOUBLE,0,comm);
    MPI_Bcast(&(P->integrator),MAX_INTEGRATOR_CHAR,MPI_CHAR,0,comm); // Integrator
    MPI_Bcast(&(P->dt),1,MPI_DOUBLE,0,comm);
    MPI_Bcast(&(P->Tintegration),1,MPI_DOUBLE,0,comm);
    MPI_Bcast(&(F->gauge),MAX_GAUGE_CHAR,MPI_CHAR,0,comm); // Field
    for (j=0;j<F->N_modes;j++){MPI_Bcast(F->envelope[j],MAX_ENVELOPE_CHAR,MPI_CHAR,0,comm);}
    MPI_Bcast(F->intensity,F->N_modes,MPI_DOUBLE,0,comm);
    MPI_Bcast(F->ellipticity,F->N_modes,MPI_DOUBLE,0,comm);
    MPI_Bcast(F->lambda,F->N_modes,MPI_DOUBLE,0,comm);
    MPI_Bcast(F->cep,F->N_modes,MPI_DOUBLE,0,comm);
    MPI_Bcast(F->delay,F->N_modes,MPI_DOUBLE,0,comm);
    MPI_Bcast(F->duration,F->N_modes,MPI_DOUBLE,0,comm);
    MPI_Bcast(P_imag->Nr_global,P_imag->dim,MPI_LONG_LONG_INT,0,comm); // Imaginary data
    MPI_Bcast(&(P_imag->N_states),1,MPI_INT,0,comm);
    
    /* Initialize the parameters accordingly and display them */
    /** Normalize the coefficients of the initial state (assuming the eigenstates are orthonormal) **/
    double norm2=0.;
    for (j=0;j<P->N_states;j++){norm2+=cabs(P->coeffs_states[j])*cabs(P->coeffs_states[j]);}
    for (j=0;j<P->N_states;j++){P->coeffs_states[j]/=sqrt(norm2);}
    
    /** Laser field parameters **/
    for (j=0;j<F->N_modes;j++){
        F->amplitude[j]=5.33803e-9*sqrt(F->intensity[j]);
        F->frequency[j]=45.5637639947958/(double)F->lambda[j];
    }
    
    /** Display the parameters that have been initiated **/
    if (rank==0){
        printf("> For real-time propagation:\n");
        printf("  DIMENSION\t\t%d\n",P->dim);
        printf("  NR\t\t");
        for (j=0;j<P->dim;j++){printf("\t%td",P->Nr_global[j]);}
        printf("\n");
        printf("  DR\t\t");
        for (j=0;j<P->dim;j++){printf("\t%16.15f",P->dr[j]);}
        printf("\n");
        printf("  N_STATES\t\t%d\n",P->N_states);
        printf("  J_STATES\t");
        for (j=0;j<P->N_states;j++){printf("\t%d",P->j_states[j]);}
        printf("\n");
        printf("  COEFFS_STATES\t");
        for (j=0;j<P->N_states;j++){printf("\t%16.15f+I*%16.15f",creal(P->coeffs_states[j]),cimag(P->coeffs_states[j]));}
        printf("\n");
        printf("  POTENTIAL\t\t%s\n",P->potential); // Read from output_imag
        printf("  CAP\t\t\t%16.15f\t%16.15f\n",P->cap[0],P->cap[1]);
        printf("  INTEGRATOR\t\t%s\n",P->integrator);
        printf("  DT\t\t\t%16.15f\n",P->dt);
        printf("  TINTEGRATION\t\t%16.15f\n",P->Tintegration);
        printf("\n");
        
        printf("> Field parameters:\n");
        printf("  GAUGE\t\t\t%s\n",F->gauge);
        printf("  N_MODES\t\t%d\n",F->N_modes);
        printf("  ENVELOPE\t");
        for (j=0;j<F->N_modes;j++){printf("\t%s",F->envelope[j]);}
        printf("\n");
        printf("  DURATION\t");
        for (j=0;j<F->N_modes;j++){printf("\t%16.15f",F->duration[j]);}
        printf("\n");
        printf("  INTENSITY\t");
        for (j=0;j<F->N_modes;j++){printf("\t%6.5e",F->intensity[j]);}
        printf("\n");
        printf("  LAMBDA\t");
        for (j=0;j<F->N_modes;j++){printf("\t%16.15f",F->lambda[j]);}
        printf("\n");
        printf("  ELLIPTICITY\t");
        for (j=0;j<F->N_modes;j++){printf("\t%16.15f",F->ellipticity[j]);}
        printf("\n");
        printf("  CEP\t\t");
        for (j=0;j<F->N_modes;j++){printf("\t%16.15f",F->cep[j]);}
        printf("\n");
        printf("  DELAY\t\t");
        for (j=0;j<F->N_modes;j++){printf("\t%16.15f",F->delay[j]);}
        printf("\n\n");
        
        printf("> Observables:\n");
        printf("  N_OBSERVABLES\t\t%d\n",P->N_observables);
        printf("  OBSERVABLES\t");
        for (j=0;j<P->N_observables;j++){printf("\t%s",P->observables[j]);}
        printf("\n");
        printf("  DT_SAVE\t\t%16.15f\n",P->dt_save);
        printf("  P_MAX\t\t");
        for (j=0;j<P->dim;j++){printf("\t%16.15f",P->pr_max[j]);}
        printf("\n");
        printf("  R_MASK\t\t%16.15f\t%16.15f\n",P->R_mask[0],P->R_mask[1]);
        printf("  DE\t\t\t%16.15f\n",P->de);
        printf("  NE\t\t\t%d\n",P->Ne);
        printf("\n");
        
        printf("> For the imaginary-time propagation's data taken from '%s':\n",P->output_imag);
        printf("  OUTPUT_IMAG\t\t%s\n",P->output_imag);
        printf("  DIMENSION\t\t%d\n",P_imag->dim);
        printf("  NR\t\t");
        for (j=0;j<P->dim;j++){printf("\t%td",P_imag->Nr_global[j]);}
        printf("\n");
        printf("  N_STATES\t\t%d\n",P_imag->N_states);
        
        fflush(stdout); // CRUCIAL: Force the output buffer to be written immediately
    }
    
    /** Time vector & number of integration steps **/
    ptrdiff_t i;
    P->N_time=(ptrdiff_t) (P->Tintegration/(double)P->dt);
    P->time=(double*) malloc(P->N_time*sizeof(double));
    if (P->time==NULL){
        fprintf(stderr, "ERROR: memory allocation failed for P->time in Initialize_parameters.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (i=0;i<P->N_time;i++){P->time[i]=i*P->dt;}
}

/* Definition of the field vectors */
void Field_vectors(const Parameters* P, const Field* F, const double t, double* A, double *E, MPI_Comm comm){ // Field vectors functions in dim: Electric field E(t)=(E0[t],E1[t]...), Potential vector A(t)=(A0[t],A1[t]...), time t
    double f=0.,df=0.;
    
    /* Initialization */
    for (int j=0;j<P->dim;j++){
        E[j]=0.;
        A[j]=0.;
    }
    
    /* Summing all the modes */
    for (int j=0;j<F->N_modes;j++){
        double w=F->frequency[j];
        double F0=F->amplitude[j];
        double xi=F->ellipticity[j];
        double tau=F->duration[j];
        double phase=F->cep[j];
        double t_shift=t-F->delay[j];
        
        /** Laser envelope **/
        if (strcmp(F->envelope[j],"constant")==0){
            f=1.;
            df=0.;
        }
        else if (strcmp(F->envelope[j],"trapz")==0){
            double trud=2.*2.*M_PI/w; // Ramp-up & ramp-down durations
            f=pow(sin(.5*M_PI*t_shift/(double)trud),4) * (t_shift>=0. && t_shift<=trud) + (t_shift>trud && t_shift<(tau-trud)) + pow(sin(.5*M_PI*(t_shift-tau)/(double)trud),4) * (t_shift>=tau-trud && t_shift<=tau);
            df=M_PI/(double)trud*(pow(sin(.5*M_PI*t_shift/(double)trud),2)*sin(M_PI*t_shift/(double)trud) * (t_shift>=0. && t_shift<=trud) + pow(sin(.5*M_PI*(t_shift-tau)/(double)trud),2)*sin(M_PI*(t_shift-tau)/(double)trud) * (t_shift>=tau-trud && t_shift<=tau));
        }
        else if (strcmp(F->envelope[j],"cos2")==0){
            f=pow(sin(M_PI*t_shift/(double)tau),2) * (t_shift>=0. && t_shift<=tau);
            df=M_PI*sin(2.*M_PI*t_shift/(double)tau)/(double)tau  * (t_shift>=0. && t_shift<=tau);
        }
        else if (strcmp(F->envelope[j],"cos4")==0){
            f=pow(sin(M_PI*t_shift/(double)tau),4) * (t_shift>=0. && t_shift<=tau);
            df=4.*M_PI*cos(M_PI*t_shift/(double)tau)*pow(sin(M_PI*t_shift/(double)tau),3)/(double)tau * (t_shift>=0. && t_shift<=tau);
        }
        else{
            printf("\nUnknown envelope: %s\n",F->envelope[j]);
            perror("Error with the envelope name");
            MPI_Abort(comm,1);
        }
        
        /** Laser electric field E=(E1[t0+as[0]*h],...,EN[t0+as[0]*h],E1[t0+(as[0]+as[1]+as[2])*h],...,EN[t0+(as[0]+as[1]+as[2])*h],...) **/
        E[0]+=-F0*(df*cos(w*t_shift+phase)-f*w*sin(w*t_shift+phase))/(double)w; // Fx
        if (P->dim>=2){E[1]+=-F0*xi*(df*sin(w*t_shift+phase)+f*w*cos(w*t_shift+phase))/(double)w;} // Fy
        
        /** Laser vector potential A=(A1[t0+as[0]*h],...,AN[t0+as[0]*h],A1[t0+(as[0]+as[1]+as[2])*h],...,AN[t0+(as[0]+as[1]+as[2])*h],...) **/
        A[0]+=F0*f*cos(w*t_shift+phase)/(double)w; // Ax
        if (P->dim>=2){A[1]+=F0*xi*f*sin(w*t_shift+phase)/(double)w;} // Ay
    }
}

/*######################################*/
// FINALIZATION
/*######################################*/
/* Finalization of the parameters */
void Finalize_parameters(Parameters* P){
    if (P==NULL) {return;} // Check if P is allocated
    if (P->Nr_global!=NULL) {free(P->Nr_global); P->Nr_global=NULL;}
    if (P->dr!=NULL) {free(P->dr); P->dr=NULL;}
    if (P->pr_max!=NULL) {free(P->pr_max); P->pr_max=NULL;}
    if (P->time!=NULL) {free(P->time); P->time=NULL;}
    if (P->j_states!=NULL) {free(P->j_states); P->j_states=NULL;}
    if (P->coeffs_states!=NULL) {free(P->coeffs_states); P->coeffs_states=NULL;}
    if (P->observables!=NULL) {
        for (int j=0;j<P->N_observables;j++) {
            if (P->observables[j]!=NULL) {free(P->observables[j]); P->observables[j]=NULL;}
        }
    }
    free(P->observables);
    P->observables=NULL;
}

/* Finalization of the field vectors */
void Finalize_field_vectors(Field* F){
    if (F==NULL) {return;} // Check if F is allocated
    if (F->intensity!=NULL) {free(F->intensity); F->intensity=NULL;}
    if (F->lambda!=NULL) {free(F->lambda); F->lambda=NULL;}
    if (F->cep!=NULL) {free(F->cep); F->cep=NULL;}
    if (F->ellipticity!=NULL) {free(F->ellipticity); F->ellipticity=NULL;}
    if (F->duration!=NULL) {free(F->duration); F->duration=NULL;}
    if (F->delay!=NULL) {free(F->delay); F->delay=NULL;}
    if (F->frequency!=NULL) {free(F->frequency); F->frequency=NULL;}
    if (F->amplitude!=NULL) {free(F->amplitude); F->amplitude=NULL;}
    if (F->envelope!=NULL) {
        for (int j=0;j<F->N_modes;j++) {
            if (F->envelope[j]!=NULL) {free(F->envelope[j]); F->envelope[j]=NULL;}
        }
    }
    free(F->envelope);
    F->envelope=NULL;
    if (F->A!=NULL) {free(F->A); F->A=NULL;}
    if (F->E!=NULL) {free(F->E); F->E=NULL;}
}

/*######################################*/
// UTILITY FUNCTIONS
/*######################################*/
/* Find the dimension in a file */
int Find_dimension_in_file_MPI(const char* filename, MPI_Comm comm){
    int rank;
    MPI_Comm_rank(comm,&rank);
    int dim=0;
    if (rank==0){
        FILE* file=NULL;
        file=fopen(filename,"r");
        if (!file){
            perror("Error opening parameter file");
            MPI_Abort(comm,1);
        }
        
        char key[32];
        while (fscanf(file,"%31s",key)!=EOF){
            if (strcmp(key,"DIMENSION")==0){
                if (fscanf(file,"%d",&dim)!=1) {
                    perror("The dimension is undetermined");
                    MPI_Abort(comm,1);
                }
                break;
            }
        }
        fclose(file);
        
        if (dim==0){
            printf("\nThe dimension is not specified\n");
            MPI_Abort(comm,1);
        }
    }
    MPI_Bcast(&dim,1,MPI_INT,0,comm); // Broadcast DIMENSION to all ranks so they know how to allocate memory
    return dim;
}

/* Allocate the parameters that depend on the dimension */
void Allocate_dim_arrays(Parameters* P, int dim, MPI_Comm comm){
    P->dim=dim;
    
    /* Memory allocation */
    P->Nr_global=(ptrdiff_t*)malloc(P->dim*sizeof(ptrdiff_t));
    P->dr=(double*)malloc(P->dim*sizeof(double));
    P->pr_max=(double*)malloc(P->dim*sizeof(double));
    
    if (!P->Nr_global || !P->dr || !P->pr_max){
        if (P->Nr_global!=NULL){free(P->Nr_global);}
        if (P->dr!=NULL){free(P->dr);}
        if (P->pr_max!=NULL){free(P->pr_max);}
        fprintf(stderr, "ERROR: memory allocation failed for Nr_global or dr.\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    
    /* Providing *default* parameters */
    for (int j=0;j<P->dim;j++){
        P->Nr_global[j]=256;
        P->dr[j]=.1;
        P->pr_max[j]=1.;
    }
}

/* Read complex numbers from a string */
double complex parse_complex(const char* str){ // Parses a string like "0.5+0.5j", "1.0", "-0.3j", "2-3j", etc.
    double re=0.,im=0.;
    char* plus_ptr=strchr(str,'+');
    char* minus_ptr=strrchr(str,'-'); // last minus (might be inside exponent)

    if (plus_ptr && strchr(str,'j')){sscanf(str,"%lf+%lfj",&re,&im);} // "a+bj"
    else if (minus_ptr && minus_ptr!=str && strchr(str,'j')){ // "a-bj"
        sscanf(str,"%lf-%lfj",&re,&im);
        im=-im;
    }
    else if (strchr(str,'j')){ // "bj"
        sscanf(str,"%lfj",&im);
        re=0.;
    }
    else { // "a"
        sscanf(str,"%lf",&re);
        im=0.;
    }
    return re+im*I;
}

/* Check if there is the right amount of parameters in the raws of params.imag or params.real */
int Expected_arguments(FILE *file_ptr, const char *parameter_name, int expected_number, const char *KnownKeys[], MPI_Comm comm) {
    long initial_position;
    int count=0;
    char buffer[MAX_PARAM_NAME_LEN];
    
    /* Initialization */
    int rank;
    MPI_Comm_rank(comm,&rank); // Safety: Get rank for reporting
    initial_position=ftell(file_ptr); // Store the starting position of the file pointer
    
    /* Counting loop */
    while (count<expected_number){ // Start Counting Loop
        if (fscanf(file_ptr,"%s",buffer)!=1){ // Use fscanf to read the next token (string or number), "%s" reads a token delimited by whitespace
            // Reached the end of the file (EOF) before reading expected arguments
            fprintf(stderr,"Rank %d ERROR reading file: Parameter '%s' expected %d argument(s), but found only %d (reached EOF).\n",rank,parameter_name,expected_number,count);
            MPI_Abort(comm,1); // Critical: Abort before attempting to rewind, as EOF handling can be tricky
            return 1; // Should not be reached
        }
        if (Is_known_key(buffer,KnownKeys)){ // Found a new key/parameter before reaching the expected count
            fprintf(stderr,"Rank %d ERROR reading file: Parameter '%s' expected %d argument(s), but found only %d (hit new parameter '%s').\n",rank,parameter_name,expected_number,count,buffer);
            MPI_Abort(comm,1);
            return 1;
        }
        count++; // Increment the argument count
    }
    long position_after_expected=ftell(file_ptr);
    
    /* Check for Extra Arguments */
    if (fscanf(file_ptr,"%s",buffer)==1){
        if (Is_known_key(buffer,KnownKeys)){ // The argument count was EXACTLY right (N arguments), and we hit the next key
            if (fseek(file_ptr,position_after_expected,SEEK_SET)!=0){ // Rewind the file pointer to the position *before* the new key was read (position_after_expected)
                perror("Error rewinding file pointer to new key");
                MPI_Abort(comm,1);
                return 1;
            } // The count was correct. Proceed to final rewind (Step 4).
        }
        else { // If the token is NOT a known key, it's an extraneous, unexpected argument.
            fprintf(stderr,"Rank %d ERROR reading file: Parameter '%s' expected exactly %d argument(s), but found extra, unrecognized token '%s'.\n",rank,parameter_name,expected_number,buffer);
            MPI_Abort(comm,1);
            return 1; // Should not be reached
        }
    }
    
    /* Rewind the file pointer */
    if (fseek(file_ptr,initial_position,SEEK_SET)!=0){ // Return the file pointer to the position recorded in step 1
         perror("Error rewinding file pointer in Expected_arguments");
         MPI_Abort(comm,1);
         return 1;
    }
    return 0; // Success
}

bool Is_known_key(const char *token, const char **key_list) {
    for (int k=0;key_list[k]!=NULL;k++){if (strcmp(token,key_list[k])==0){return true;}}
    return false;
}
