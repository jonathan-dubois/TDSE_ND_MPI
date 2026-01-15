
#include "io_functions.h"
#include <stdio.h> // for fprintf, fopen, fclose
#include <stdlib.h> // for malloc, EXIT_FAILURE
#include <string.h> // for strcmp
#include <math.h> // for sqrt
#include <limits.h> // Provides INT_MAX

/*######################################*/
// WRITING
/*######################################*/
/* Writing the full wave function into a file */
void Write_wavefunction_to_file(const char* filename, double complex* Psi, const Parameters* P, const MeshGrid* grid, MPI_Comm comm){
    int j;
    FILE *file=NULL;
    
    /* Determine the rank & size */
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    
    /* Creation of the file */
    if (rank==0) { // Node 0 creates/truncates the file
        file=fopen(filename,"w");
        if (!file){perror("Error opening the file in write_wavefunction_imag_to_file"); MPI_Abort(comm,1);}
        fprintf(file,"DIMENSION\t%d\n",P->dim);
        fprintf(file,"NR");
        for (j=0;j<P->dim;j++){fprintf(file,"\t%td",P->Nr_global[j]);}
        fprintf(file,"\n");
        fprintf(file,"DR");
        for (j=0;j<P->dim;j++){fprintf(file,"\t%16.15f",P->dr[j]);}
        fprintf(file,"\n");
        fprintf(file,"PSI_BINARY_START\n");
        fclose(file);
    }
    MPI_Barrier(comm); // Wait until file is created
       
    /* All the ranks write down the wave functions */
    size_t local_count=grid->dim_HS;
    for (int r=0;r<size;r++){
        if (rank==r){
            FILE *file=fopen(filename,"ab"); // append mode
            if (!file) {perror("Error opening the appended binary file in Write_wavefunction_to_file"); MPI_Abort(comm,1);}
            size_t written=fwrite(Psi,sizeof(complex double),local_count,file);
            if (written!=(size_t)local_count){
                fprintf(stderr,"Rank %d failed to write all elements. Expected %zu, wrote %zu.\n",rank,(size_t)local_count,written);
                perror("Error writing Psi array to binary file");
                MPI_Abort(comm,1);
            }
            fclose(file);
        }
        MPI_Barrier(comm); // Wait until this rank finishes writing
    }
}

/* Writing the OUTPUT_IMAG file */
void Write_output_imag(const char* filename, double complex** Psi, double** Ej_global, const Parameters* P, const MeshGrid* grid, MPI_Comm comm){
    int j;
    FILE *file=NULL;
    
    /* Determine the rank & size */
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    
    /* Write the header in text mode (e.g., "w") so it remains human-readable */
    if (rank==0){
        file=fopen(filename,"w");
        if (!file){perror("Error opening the file in write_output_imag"); MPI_Abort(comm,1);}
        fprintf(file,"DIMENSION\t%d\n",P->dim);
        fprintf(file,"N_STATES\t%d\n",P->N_states);
        fprintf(file,"NR");
        for (j=0;j<P->dim;j++){fprintf(file,"\t%td",P->Nr_global[j]);}
        fprintf(file,"\n");
        fprintf(file,"DR");
        for (j=0;j<P->dim;j++){fprintf(file,"\t%16.15f",P->dr[j]);}
        fprintf(file,"\n");
        fprintf(file,"POTENTIAL\t%s\n",P->potential);
        fprintf(file,"E\t");
        for (j=0;j<P->N_states;j++){fprintf(file,"\t%16.15f",Ej_global[j][0]);}
        fprintf(file,"\n");
        fprintf(file,"DE");
        for (j=0;j<P->N_states;j++){fprintf(file,"\t%16.15f",Ej_global[j][1]-Ej_global[j][0]*Ej_global[j][0]);}
        fprintf(file,"\n");
        fprintf(file,"PSI\n");
        fclose(file);
    }
    MPI_Barrier(comm); // Wait until rank 0 finishes writing header
    
    /* All the ranks write down the wave functions (BINARY I/O) */
    for (j=0;j<P->N_states;j++){ // For each state
        for (int r=0;r<size;r++){ // Each rank writes its wavefunction part sequentially
            if (rank==r){
                FILE *file=fopen(filename,"ab"); // Open the file in BINARY APPEND mode ("ab")
                if (!file){perror("Error opening the appended file in write_output_imag"); MPI_Abort(comm,1);}
                size_t elements_written=fwrite(Psi[j],sizeof(double complex),grid->dim_HS,file);
                if (elements_written!=grid->dim_HS){
                    fprintf(stderr,"Rank %d ERROR: Failed to write all %td elements to binary file.\n",rank,grid->dim_HS);
                    MPI_Abort(comm,1);
                }
                fclose(file);
            }
            MPI_Barrier(comm); // Ensure sequential writing
        }
    }
}

/* Write observables on the fly (during the simulation) */
void Write_observables(FILE *file_ptr, double time, const Observables* Obs, MPI_Comm comm){
    /* Determine the rank */
    int rank;
    MPI_Comm_rank(comm,&rank);
    
    /* Write the observables */
    if (Obs==NULL || Obs->total_count==0){return;} // If no observables are required
    int j,k;
    double value;
    if (rank==0) { // Only rank 0 handles the file pointer and writing logic
        if (file_ptr==NULL){ // Safe guard
            fprintf(stderr, "Rank 0: ERROR: Observables file pointer is NULL.\n");
            return;
        }
        fprintf(file_ptr,"%16.15f",time); // Write the time advancement
        for (j=0;j<Obs->total_count;j++){
            for (k=0;k<Obs->dimensions[j];k++) { // Read results from the global buffer using the calculated offset
                value=Obs->values[Obs->offsets[j]+k];
                fprintf(file_ptr,"\t%16.15f",value);
            }
        }
        fprintf(file_ptr,"\n");
        fflush(file_ptr); // Ensure data is written immediately (important for long runs)
    }
}

/* Global observables calculated at the end of the simulation */
void Write_final_observables(double complex *Psi, const Parameters *P, MeshGrid *grid, const Observables *Obs, ptrdiff_t *local_start, MPI_Comm comm){
    ptrdiff_t i;
    int j;
    double sigma=P->de;
    
    /* Determine the rank & size */
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    
    /* Check whether the PED or PMD must be calculated, if yes go to p-representation */
    /** Check if the PED or the PMD must be computed **/
    int is_PMD=0;
    int is_PED=0;
    int is_found_any=0;
    for (j=0;j<Obs->total_count;j++){
        if (strcmp(Obs->names[j],"PED")==0){is_PED=1; is_found_any=1;}
        else if (strcmp(Obs->names[j],"PMD")==0){is_PMD=1; is_found_any=1;}
        else if (strcmp(Obs->names[j],"wavefunction")==0){Write_wavefunction_to_file("wavefunction.real",Psi,P,grid,comm);} // Save the wavefunction directly
    }
    if (is_found_any==0){return;} // Continue only if the PED or the PMD has to be computed
    
    /** Apply the mask & go to p-representation for the calculation of these observables **/
    for(i=0;i<grid->dim_HS;i++){
        double radius=0.;
        for (j=0;j<P->dim;j++){radius+=grid->r[grid->index[j*grid->dim_HS+i]]*grid->r[grid->index[j*grid->dim_HS+i]];}
        radius=sqrt(radius);
        double fmask=0.;
        if (radius>=P->R_mask[0] && radius<=P->R_mask[1]){fmask=pow(sin(.5*M_PI*(radius-P->R_mask[0])/(double)(P->R_mask[1]-P->R_mask[0])),2.);}
        else if (radius>P->R_mask[1]){fmask=1.;}
        grid->in[i]=fmask*Psi[i];
    }
    fftw_execute(grid->plan_forward);
    for(i=0;i<grid->dim_HS;i++){Psi[i]=grid->out[i]*sqrt(grid->fft_scaling);} // Psi in p-representation & rescaled
    
    /* Calculate the photoelectron energy distribution */
    if (is_PED){
        if (rank==0){printf("> Calculation of the photoelectron energy distribution\n");}
        
        /** Memory allocation **/
        double *S_local=NULL;
        double *S_global=NULL;
        S_local=(double*) malloc(P->Ne*sizeof(double));
        S_global=(double*) malloc(P->Ne*sizeof(double));
        if (S_local==NULL || S_global==NULL){
            if (S_local!=NULL) free(S_local);
            if (S_global!=NULL) free(S_global);
            perror("Error: malloc for S in Calculation_final_global_observables\n");
            MPI_Abort(comm,1);
        }
        for (j=0;j<P->Ne;j++){
            S_local[j]=0.;
            S_global[j]=0.;
        }
        
        /** PED calculation **/
        double normalization_factor=1./(double)pow(2.*M_PI*sigma*sigma,.5*P->dim);
        for (i=0;i<grid->dim_HS;i++){
            for (j=0;j<P->dim;j++){if (fabs(grid->pr[grid->index[j*grid->dim_HS+i]])>P->pr_max[j]){break;}} // Check if the momenta are within the range
            if (j==P->dim){ // The momenta are within the range, the PMD can be saved, and the spectrum computed
                for (j=0;j<P->Ne;j++){S_local[j]+=cabs(Psi[i])*cabs(Psi[i])*exp(-.5*(grid->Kin[i]-j*P->de)*(grid->Kin[i]-j*P->de)/(double)(sigma*sigma))*normalization_factor;} // Update the spectrum
            }
        }
        MPI_Allreduce(S_local,S_global,P->Ne,MPI_DOUBLE,MPI_SUM,comm); // Global summation of S_local all at once
        
        /** Save the spectrum to a file **/
        FILE *file_PED=NULL;
        if (rank==0){
            file_PED=fopen("PED.real","w");
            if (!file_PED){perror("Error opening the file in write_output_imag"); MPI_Abort(comm,1);}
            for (j=0;j<P->Ne;j++){fprintf(file_PED,"%16.15f\t%16.15f\n",j*P->de,S_global[j]);}
            fflush(file_PED); // Ensure data is written immediately (important for long runs)
            fclose(file_PED);
        }
        
        /** Cleanup **/
        free(S_local);
        free(S_global);
    }
    
    /* Calculate the photoelectron momentum distribution */
    if (is_PMD){
        if (rank==0){printf("> Calculation of the photoelectron momentum distribution\n");}
        
        /** Counting the number of elements falling into the range **/
        ptrdiff_t count_local[P->dim];
        ptrdiff_t count_global[P->dim];
        for (j=0;j<P->dim;j++){
            count_local[j]=0; // Initialize local counters to zero
            count_global[j]=0; // Initialize global counters to zero
            for (i=0;i<grid->Nr_local[j];i++){
                if (fabs(grid->pr[grid->Nc[j]+i])<=P->pr_max[j]){
                    count_local[j]++;
                }
            }
        }
        MPI_Allreduce(&count_local[0],&count_global[0],1,MPI_LONG_LONG_INT,MPI_SUM,comm); // Only dim 0 is summed in the global summation
        for (j=1;j<P->dim;j++){count_global[j]=count_local[j];} // In the other dimensions they are all the same
        
        /** Save the data in a file **/
        if (rank==0){ // Rank 0 creates the file in binary mode
            FILE *file_PMD=fopen("PMD.real","w");
            if (!file_PMD){perror("Error opening the PMD.real file"); MPI_Abort(comm,1);}
            fprintf(file_PMD,"DIMENSION\t%d\n",P->dim); // Write down the dimension
            fprintf(file_PMD,"NPR"); // Write down the number of momenta in each dimension
            for (j=0;j<P->dim;j++){fprintf(file_PMD,"\t%td",count_global[j]);}
            fprintf(file_PMD,"\n");
            fprintf(file_PMD,"DP");
            for (j=0;j<P->dim;j++){
                double dp=2.*M_PI/(double)(P->dr[j]*P->Nr_global[j]);
                fprintf(file_PMD,"\t%16.15f",dp);
            }
            fprintf(file_PMD,"\n");
            fprintf(file_PMD,"DATA_START\n");
            fclose(file_PMD);
        }
        MPI_Barrier(comm); // Wait until file is created

        size_t max_buffer_size=2*grid->dim_HS*sizeof(double); // Max possible size
        double *buffer=malloc(max_buffer_size);
        if (buffer==NULL){
            fprintf(stderr,"Rank %d ERROR: Memory allocation failed for PMD buffer.\n",rank);
            perror("malloc"); // Prints the specific system error message (e.g., "Cannot allocate memory")
            MPI_Abort(comm,1); // Abort the entire MPI job cleanly
        }
        size_t buffer_index=0; // Tracks position in buffer
        for (int r=0;r<size;r++){ // Sequential binary append loop
            if (rank==r){
                for (i=0;i<grid->dim_HS;i++){
                    for (j=0;j<P->dim;j++){ // Check if the momenta are within the range
                        if (fabs(grid->pr[grid->index[j*grid->dim_HS+i]])>P->pr_max[j]) {break;}
                    }
                    if (j==P->dim){ // Write to buffer
                        buffer[buffer_index++]=cabs(Psi[i])*cabs(Psi[i]); // Density
                        buffer[buffer_index++]=carg(Psi[i]); // Phase
                    }
                }
                FILE *file_PMD=fopen("PMD.real","ab");
                if (!file_PMD) {perror("Error opening appended PMD.bin file"); MPI_Abort(comm,1);}
                size_t items_to_write=buffer_index; // Total doubles to write
                size_t items_written=fwrite(buffer,sizeof(double),items_to_write,file_PMD);
                if (items_written!=items_to_write){
                    fprintf(stderr,"Rank %d ERROR: Failed to write complete PMD record.\n",rank);
                    // Optional: MPI_Abort(comm, 1);
                }
                fclose(file_PMD);
            }
            MPI_Barrier(comm); // Wait until this rank finishes writing before the next rank starts
        }
        free(buffer);
    }
}

/*######################################*/
// READING
/*######################################*/
void Read_wavefunction_MPI(const Parameters *P, const Parameters *P_imag, const MeshGrid *grid, double complex *Psi, ptrdiff_t *local_start_real, MPI_Comm comm){
    int j,k;
    char key[32];
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    
    /* Initialization */
    /** Ensure all non-decomposed dimensions start at 0 **/
    for (j=1;j<P->dim;j++){local_start_real[j]=0;}
    
    /** Memory allocation and Setup **/
    ptrdiff_t *index=NULL;
    ptrdiff_t *centering_offset=NULL;
    index=(ptrdiff_t*) malloc(P_imag->dim*sizeof(ptrdiff_t));
    centering_offset=(ptrdiff_t*) malloc(P_imag->dim*sizeof(ptrdiff_t));
    if (index==NULL || centering_offset==NULL){
        if (index!=NULL) {free(index);}
        if (centering_offset!=NULL) {free(centering_offset);}
        fprintf(stderr, "Rank %d: ERROR: memory allocation failed in Reand_wavefunction_MPI.\n",rank);
        MPI_Abort(comm,1);
    }
    
    /** Initialization Constants **/
    const int element_size_doubles=2; // Re, Im
    ptrdiff_t dim_HS_imag=1; // Calculate global dimension of the imaginary grid (for offset calculation)
    for (j=0;j<P_imag->dim;j++){dim_HS_imag*=P_imag->Nr_global[j];}
    const MPI_Offset full_state_size_elements=(MPI_Offset)dim_HS_imag;
    
    /** Grid Alignment: Calculate Centering Offset **/
    for (j=0;j<P->dim;j++){centering_offset[j]=(P->Nr_global[j]-P_imag->Nr_global[j])/2;}

    /** Calculate Contiguous Block Size (dim_x)  **/
    ptrdiff_t dim_x=1; // dim_x is the number of elements in all other dimensions (j=1 to P->dim-1)
    for (j=1;j<P_imag->dim;j++){dim_x*=P_imag->Nr_global[j];} // If P_imag->dim is 1, dim_x remains 1

    /** Calculate Overlap in the Slow Dimension (Dimension 0) **/
    const ptrdiff_t dim_slow=0; // We assume the slow variable is P.dim=0 index
    ptrdiff_t G_real_start=local_start_real[dim_slow]; // Real-time grid boundaries (in global index space)
    ptrdiff_t G_real_end=local_start_real[dim_slow]+grid->Nr_local[dim_slow];
    ptrdiff_t G_imag_start=centering_offset[dim_slow]; // Imaginary-time grid boundaries (in global index space)
    ptrdiff_t G_imag_end=G_imag_start+P_imag->Nr_global[dim_slow];
    ptrdiff_t overlap_start=G_real_start > G_imag_start ? G_real_start : G_imag_start; // Calculate the start and end of the intersection (Overlap) in the G_real space
    ptrdiff_t overlap_end=G_real_end < G_imag_end ? G_real_end : G_imag_end;
    ptrdiff_t N_blocks_to_read=overlap_end-overlap_start; // Calculate the number of blocks to READ (N_blocks) and the start position in the FILE (I_start)
    ptrdiff_t I_start_in_file=overlap_start-G_imag_start; // The index in the FILE where reading begins (relative to the start of the imaginary grid data)
    
    if (N_blocks_to_read<=0){ // Safety checks for rank not overlapping the imaginary grid (This rank's chunk is entirely outside the imaginary data)
        N_blocks_to_read=0;
        I_start_in_file=0;
    }
    
    ptrdiff_t total_local_read_elements=N_blocks_to_read*dim_x; // Total number of elements (complex doubles) this rank has to treat
    
    double complex *temp_psi_local=fftw_alloc_complex(total_local_read_elements);
    if (temp_psi_local==NULL){
        fprintf(stderr,"Rank %d: Memory allocation error for temp_psi_local\n",rank);
        MPI_Abort(comm,1);
    }
    memset(temp_psi_local,0,total_local_read_elements*sizeof(double complex));
    
    /* MPI-IO Setup and Read */
    /** Calculation of the offset where PSI starts in the file **/
    MPI_Offset psi_offset=0;
    FILE* c_file; // Determination of the pointer for PSI in the output file
    if (rank==0){
        c_file=fopen(P->output_imag,"rb");
        if (!c_file) {
            perror("Error opening file for header read");
            MPI_Abort(comm,1);
        }
        while (fscanf(c_file,"%31s",key)!=EOF){
            if (strcmp(key,"PSI")==0){
                psi_offset=ftell(c_file);
                while (fgetc(c_file)!='\n' && !feof(c_file)); // Consume until end of line or EOF
                psi_offset=ftell(c_file); // The *new* offset is the correct boundary for the binary block
                break;
            }
            if (strcmp(key,"DIMENSION")==0){int dimen; if (fscanf(c_file,"%d",&dimen)!=1){printf("HERE \n"); break;}}
            else if (strcmp(key,"N_STATES")==0){int nstates; if (fscanf(c_file,"%d",&nstates)!=1){break;}}
            else if (strcmp(key,"POTENTIAL")==0){if (fscanf(c_file,"%s",key)!=1){break;}}
            else if (strcmp(key,"NR")==0){ptrdiff_t nr; for (j=0;j<P_imag->dim;j++){if (fscanf(c_file,"%td",&nr)!=1) {break;}}}
            else if (strcmp(key,"DR")==0){double dr; for (j=0;j<P_imag->dim;j++){if (fscanf(c_file,"%lf",&dr)!=1) {break;}}}
            else if (strcmp(key,"E")==0){double en; for (j=0;j<P_imag->N_states;j++) {if (fscanf(c_file,"%lf",&en)!=1) {break;}}}
            else if (strcmp(key,"DE")==0){double den; for (j=0;j<P_imag->N_states;j++) {if (fscanf(c_file,"%lf",&den)!=1) {break;}}}
            else {printf("Warning: Skipping unknown key %s in Read_wavefunction_MPI\n",key); if (fscanf(c_file,"%*s")!=1) {break;}}
        }
        fclose(c_file);
    }
    MPI_Bcast(&psi_offset,1,MPI_OFFSET,0,comm); // Broadcast the offset to all ranks
    if (psi_offset==0){
        fprintf(stderr,"Rank %d: Error: Could not find 'PSI' marker in file.\n",rank);
        MPI_Abort(comm,1);
    }
    
    /** Reading the Wave Function by Chunks **/
    MPI_File file;
    int mpi_err=MPI_File_open(comm,P->output_imag,MPI_MODE_RDONLY,MPI_INFO_NULL,&file);
    if (mpi_err!=MPI_SUCCESS){
        char err_string[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(mpi_err,err_string,&len);
        fprintf(stderr,"Rank %d: Error opening MPI file: %s\n",rank,err_string);
        MPI_Abort(comm,1);
    }
    
    for (k=0;k<P->N_states;k++){
        int state_index_zero_based=P->j_states[k]-1;
        MPI_Offset state_offset_bytes=(MPI_Offset)state_index_zero_based*full_state_size_elements*sizeof(double complex); // Base offset to the start of the state data in the file
        MPI_Offset file_byte_offset_start=psi_offset+state_offset_bytes+(MPI_Offset)I_start_in_file*dim_x*sizeof(double complex); // Final calculation of the file position to start reading the local chunk
        
        /** Read the contiguous block of data directly into the local temporary buffer **/
        MPI_Count total_read_count=(MPI_Count)(total_local_read_elements*element_size_doubles); // Ensure necessary variables are defined:
        MPI_Count remaining_count=total_read_count;
        void *current_buf_ptr=temp_psi_local; // Buffer pointer needs to be updated during the loop
        MPI_Offset current_file_offset=file_byte_offset_start;
        if (N_blocks_to_read>0){
            while (remaining_count>0){ // Chunked reading loop
                int chunk_count=(remaining_count>INT_MAX) ? INT_MAX : (int)remaining_count;
                int mpi_err=MPI_File_read_at(file,current_file_offset,current_buf_ptr,chunk_count,MPI_DOUBLE,MPI_STATUS_IGNORE); // Execute the read for the current chunk
                if (mpi_err!=MPI_SUCCESS) { // Handle error (optional, but good practice)
                    fprintf(stderr,"Rank %d: MPI Read Error during chunking\n",rank);
                    MPI_Abort(comm,1);
                }
                remaining_count-=chunk_count; // Update pointers and counters for the next iteration
                current_buf_ptr=(char*)current_buf_ptr+chunk_count*sizeof(double); // Advance by bytes
                current_file_offset+=chunk_count*sizeof(double); // Advance file offset by bytes
            }
        }
        else {memset(temp_psi_local,0,total_local_read_elements*sizeof(double complex));} // If rank is outside the imaginary grid, just zero the buffer
        
        /** Process Data (Map, Re-center, Accumulate) **/
        if (N_blocks_to_read>0){
            ptrdiff_t local_read_index=0; // Index into the temp_psi_local buffer (0 to total_local_read_elements - 1)
            const ptrdiff_t G_imag_start_dim0=centering_offset[0]; // Absolute start of the imaginary grid in the global real-time frame
            for (ptrdiff_t i_x=overlap_start;i_x<overlap_end;i_x++){ // Global x-indices read
                ptrdiff_t i_imag_slow=i_x-G_imag_start_dim0; // The 0-based index in the IMAGINARY grid (slow dimension) starts at 0 for the first point read and increments sequentially
                for (ptrdiff_t i_fast=0;i_fast<dim_x;i_fast++){ // Faster indices read
                    /** Recover IMAGINARY grid coordinates from the sequential global index **/
                    ptrdiff_t i_global_imag=i_imag_slow*dim_x+i_fast;
                    Recover_indices(index,i_global_imag,P_imag->dim,P_imag->Nr_global);
                    
                    /** Map IMAGINARY coordinates to REAL grid index (ind_local) **/
                    ptrdiff_t ind_local=0;
                    ptrdiff_t Nm_local=1; // Multiplier starts at 1
                    int is_valid=1;
                    for (j=P->dim-1;j>=0;j--){
                        ptrdiff_t global_index_centered=index[j]+centering_offset[j]; // Get global coordinate of the point in the REAL grid frame
                        ptrdiff_t rank_start_j=(j==0) ? local_start_real[j] : 0; // Calculate local index (relative to this rank's start)
                        ptrdiff_t local_idx_j=global_index_centered-rank_start_j;
                        if (local_idx_j<0 || local_idx_j>=grid->Nr_local[j]){ // Guard Rail Check: Skip if point is outside this rank's local chunk
                            is_valid=0;
                            break;
                        }
                        ind_local+=local_idx_j*Nm_local; // Accumulate index: Index = i_y + N_local_y * i_x + ...
                        Nm_local*=grid->Nr_local[j];
                    }
                    
                    /** Accumulate the state into the total Psi **/
                    if (is_valid){
                        double complex psi_val=temp_psi_local[local_read_index]; // Read from the contiguous local buffer using the correct sequential index
                        Psi[ind_local]+=P->coeffs_states[k]*psi_val;
                    }
                    local_read_index++;
                }
            }
        }
    }
    
    /* Calculation of the initial state properties */
    /** Parameter for the length vs velocity gauge calculation of the observables (here 0) **/
    double *alpha=NULL;
    alpha=(double*) malloc(P->dim*sizeof(double));
    if (alpha==NULL){
        fprintf(stderr, "ERROR: memory allocation failed for alpha in Read_wavefunction_MPI\n");
        perror("malloc");
        MPI_Abort(comm,1);
    }
    for (j=0;j<P->dim;j++){alpha[j]=0.;}
    
    /** Calculation **/
    double norm_local=0.;
    double norm_global;
    for (ptrdiff_t i=0;i<grid->dim_HS;i++){norm_local+=cabs(Psi[i])*cabs(Psi[i])*grid->dNr;} // Local norm calculation
    MPI_Allreduce(&norm_local,&norm_global,1,MPI_DOUBLE,MPI_SUM,comm); // Global summation
    double E_global[2];
    Calculation_global_energies(E_global,2,Psi,grid,comm); // Calculation of the energies of the eigenstate
    Observables Obs_Lz;
    char *name_Lz="Lz";
    Initialize_observables(&Obs_Lz,&name_Lz,1,P->dim,comm);
    Calculation_global_observables(Psi,P->dim,grid,&Obs_Lz,alpha,comm);
    if (rank==0){
        printf("\n");
        printf("  norm\t\t\t%16.15f\n",sqrt(norm_global));
        printf("  E\t\t\t%16.15f\n",E_global[0]);
        printf("  DE\t\t\t%5.4e\n",E_global[1]-E_global[0]*E_global[0]);
        printf("  Lz\t\t\t%16.15f\n",Obs_Lz.values[0]);
        printf("\n");
    }
    Finalize_observables(&Obs_Lz);
    
    /* Cleanup */
    fftw_free(temp_psi_local);
    free(alpha);
    free(index);
    free(centering_offset);
    MPI_File_close(&file);
}


