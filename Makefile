# =======================================================
# Compiler and Flags
# =======================================================
CC = mpicc

CFLAGS = -Wall -O3 -march=native -ffast-math -fopenmp -c # -c flag means "compile only"
LDFLAGS = -fopenmp -lfftw3_mpi -lfftw3 -lfftw3_threads -lpthread -lm

MPI_INC = -I/opt/homebrew/opt/fftw/include
MPI_LIB = -L/opt/homebrew/opt/fftw/lib

# Directories
SRC_DIR = src
BIN_DIR = .
OBJ_DIR = obj

# =======================================================
# Files and Targets
# =======================================================
TARGETS = QuantumND_imag QuantumND_real

# Define ALL source files needed to build the executable
COMMON_SRC = \
	$(SRC_DIR)/calculate_observables.c \
	$(SRC_DIR)/parameters.c \
	$(SRC_DIR)/meshgrid.c \
	$(SRC_DIR)/integrators.c \
	$(SRC_DIR)/io_functions.c

# Calculate Object Files (.o) from common sources
COMMON_OBJ = $(COMMON_SRC:.c=.o)

# Calculate ALL required object files (.o)
OBJ_IMAG = $(SRC_DIR)/QuantumND_imag.o
OBJ_REAL = $(SRC_DIR)/QuantumND_real.o

# Full list of objects for each executable
FULL_OBJ_IMAG = $(COMMON_OBJ) $(OBJ_IMAG)
FULL_OBJ_REAL = $(COMMON_OBJ) $(OBJ_REAL)

# =======================================================
# Rules
# =======================================================

.PHONY: all clean

all: $(TARGETS)

# Rule to compile a generic .c file into a .o file
# This handles the compilation for ALL files in COMMON_SRC and QuantumND_imag.c
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(MPI_INC) -o $@ $<

# Target: QuantumND_imag (Linking Step)
QuantumND_imag: $(FULL_OBJ_IMAG)
	$(CC) $^ $(MPI_LIB) $(LDFLAGS) -o $@

QuantumND_real: $(FULL_OBJ_REAL)
	$(CC) $^ $(MPI_LIB) $(LDFLAGS) -o $@

# -- Dependency Rules (Ensure rebuilding when headers change) --
# Create the object files needed for the imaginary executable
# This rule now only lists dependencies, relying on the generic rule above for the command.
$(SRC_DIR)/calculate_observables.o: $(SRC_DIR)/calculate_observables.c $(SRC_DIR)/meshgrid.h
$(SRC_DIR)/parameters.o: $(SRC_DIR)/parameters.c $(SRC_DIR)/parameters.h
$(SRC_DIR)/meshgrid.o: $(SRC_DIR)/meshgrid.c $(SRC_DIR)/meshgrid.h $(SRC_DIR)/parameters.h
$(SRC_DIR)/integrators.o: $(SRC_DIR)/integrators.c $(SRC_DIR)/integrators.h $(SRC_DIR)/meshgrid.h $(SRC_DIR)/parameters.h
$(SRC_DIR)/io_functions.o: $(SRC_DIR)/io_functions.c $(SRC_DIR)/io_functions.h $(SRC_DIR)/calculate_observables.h $(SRC_DIR)/meshgrid.h $(SRC_DIR)/parameters.h

# --- Dependency for the shared object ---
# This rule now only lists dependencies.
$(SRC_DIR)/QuantumND_imag.o: $(SRC_DIR)/QuantumND_imag.c $(SRC_DIR)/integrators.h $(SRC_DIR)/meshgrid.h $(SRC_DIR)/parameters.h $(SRC_DIR)/calculate_observables.h $(SRC_DIR)/io_functions.h
$(SRC_DIR)/QuantumND_real.o: $(SRC_DIR)/QuantumND_real.c $(SRC_DIR)/integrators.h $(SRC_DIR)/meshgrid.h $(SRC_DIR)/parameters.h $(SRC_DIR)/calculate_observables.h $(SRC_DIR)/io_functions.h

# Clean rule
clean:
	rm -f $(TARGETS)
	rm -f $(SRC_DIR)/*.o
