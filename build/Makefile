NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include
OPENCV_FLAGS = `pkg-config opencv4 --cflags --libs`

ifdef SM
	NVCC_FLAGS  += -gencode=arch=compute_$(SM),code=\"sm_$(SM)\"
else
	NVCC_FLAGS  += -gencode=arch=compute_75,code=\"sm_75\"
endif
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = cuda_canny
OBJ	        = main.o canny.o kernels.o serial.o

default: $(EXE)

main.o: ../src/main.cu
	$(NVCC) -c -o $@ ../src/main.cu $(NVCC_FLAGS) $(OPENCV_FLAGS)

canny.o: ../src/canny.cu
	$(NVCC) -c -o $@ ../src/canny.cu $(NVCC_FLAGS) $(OPENCV_FLAGS)

kernels.o: ../src/kernels.cu
	$(NVCC) -c -o $@ ../src/kernels.cu $(NVCC_FLAGS) $(OPENCV_FLAGS)

serial.o: ../src/serial.cu
	$(NVCC) -c -o $@ ../src/serial.cu $(NVCC_FLAGS) $(OPENCV_FLAGS)


$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS) $(OPENCV_FLAGS)

clean:
	rm -rf *.o $(EXE)
