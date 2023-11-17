NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include
OPENCV_FLAGS = `pkg-config opencv4 --cflags --libs`

ifdef rtx38ti
	NVCC_FLAGS  += -gencode=arch=compute_86,code=\"sm_86\"
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
OBJ	        = main.o canny.o

default: $(EXE)

main.o: main.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) $(OPENCV_FLAGS)

canny.o: canny.cu
	$(NVCC) -c -o $@ canny.cu $(NVCC_FLAGS) $(OPENCV_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS) $(OPENCV_FLAGS)

clean:
	rm -rf *.o $(EXE)
