CC := cc
NVCC := nvcc
CFLAGS := -Wall -Wpedantic -Werror -lm
NCFLAGS := -Wall -Werror -lm

ifdef DEBUG
CFLAGS += -g -O0 -DDEBUG
NCFLAGS += -g -O0 -DDEBUG
else
CFLAGS += -O3
NCFLAGS += -O3
endif

ifdef PROGRESS
CFLAGS += -DPROGRESS
endif

all: gaussian_blur_serial gaussian_blur_cuda

gaussian_blur_serial: gaussian_blur_serial.o
	$(CC) $^ -o $@ $(CFLAGS)

gaussian_blur_cuda: gaussian_blur_cuda.o
	$(NVCC) $^ -o $@ -Xcompiler "$(NCFLAGS)"

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cu
	$(NVCC) -c -o $@ $< -Xcompiler "$(NCFLAGS)"

clean:
	rm -f gaussian_blur_serial
	rm -f gaussian_blur_cuda
	rm -f gaussian_blur_serial.o
	rm -f gaussian_blur_cuda.o

.PHONY: all clean

