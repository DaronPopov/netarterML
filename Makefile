CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC -I/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13
LDFLAGS = -L/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib -lpython3.13

SRCS = c_inference_engine.c image_utils.c diffusion_kernels.c
OBJS = $(SRCS:.c=.o)
TARGET = inference_engine

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) 