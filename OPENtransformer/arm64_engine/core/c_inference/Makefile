# Makefile for Stable Diffusion C Wrapper

CC = gcc

# Python configuration - try to automatically detect Python setup
PYTHON = python3
PYTHON_VERSION = $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo 3.13)
HOMEBREW_PREFIX = $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)

# Try multiple locations for Python headers
PYTHON_INCLUDE ?= $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
ifeq ($(PYTHON_INCLUDE),)
  # Fallback to homebrew location
  PYTHON_INCLUDE = $(HOMEBREW_PREFIX)/opt/python@$(PYTHON_VERSION)/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/include/python$(PYTHON_VERSION)
  # Check if miniconda/anaconda is available
  CONDA_PREFIX ?= $(HOME)/miniconda3
  ifneq ($(wildcard $(CONDA_PREFIX)),)
    PYTHON_INCLUDE = $(CONDA_PREFIX)/include/python$(PYTHON_VERSION)
  endif
  # Also try the system Python location if headers aren't found
  ifeq ($(wildcard $(PYTHON_INCLUDE)/Python.h),)
    SYSTEM_PYTHON_INCLUDE = /opt/homebrew/Caskroom/miniconda/base/include/python$(PYTHON_VERSION)
    ifneq ($(wildcard $(SYSTEM_PYTHON_INCLUDE)/Python.h),)
      PYTHON_INCLUDE = $(SYSTEM_PYTHON_INCLUDE)
    endif
  endif
endif

# Check if Python.h exists in the specified location
ifeq ($(wildcard $(PYTHON_INCLUDE)/Python.h),)
  $(warning Python.h not found in $(PYTHON_INCLUDE))
  $(warning Using find to locate Python.h...)
  PYTHON_INCLUDE = $(shell find /opt/homebrew -name Python.h -type f | head -n 1 | xargs dirname 2>/dev/null || echo "/usr/include/python$(PYTHON_VERSION)")
  $(warning Found Python.h at: $(PYTHON_INCLUDE))
endif

# Python library location
PYTHON_LIB = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null || echo "$(HOMEBREW_PREFIX)/opt/python@$(PYTHON_VERSION)/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/lib")

# Numpy include path
NUMPY_INCLUDE = $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())" 2>/dev/null || echo ".")

# Print Python configuration for debugging
$(info Python include path: $(PYTHON_INCLUDE))
$(info Python lib directory: $(PYTHON_LIB))

CFLAGS = -O3 -Wall -Wextra -march=native -mtune=native -fPIC -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
LDFLAGS = -L$(PYTHON_LIB) -lpython$(PYTHON_VERSION) -Wl,-rpath,$(PYTHON_LIB)

# Source files for LLM inference
SRCS = c_inference_engine.c \
       asm_kernel_orchestrator.c \
       python_bridge.c \
       diffusion_kernels.c \
       weight_converter.c \
       diffusion_wrapper.c \
       llm_inference_api.c

OBJS = $(SRCS:.c=.o)

# Target shared library
TARGET = libllm_inference.so

# Targets
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)

# Show Python configuration
python-config:
	@echo "=== Python Configuration ==="
	@echo "Python executable: $(PYTHON)"
	@echo "Python version: $(PYTHON_VERSION)"
	@echo "Python include path: $(PYTHON_INCLUDE)"
	@echo "Python lib directory: $(PYTHON_LIB)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "=== Python Path Info ==="
	@$(PYTHON) -c "import sys; print('Executable:', sys.executable)"
	@$(PYTHON) -c "import sys; print('Path:', sys.path)"

# Compile individual source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJS) $(TARGET)

# Deep clean (also remove Python cache)
distclean: clean
	rm -rf __pycache__ *.egg-info

.PHONY: all test test-detailed clean distclean python-config py_diffusion_interface debug

# Dependencies
diffusion_wrapper.o: diffusion_wrapper.c diffusion_wrapper.h
python_bridge.o: python_bridge.c diffusion_wrapper.h asm_kernel_orchestrator.h
asm_kernel_orchestrator.o: asm_kernel_orchestrator.c asm_kernel_orchestrator.h
diffusion_kernels.o: diffusion_kernels.c diffusion_kernels.h
weight_converter.o: weight_converter.c weight_converter.h
llm_inference_api.o: llm_inference_api.c c_inference_engine.h 