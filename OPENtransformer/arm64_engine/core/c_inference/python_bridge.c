/**
 * @file python_bridge.c
 * @brief Bridge between C wrapper and Python diffusion implementation with ASM kernels
 * 
 * This file handles the interaction between C and the optimized Python implementation with ASM kernels
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "diffusion_wrapper.h"
#include "asm_kernel_orchestrator.h"
#include <stdbool.h>
#include "python_bridge.h"
#include "c_inference_engine.h"
#include "c_inference_internal.h"

// Global Python objects
PyObject* py_module = NULL;
PyObject* py_inference_func = NULL;
PyObject* py_sys_module = NULL;
static bool python_initialized = false;

// ASM kernel context
ASMKernelContext* asm_ctx = NULL;

// Forward declarations
static int ensure_python_initialized(void);
static int initialize_asm_kernels(void);
static void cleanup_asm_kernels(void);
static int verify_asm_kernels(void);
static int install_python_dependencies(void);
PyObject* c_progress_callback(PyObject* self, PyObject* args);
PyMODINIT_FUNC PyInit_c_callback(void);

// Clean up ASM kernel system
static void cleanup_asm_kernels(void) {
    if (asm_ctx) {
        free_asm_kernels(asm_ctx);
        asm_ctx = NULL;
    }
}

// Initialize Python with the correct interpreter
static int ensure_python_initialized(void) {
    if (python_initialized) {
        return 0;
    }

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        fprintf(stderr, "Failed to get current working directory\n");
        return -1;
    }

    // Initialize Python with configuration
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    // Set Python home to the virtual environment
    char python_home[PATH_MAX];
    snprintf(python_home, sizeof(python_home), "%s/venv", cwd);
    wchar_t wpython_home[PATH_MAX];
    mbstowcs(wpython_home, python_home, PATH_MAX);
    PyStatus status = PyConfig_SetString(&config, &config.home, wpython_home);
    if (PyStatus_Exception(status)) {
        fprintf(stderr, "Failed to set Python home\n");
        PyConfig_Clear(&config);
        return -1;
    }

    // Set Python executable path
    char python_exe[PATH_MAX];
    snprintf(python_exe, sizeof(python_exe), "%s/venv/bin/python3", cwd);
    wchar_t wpython_exe[PATH_MAX];
    mbstowcs(wpython_exe, python_exe, PATH_MAX);
    status = PyConfig_SetString(&config, &config.program_name, wpython_exe);
    if (PyStatus_Exception(status)) {
        fprintf(stderr, "Failed to set program name\n");
        PyConfig_Clear(&config);
        return -1;
    }

    // Initialize Python with the configuration
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) {
        fprintf(stderr, "Failed to initialize Python\n");
        PyConfig_Clear(&config);
        return -1;
    }
    PyConfig_Clear(&config);

    // Import sys module to modify Python path
    PyObject* sys = PyImport_ImportModule("sys");
    if (!sys) {
        fprintf(stderr, "Failed to import sys module\n");
        Py_Finalize();
        return -1;
    }

    // Get sys.path
    PyObject* sys_path = PyObject_GetAttrString(sys, "path");
    if (!sys_path) {
        fprintf(stderr, "Failed to get sys.path\n");
        Py_DECREF(sys);
        Py_Finalize();
        return -1;
    }

    // Add current directory to Python path
    PyObject* cwd_path = PyUnicode_FromString("");
    if (!cwd_path) {
        fprintf(stderr, "Failed to create current directory path string\n");
        Py_DECREF(sys_path);
        Py_DECREF(sys);
        Py_Finalize();
        return -1;
    }
    PyList_Insert(sys_path, 0, cwd_path);
    Py_DECREF(cwd_path);

    // Print debug information
    PyObject* path_list = PyObject_Str(sys_path);
    if (path_list) {
        const char* path_str = PyUnicode_AsUTF8(path_list);
        printf("Python path: %s\n", path_str);
        Py_DECREF(path_list);
    }
    Py_DECREF(sys_path);
    Py_DECREF(sys);

    // Verify required modules can be imported
    PyObject* diffusers = PyImport_ImportModule("diffusers");
    if (!diffusers) {
        PyErr_Print();
        fprintf(stderr, "Failed to import diffusers module\n");
        Py_Finalize();
        return -1;
    }
    Py_DECREF(diffusers);

    python_initialized = 1;
    return 0;
}

// Initialize ASM kernels separately
static int initialize_asm_kernels(void) {
    if (asm_ctx) return 0;  // Already initialized
    
    printf("Initializing ASM kernel system...\n");
    asm_ctx = init_asm_kernels();
    if (!asm_ctx) {
        fprintf(stderr, "Failed to create ASM kernel context\n");
        return -1;
    }
    
    if (!asm_is_simd_available()) {
        fprintf(stderr, "Warning: SIMD operations not available\n");
    } else {
        asm_enable_simd(asm_ctx);
        printf("SIMD operations enabled\n");
    }
    
    return 0;
}

// Initialize Python bridge with separate steps
int init_python_bridge(void) {
    // Step 1: Initialize Python environment
    if (ensure_python_initialized() != 0) {
        fprintf(stderr, "Failed to initialize Python environment\n");
        return -1;
    }
    
    // Step 2: Initialize ASM kernels
    if (initialize_asm_kernels() != 0) {
        fprintf(stderr, "Failed to initialize ASM kernels\n");
        Py_Finalize();
        return -1;
    }
    
    // Step 3: Verify ASM kernels are working
    if (verify_asm_kernels() != 0) {
        fprintf(stderr, "Failed to verify ASM kernels\n");
        cleanup_asm_kernels();
        Py_Finalize();
        return -1;
    }
    
    printf("Python bridge initialized successfully\n");
    return 0;
}

// Method definition for the callback
PyMethodDef CallbackMethods[] = {
    {"c_progress_callback", c_progress_callback, METH_VARARGS, 
     "Callback function for reporting progress to C"},
    {NULL, NULL, 0, NULL}
};

// Module definition for the callback module
static struct PyModuleDef callbackmodule = {
    PyModuleDef_HEAD_INIT,
    "c_callback",
    "Module for C callbacks",
    -1,
    CallbackMethods,
    NULL,  // m_slots
    NULL,  // m_traverse
    NULL,  // m_clear
    NULL   // m_free
};

// Initialize the callback module
PyMODINIT_FUNC PyInit_c_callback(void) {
    return PyModule_Create(&callbackmodule);
}

// Callback function to be called from Python
PyObject* c_progress_callback(PyObject* self, PyObject* args) {
    (void)self;  // Unused parameter
    
    int step, total_steps;
    float step_time;
    
    if (!PyArg_ParseTuple(args, "iif", &step, &total_steps, &step_time)) {
        return NULL;
    }
    
    // Print progress
    printf("\rStep %d/%d (%.1f%%)", step, total_steps,
           (float)step / total_steps * 100.0f);
    fflush(stdout);
    
    Py_RETURN_NONE;
}

// Function to verify ASM kernel module is loaded and working
static int verify_asm_kernels(void) {
    printf("Attempting to load ASM kernels...\n");
    
    // Try to import the ASM kernel module
    py_module = PyImport_ImportModule("OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.diffusion_kernels");
    if (!py_module) {
        fprintf(stderr, "ERROR: Failed to import ASM kernel module. This is required for optimized execution.\n");
        PyErr_Print();
        return -1;  // Fail if ASM kernels are not available
    }
    
    // Get the DiffusionKernels class
    PyObject* diffusion_kernels_class = PyObject_GetAttrString(py_module, "DiffusionKernels");
    if (!diffusion_kernels_class) {
        fprintf(stderr, "ERROR: Failed to get DiffusionKernels class\n");
        PyErr_Print();
        Py_DECREF(py_module);
        return -1;
    }
    
    // Create an instance of DiffusionKernels
    PyObject* args = PyTuple_New(0);
    PyObject* kwargs = PyDict_New();
    PyObject* diffusion_kernels_instance = PyObject_Call(diffusion_kernels_class, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(diffusion_kernels_class);
    
    if (!diffusion_kernels_instance) {
        fprintf(stderr, "ERROR: Failed to create DiffusionKernels instance\n");
        PyErr_Print();
        Py_DECREF(py_module);
        return -1;
    }
    
    // Store the instance in the module for later use
    PyObject_SetAttrString(py_module, "_instance", diffusion_kernels_instance);
    
    return 0;
}

// Function to load model using diffusers
PyObject* load_model_with_diffusers(const char* model_path) {
    PyObject* diffusers = PyImport_ImportModule("diffusers");
    if (!diffusers) {
        fprintf(stderr, "Failed to import diffusers module\n");
        return NULL;
    }
    
    // Get StableDiffusionPipeline class
    PyObject* pipeline_class = PyObject_GetAttrString(diffusers, "StableDiffusionPipeline");
    Py_DECREF(diffusers);  // Done with diffusers module
    
    if (!pipeline_class) {
        fprintf(stderr, "Failed to get StableDiffusionPipeline class\n");
        return NULL;
    }
    
    // Get from_pretrained method
    PyObject* from_pretrained = PyObject_GetAttrString(pipeline_class, "from_pretrained");
    Py_DECREF(pipeline_class);  // Done with pipeline class
    
    if (!from_pretrained) {
        fprintf(stderr, "Failed to get from_pretrained method\n");
        return NULL;
    }
    
    // Create args tuple with model path
    PyObject* model_path_str = PyUnicode_FromString(model_path ? model_path : "runwayml/stable-diffusion-v1-5");
    if (!model_path_str) {
        fprintf(stderr, "Failed to create model path string\n");
        Py_DECREF(from_pretrained);
        return NULL;
    }
    
    PyObject* args = PyTuple_Pack(1, model_path_str);
    Py_DECREF(model_path_str);  // PyTuple_Pack increases refcount
    
    if (!args) {
        fprintf(stderr, "Failed to create args tuple\n");
        Py_DECREF(from_pretrained);
        return NULL;
    }
    
    // Create kwargs dictionary
    PyObject* kwargs = PyDict_New();
    if (!kwargs) {
        fprintf(stderr, "Failed to create kwargs dict\n");
        Py_DECREF(args);
        Py_DECREF(from_pretrained);
        return NULL;
    }
    
    // Set torch_dtype to float32
    PyObject* torch = PyImport_ImportModule("torch");
    if (!torch) {
        fprintf(stderr, "Failed to import torch module\n");
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(from_pretrained);
        return NULL;
    }
    
    PyObject* float32 = PyObject_GetAttrString(torch, "float32");
    Py_DECREF(torch);  // Done with torch module
    
    if (!float32) {
        fprintf(stderr, "Failed to get torch.float32\n");
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(from_pretrained);
        return NULL;
    }
    
    if (PyDict_SetItemString(kwargs, "torch_dtype", float32) < 0) {
        fprintf(stderr, "Failed to set torch_dtype\n");
        Py_DECREF(float32);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(from_pretrained);
        return NULL;
    }
    Py_DECREF(float32);  // PyDict_SetItemString increases refcount
    
    if (PyDict_SetItemString(kwargs, "use_safetensors", Py_True) < 0) {
        fprintf(stderr, "Failed to set use_safetensors\n");
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(from_pretrained);
        return NULL;
    }
    
    // Call from_pretrained
    PyObject* pipeline = PyObject_Call(from_pretrained, args, kwargs);
    Py_DECREF(kwargs);
    Py_DECREF(args);
    Py_DECREF(from_pretrained);
    
    if (!pipeline) {
        fprintf(stderr, "Failed to create pipeline instance\n");
        PyErr_Print();
        return NULL;
    }
    
    // Move to CPU
    PyObject* to_method = PyObject_GetAttrString(pipeline, "to");
    if (!to_method) {
        fprintf(stderr, "Failed to get to method\n");
        Py_DECREF(pipeline);
        return NULL;
    }
    
    PyObject* device_str = PyUnicode_FromString("cpu");
    if (!device_str) {
        fprintf(stderr, "Failed to create device string\n");
        Py_DECREF(to_method);
        Py_DECREF(pipeline);
        return NULL;
    }
    
    PyObject* device_args = PyTuple_Pack(1, device_str);
    Py_DECREF(device_str);  // PyTuple_Pack increases refcount
    
    if (!device_args) {
        fprintf(stderr, "Failed to create device args\n");
        Py_DECREF(to_method);
        Py_DECREF(pipeline);
        return NULL;
    }
    
    PyObject* pipeline_on_cpu = PyObject_Call(to_method, device_args, NULL);
    Py_DECREF(device_args);
    Py_DECREF(to_method);
    Py_DECREF(pipeline);
    
    if (!pipeline_on_cpu) {
        fprintf(stderr, "Failed to move pipeline to CPU\n");
        PyErr_Print();
        return NULL;
    }
    
    return pipeline_on_cpu;
}

// Function to prepare model for C inference
int prepare_model_for_c_inference(PyObject* pipeline, InferenceContext* ctx) {
    if (!pipeline || !ctx) return -1;
    
    // Get model components
    PyObject* unet = PyObject_GetAttrString(pipeline, "unet");
    if (!unet) {
        fprintf(stderr, "Failed to get unet\n");
        return -1;
    }
    
    PyObject* vae = PyObject_GetAttrString(pipeline, "vae");
    if (!vae) {
        fprintf(stderr, "Failed to get vae\n");
        Py_DECREF(unet);
        return -1;
    }
    
    PyObject* text_encoder = PyObject_GetAttrString(pipeline, "text_encoder");
    if (!text_encoder) {
        fprintf(stderr, "Failed to get text_encoder\n");
        Py_DECREF(vae);
        Py_DECREF(unet);
        return -1;
    }
    
    // Get state dictionaries
    PyObject* unet_state = PyObject_CallMethod(unet, "state_dict", NULL);
    if (!unet_state) {
        fprintf(stderr, "Failed to get unet state dict\n");
        Py_DECREF(text_encoder);
        Py_DECREF(vae);
        Py_DECREF(unet);
        return -1;
    }
    
    PyObject* vae_state = PyObject_CallMethod(vae, "state_dict", NULL);
    if (!vae_state) {
        fprintf(stderr, "Failed to get vae state dict\n");
        Py_DECREF(unet_state);
        Py_DECREF(text_encoder);
        Py_DECREF(vae);
        Py_DECREF(unet);
        return -1;
    }
    
    PyObject* text_encoder_state = PyObject_CallMethod(text_encoder, "state_dict", NULL);
    if (!text_encoder_state) {
        fprintf(stderr, "Failed to get text_encoder state dict\n");
        Py_DECREF(vae_state);
        Py_DECREF(unet_state);
        Py_DECREF(text_encoder);
        Py_DECREF(vae);
        Py_DECREF(unet);
        return -1;
    }
    
    // Store references to state dictionaries
    ctx->weights->unet_weights = unet_state;
    ctx->weights->vae_weights = vae_state;
    ctx->weights->text_encoder_weights = text_encoder_state;
    
    // Clean up component references
    Py_DECREF(text_encoder);
    Py_DECREF(vae);
    Py_DECREF(unet);
    
    // Note: We don't DECREF the state dictionaries since we're storing them
    
    ctx->weights_loaded = true;
    return 0;
}

// Function to clean up Python resources
void cleanup_python_bridge(void) {
    Py_XDECREF(py_inference_func);
    Py_XDECREF(py_module);
    Py_XDECREF(py_sys_module);
    
    cleanup_asm_kernels();
}

// Main inference function
int call_python_inference(
    const char* model_path,
    const char* prompt,
    int num_inference_steps,
    float guidance_scale,
    int width,
    int height,
    uint32_t seed,
    bool use_memory_optimizations,
    uint8_t** out_data,
    int* out_width,
    int* out_height,
    int* out_channels,
    DiffusionPerfCallback callback,
    void* user_data) {
    
    PyObject* py_result = NULL;
    
    // Initialize Python bridge if needed
    if (init_python_bridge() != 0) {
        fprintf(stderr, "Failed to initialize Python bridge\n");
        return -1;
    }
    
    // Import the Python diffusion interface module
    PyObject* py_interface = PyImport_ImportModule("py_diffusion_interface");
    if (!py_interface) {
        fprintf(stderr, "Failed to import py_diffusion_interface module\n");
        PyErr_Print();
        
        // Try to find the module
        PyRun_SimpleString(
            "import sys\n"
            "import os\n"
            "print('Working directory:', os.getcwd())\n"
            "print('Python path:', sys.path)\n"
            "try:\n"
            "    import py_diffusion_interface\n"
            "    print('Found py_diffusion_interface at:', py_diffusion_interface.__file__)\n"
            "except ImportError as e:\n"
            "    print('Error importing py_diffusion_interface:', e)\n"
            "    # Try to find the file\n"
            "    found = False\n"
            "    for path in sys.path:\n"
            "        potential_path = os.path.join(path, 'py_diffusion_interface.py')\n"
            "        if os.path.exists(potential_path):\n"
            "            print('Found module at:', potential_path)\n"
            "            found = True\n"
            "    if not found:\n"
            "        print('Module not found in any path')\n"
        );
        
        // Manually add the module to sys.path
        PyRun_SimpleString(
            "import sys\n"
            "import os\n"
            "current_dir = os.getcwd()\n"
            "if current_dir not in sys.path:\n"
            "    sys.path.append(current_dir)\n"
            "print('Updated Python path:', sys.path)\n"
        );
        
        // Try again after updating path
        py_interface = PyImport_ImportModule("py_diffusion_interface");
        if (!py_interface) {
            fprintf(stderr, "Failed to import py_diffusion_interface module after path update\n");
            PyErr_Print();
            return -1;
        }
    }
    
    // Get the run_inference function
    PyObject* py_inference_func = PyObject_GetAttrString(py_interface, "run_inference");
    if (!py_inference_func) {
        fprintf(stderr, "Failed to get run_inference function\n");
        PyErr_Print();
        Py_DECREF(py_interface);
        return -1;
    }
    
    // Create arguments for the Python function
    PyObject* py_args = Py_BuildValue("ssiifiiOKK", 
        model_path ? model_path : "runwayml/stable-diffusion-v1-5", // Default model if NULL
        prompt ? prompt : "a beautiful landscape",                   // Default prompt if NULL
        num_inference_steps > 0 ? num_inference_steps : 7,          // Default steps
        width > 0 ? width : 512,                                    // Default width
        guidance_scale > 0 ? guidance_scale : 7.5f,                 // Default guidance scale
        height > 0 ? height : 512,                                  // Default height
        seed,                                                       // Seed as provided
        use_memory_optimizations ? Py_True : Py_False,              // Memory optimizations
        (unsigned long long)(callback ? callback : NULL),           // Callback function pointer
        (unsigned long long)user_data);                             // User data pointer
    
    if (!py_args) {
        fprintf(stderr, "Failed to build Python arguments\n");
        PyErr_Print();
        Py_DECREF(py_inference_func);
        Py_DECREF(py_interface);
        return -1;
    }
    
    // Call the Python function
    printf("Calling Python inference function with prompt: %s\n", prompt);
    py_result = PyObject_CallObject(py_inference_func, py_args);
    Py_DECREF(py_args);
    Py_DECREF(py_inference_func);
    
    if (!py_result) {
        fprintf(stderr, "Python inference call failed\n");
        PyErr_Print();
        Py_DECREF(py_interface);
        return -1;
    }
    
    // Parse the result tuple (width, height, channels, data)
    PyObject* py_width = NULL;
    PyObject* py_height = NULL;
    PyObject* py_channels = NULL;
    PyObject* py_image_data = NULL;
    
    if (!PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
        fprintf(stderr, "Invalid result format from Python (expected tuple of size 4)\n");
        Py_DECREF(py_result);
        Py_DECREF(py_interface);
        return -1;
    }
    
    py_width = PyTuple_GetItem(py_result, 0);     // Borrowed reference
    py_height = PyTuple_GetItem(py_result, 1);    // Borrowed reference
    py_channels = PyTuple_GetItem(py_result, 2);  // Borrowed reference
    py_image_data = PyTuple_GetItem(py_result, 3); // Borrowed reference
    
    if (!py_width || !py_height || !py_channels || !py_image_data) {
        fprintf(stderr, "Invalid result tuple content\n");
        Py_DECREF(py_result);
        Py_DECREF(py_interface);
        return -1;
    }
    
    // Convert Python integers to C
    *out_width = (int)PyLong_AsLong(py_width);
    *out_height = (int)PyLong_AsLong(py_height);
    *out_channels = (int)PyLong_AsLong(py_channels);
    
    // Extract image data from bytes object
    if (!PyBytes_Check(py_image_data)) {
        fprintf(stderr, "Image data is not bytes (type=%s)\n", Py_TYPE(py_image_data)->tp_name);
        Py_DECREF(py_result);
        Py_DECREF(py_interface);
        return -1;
    }
    
    Py_ssize_t image_size = PyBytes_Size(py_image_data);
    if (image_size <= 0) {
        fprintf(stderr, "Invalid image data size: %zd\n", image_size);
        Py_DECREF(py_result);
        Py_DECREF(py_interface);
        return -1;
    }
    
    // Allocate memory for image data
    *out_data = (uint8_t*)malloc(image_size);
    if (!*out_data) {
        fprintf(stderr, "Failed to allocate memory for image data (%zd bytes)\n", image_size);
        Py_DECREF(py_result);
        Py_DECREF(py_interface);
        return -1;
    }
    
    // Copy image data
    memcpy(*out_data, PyBytes_AsString(py_image_data), image_size);
    
    printf("Successfully received image data from Python (%dx%d with %d channels, %zd bytes)\n", 
           *out_width, *out_height, *out_channels, image_size);
    
    // Clean up
    Py_DECREF(py_result);
    Py_DECREF(py_interface);
    
    return 0;
}

// Function to install required Python dependencies
static int install_python_dependencies(void) {
    printf("Attempting to install required Python dependencies...\n");
    
    // Run pip to install diffusers and torch
    PyRun_SimpleString(
        "import sys\n"
        "import subprocess\n"
        "try:\n"
        "    print('Installing diffusers...')\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'diffusers'])\n"
        "    print('Installing torch...')\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])\n"
        "    print('Successfully installed dependencies')\n"
        "except subprocess.CalledProcessError as e:\n"
        "    print('Failed to install dependencies:', e)\n"
        "    sys.exit(1)\n"
    );
    
    // Verify installations
    PyObject* diffusers = PyImport_ImportModule("diffusers");
    if (!diffusers) {
        PyErr_Print();
        fprintf(stderr, "Failed to import diffusers module after installation attempt\n");
        return -1;
    }
    Py_DECREF(diffusers);
    
    PyObject* torch = PyImport_ImportModule("torch");
    if (!torch) {
        PyErr_Print();
        fprintf(stderr, "Failed to import torch module after installation attempt\n");
        return -1;
    }
    Py_DECREF(torch);
    
    printf("Successfully installed and verified Python dependencies\n");
    return 0;
}

int init_python_environment(void) {
    Py_Initialize();
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Failed to initialize Python interpreter\n");
        return -1;
    }
    
    // Import required modules
    PyRun_SimpleString(
        "import sys\n"
        "import os\n"
        "import site\n"
        "print('Python path:', sys.path)\n"
        "print('Current directory:', os.getcwd())\n"
        "\n"
        "# Add current directory and project root to path\n"
        "sys.path.append(os.getcwd())\n"
        "project_root = os.path.abspath(os.path.join(os.getcwd(), '../../..'))\n"
        "sys.path.append(project_root)\n"
        "\n"
        "# Add virtual environment site-packages if exists\n"
        "venv_path = os.path.join(project_root, 'venv')\n"
        "if os.path.exists(venv_path):\n"
        "    venv_site_packages = os.path.join(venv_path, 'lib', 'python%s' % sys.version[:3], 'site-packages')\n"
        "    if os.path.exists(venv_site_packages):\n"
        "        sys.path.append(venv_site_packages)\n"
        "        print('Added venv site-packages:', venv_site_packages)\n"
        "\n"
        "# Try to import diffusers\n"
        "try:\n"
        "    import diffusers\n"
        "    print('Successfully imported diffusers:', diffusers.__version__)\n"
        "except ImportError as e:\n"
        "    print('Failed to import diffusers:', e)\n"
        "    print('Please install diffusers with: pip install diffusers')\n"
        "except AttributeError:\n"
        "    print('Successfully imported diffusers, but version info not available')\n"
    );
    
    // Try to import required modules
    PyObject* diffusers = PyImport_ImportModule("diffusers");
    PyObject* torch = PyImport_ImportModule("torch");
    
    // If either module is missing, try to install dependencies
    if (!diffusers || !torch) {
        Py_XDECREF(diffusers);
        Py_XDECREF(torch);
        
        printf("Missing required Python dependencies. Attempting to install...\n");
        if (install_python_dependencies() != 0) {
            fprintf(stderr, "Failed to install required Python dependencies\n");
            return -1;
        }
    } else {
        Py_DECREF(diffusers);
        Py_DECREF(torch);
    }
    
    printf("Python environment initialized successfully\n");
    return 0;
}

void cleanup_python_environment(void) {
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
} 