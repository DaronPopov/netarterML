import ctypes
import numpy as np
import logging
import time
from typing import Any, Callable, List, Optional, Tuple, Union, TypeVar, Generic
from functools import wraps

logger = logging.getLogger(__name__)

# Define type variables for better type hints
T = TypeVar('T')
FloatPtr = ctypes.POINTER(ctypes.c_float)
IntPtr = ctypes.POINTER(ctypes.c_int)

class KernelError(Exception):
    """Custom exception for kernel-related errors."""
    pass

def validate_pointer(ptr: Any, expected_type: type) -> bool:
    """Validate if a pointer is of the expected type."""
    try:
        if isinstance(ptr, np.ndarray):
            return ptr.dtype == expected_type
        return isinstance(ptr, ctypes.POINTER(expected_type))
    except:
        return False

def convert_to_pointer(value: Any, expected_type: type) -> ctypes.POINTER:
    """Convert a value to the appropriate pointer type."""
    if isinstance(value, np.ndarray):
        return value.ctypes.data_as(ctypes.POINTER(expected_type))
    elif isinstance(value, ctypes.POINTER(expected_type)):
        return value
    else:
        raise TypeError(f"Cannot convert {type(value)} to pointer of type {expected_type}")

def safe_kernel_execution(
    kernel_func: Callable,
    arg_types: List[Tuple[str, type]],
    timeout: float = 10.0,
    retry_count: int = 3
) -> Callable:
    """
    Decorator that adds safety checks and argument validation to kernel execution.
    
    Args:
        kernel_func: The kernel function to wrap
        arg_types: List of tuples containing (arg_name, expected_type)
        timeout: Maximum execution time in seconds
        retry_count: Number of times to retry on failure
    """
    @wraps(kernel_func)
    def wrapper(*args, **kwargs):
        # Validate number of arguments
        if len(args) != len(arg_types):
            raise KernelError(f"Expected {len(arg_types)} arguments, got {len(args)}")

        # Convert and validate arguments
        converted_args = []
        for i, (arg, (name, expected_type)) in enumerate(zip(args, arg_types)):
            try:
                if isinstance(arg, expected_type):
                    # If argument is already of the correct type, use it as is
                    converted_args.append(arg)
                elif isinstance(expected_type, type) and issubclass(expected_type, ctypes._Pointer):
                    # Handle pointer types
                    if isinstance(arg, np.ndarray):
                        converted_args.append(arg.ctypes.data_as(expected_type))
                    else:
                        converted_args.append(arg)
                elif isinstance(expected_type, type) and issubclass(expected_type, ctypes.c_int):
                    # Handle integer types
                    converted_args.append(expected_type(int(arg)))
                elif isinstance(expected_type, type) and issubclass(expected_type, ctypes.c_float):
                    # Handle float types
                    converted_args.append(expected_type(float(arg)))
                else:
                    # Handle other cases
                    converted_args.append(arg)
            except Exception as e:
                raise KernelError(f"Error converting argument {name}: {str(e)}")

        # Execute kernel with retries and timeout
        last_error = None
        for attempt in range(retry_count):
            try:
                start_time = time.time()
                result = kernel_func(*converted_args)
                execution_time = time.time() - start_time

                if execution_time > timeout:
                    logger.warning(f"Kernel execution took {execution_time:.2f}s, exceeding timeout of {timeout}s")

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Kernel execution attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(0.1)  # Small delay between retries

        raise KernelError(f"Kernel execution failed after {retry_count} attempts. Last error: {str(last_error)}")

    return wrapper

class KernelWrapper:
    """A class to wrap kernel functions with safety checks and argument validation."""
    
    def __init__(self):
        self.kernels = {}
        self.arg_types = {}
        
    def register_kernel(
        self,
        name: str,
        kernel_func: Callable,
        arg_types: List[Tuple[str, type]],
        timeout: float = 10.0,
        retry_count: int = 3
    ) -> None:
        """
        Register a kernel with safety checks.
        
        Args:
            name: Name of the kernel
            kernel_func: The kernel function to wrap
            arg_types: List of tuples containing (arg_name, expected_type)
            timeout: Maximum execution time in seconds
            retry_count: Number of times to retry on failure
        """
        wrapped_kernel = safe_kernel_execution(kernel_func, arg_types, timeout, retry_count)
        self.kernels[name] = wrapped_kernel
        self.arg_types[name] = arg_types
        
    def execute_kernel(
        self,
        name: str,
        *args,
        timeout: Optional[float] = None,
        retry_count: Optional[int] = None
    ) -> Any:
        """
        Execute a registered kernel with safety checks.
        
        Args:
            name: Name of the kernel to execute
            *args: Arguments to pass to the kernel
            timeout: Optional override for execution timeout
            retry_count: Optional override for retry count
            
        Returns:
            Result of kernel execution
            
        Raises:
            KernelError: If kernel execution fails
        """
        if name not in self.kernels:
            raise KernelError(f"Kernel '{name}' not registered")
            
        kernel = self.kernels[name]
        arg_types = self.arg_types[name]
        
        # Validate arguments
        if len(args) != len(arg_types):
            raise KernelError(f"Expected {len(arg_types)} arguments for kernel '{name}', got {len(args)}")
            
        # Convert arguments to the correct types
        converted_args = []
        for arg, (_, expected_type) in zip(args, arg_types):
            try:
                if isinstance(arg, expected_type):
                    # If argument is already of the correct type, use it as is
                    converted_args.append(arg)
                elif expected_type == ctypes.POINTER(ctypes.c_float):
                    if isinstance(arg, np.ndarray):
                        converted_args.append(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                    elif isinstance(arg, ctypes.POINTER(ctypes.c_float)):
                        converted_args.append(arg)
                    else:
                        raise TypeError(f"Cannot convert {type(arg)} to pointer to c_float")
                elif expected_type == ctypes.c_int:
                    converted_args.append(ctypes.c_int(int(arg)))
                elif expected_type == ctypes.c_float:
                    converted_args.append(ctypes.c_float(float(arg)))
                else:
                    converted_args.append(arg)
            except Exception as e:
                raise KernelError(f"Error converting argument: {str(e)}")
            
        # Execute kernel with safety checks
        try:
            return kernel(*converted_args)
        except Exception as e:
            raise KernelError(f"Error executing kernel '{name}': {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a wrapper instance
    wrapper = KernelWrapper()
    
    # Example kernel function with proper type hints
    def example_kernel(a: FloatPtr, b: ctypes.c_int) -> None: # type: ignore
        # Simulate kernel execution
        time.sleep(0.1)
        return None
    
    # Register the kernel
    wrapper.register_kernel(
        "example",
        example_kernel,
        [
            ("input", ctypes.c_float),
            ("size", ctypes.c_int)
        ],
        timeout=1.0,
        retry_count=2
    )
    
    # Test the kernel
    try:
        # Create test data
        input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        size = 3
        
        # Execute kernel
        wrapper.execute_kernel("example", input_data, size)
        logger.info("Kernel execution successful!")
        
    except KernelError as e:
        logger.error(f"Kernel execution failed: {str(e)}") 