"""
ARM64 SIMD-optimized implementation of noise scheduling kernel for diffusion models.
"""

import numpy as np
import ctypes
import logging
from typing import Optional, Tuple
from OPENtransformer.core.asm.assembler.builder import build_and_jit

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class NoiseSchedulingKernelASM:
    """
    ARM64 SIMD-optimized implementation of noise scheduling kernel.
    
    This kernel handles the computation of noise schedules for diffusion models,
    supporting both linear and cosine schedules.
    """
    
    def __init__(self):
        """Initialize noise scheduling kernel with JIT-compiled assembly code."""
        try:
            self._noise_scheduling_kernel = self._compile_noise_scheduling()
            self._asm_available = True
            logger.info("Successfully compiled noise scheduling kernel")
        except Exception as e:
            logger.error(f"Failed to compile noise scheduling kernel: {e}")
            print("Falling back to NumPy implementation")
            self._asm_available = False
            
    def _compile_noise_scheduling(self):
        """Compile the noise scheduling kernel."""
        asm_code = """
        .section __DATA,__data
        .align 4
        offset: .float 0.008
        scale: .float 1.008
        pi: .float 3.14159
        half: .float 0.5
        one: .float 1.0
        eps: .float 1e-9
        half_factorial: .float 0.5      // 1/2!
        quarter_factorial: .float 0.041667  // 1/4!
        sixth_factorial: .float 0.001389    // 1/6!
        
        .section __TEXT,__text
        .global _noise_scheduling_asm
        .align 4
        _noise_scheduling_asm:
            // x0: output array pointer (float*)
            // w1: schedule type (0=linear, 1=cosine)
            // s0: beta_start
            // s1: beta_end
            // w2: timesteps
            // w3: current timestep
            
            // Save frame and registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            
            // Check for null pointer
            cbz x0, error
            
            // Check schedule type
            cmp w1, #1
            b.gt error
            
            // Check timesteps
            cbz w2, error
            cmp w2, #0
            b.le error
            
            // Check current timestep
            cmp w3, w2
            b.ge error
            cmp w3, #0
            b.lt error
            
            // Convert timesteps and current timestep to float
            scvtf s2, w2  // timesteps
            scvtf s3, w3  // current timestep
            
            // Check beta range
            fcmp s0, s1
            b.ge error
            
            // Broadcast values to NEON registers
            dup v0.4s, v0.s[0]  // beta_start
            dup v1.4s, v1.s[0]  // beta_end
            dup v2.4s, v2.s[0]  // timesteps
            dup v3.4s, v3.s[0]  // current timestep
            
            // Check schedule type
            cbz w1, linear_schedule
            
        cosine_schedule:
            // Load constants from data section into NEON registers
            adrp x6, offset@PAGE
            add x6, x6, offset@PAGEOFF
            ldr s4, [x6]  // offset
            dup v4.4s, v4.s[0]
            
            adrp x6, scale@PAGE
            add x6, x6, scale@PAGEOFF
            ldr s5, [x6]  // scale
            dup v5.4s, v5.s[0]
            
            adrp x6, pi@PAGE
            add x6, x6, pi@PAGEOFF
            ldr s6, [x6]  // pi
            dup v6.4s, v6.s[0]
            
            adrp x6, half@PAGE
            add x6, x6, half@PAGEOFF
            ldr s7, [x6]  // half
            dup v7.4s, v7.s[0]
            
            // Calculate t/T using NEON
            fdiv v8.4s, v3.4s, v2.4s  // t/T
            
            // Add offset
            fadd v8.4s, v8.4s, v4.4s
            
            // Divide by scale
            fdiv v8.4s, v8.4s, v5.4s
            
            // Multiply by pi/2
            fmul v8.4s, v8.4s, v6.4s
            fmul v8.4s, v8.4s, v7.4s
            
            // Calculate cos using Taylor series approximation with NEON
            // cos(x) ≈ 1 - x^2/2! + x^4/4! - x^6/6!
            fmul v9.4s, v8.4s, v8.4s  // x^2
            
            // Load factorial constants
            adrp x6, half_factorial@PAGE
            add x6, x6, half_factorial@PAGEOFF
            ldr s10, [x6]
            dup v10.4s, v10.s[0]
            
            adrp x6, quarter_factorial@PAGE
            add x6, x6, quarter_factorial@PAGEOFF
            ldr s11, [x6]
            dup v11.4s, v11.s[0]
            
            adrp x6, sixth_factorial@PAGE
            add x6, x6, sixth_factorial@PAGEOFF
            ldr s12, [x6]
            dup v12.4s, v12.s[0]
            
            // First term: 1
            adrp x6, one@PAGE
            add x6, x6, one@PAGEOFF
            ldr s13, [x6]
            dup v13.4s, v13.s[0]
            mov v14.16b, v13.16b  // result = 1
            
            // Second term: -x^2/2!
            fmul v15.4s, v9.4s, v10.4s  // x^2/2!
            fsub v14.4s, v14.4s, v15.4s  // result -= x^2/2!
            
            // Third term: x^4/4!
            fmul v15.4s, v9.4s, v9.4s  // x^4
            fmul v15.4s, v15.4s, v11.4s  // x^4/4!
            fadd v14.4s, v14.4s, v15.4s  // result += x^4/4!
            
            // Fourth term: -x^6/6!
            fmul v15.4s, v15.4s, v9.4s  // x^6
            fmul v15.4s, v15.4s, v12.4s  // x^6/6!
            fsub v14.4s, v14.4s, v15.4s  // result -= x^6/6!
            
            // Square the result
            fmul v14.4s, v14.4s, v14.4s
            
            // Calculate 1 - cos^2
            fsub v14.4s, v13.4s, v14.4s
            
            // Store result
            str s14, [x0]
            b done
            
        linear_schedule:
            // Calculate linear schedule using NEON
            fdiv v8.4s, v3.4s, v2.4s  // t/T
            fsub v9.4s, v1.4s, v0.4s  // beta_end - beta_start
            fmul v8.4s, v8.4s, v9.4s  // (t/T) * (beta_end - beta_start)
            fadd v8.4s, v8.4s, v0.4s  // beta_start + (t/T) * (beta_end - beta_start)
            
            // Store result
            str s8, [x0]
            
        done:
            // Return success
            mov x0, #1
            
            // Restore registers
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            ret
            
        error:
            // Return error
            mov x0, #0
            
            // Restore registers
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            ret
        """
        return build_and_jit(asm_code, "_noise_scheduling_asm")
        
    def get_beta_schedule(self,
                         schedule_type: str,
                         beta_start: float,
                         beta_end: float,
                         timesteps: int) -> np.ndarray:
        """
        Get the beta schedule for the diffusion process.
        
        Args:
            schedule_type: The type of beta schedule ('linear' or 'cosine')
            beta_start: Starting value for beta schedule
            beta_end: Ending value for beta schedule
            timesteps: Number of timesteps
            
        Returns:
            Array of beta values for each timestep
            
        Raises:
            ValueError: If schedule_type is invalid
            RuntimeError: If timesteps <= 0 or beta_start >= beta_end
        """
        # Input validation
        if schedule_type not in ['linear', 'cosine']:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")
            
        if timesteps <= 0:
            raise RuntimeError("Number of timesteps must be positive")
            
        if beta_start >= beta_end:
            raise RuntimeError("beta_start must be less than beta_end")
            
        if not self._asm_available:
            logger.info("ASM not available, using NumPy implementation")
            return self._numpy_get_beta_schedule(schedule_type, beta_start, beta_end, timesteps)
            
        # Convert schedule type to integer
        schedule_type_int = 0 if schedule_type == 'linear' else 1
        
        # Prepare output array
        betas = np.empty(timesteps, dtype=np.float32)
        
        try:
            logger.debug(f"Calling assembly kernel with parameters:")
            logger.debug(f"  schedule_type: {schedule_type} ({schedule_type_int})")
            logger.debug(f"  beta_start: {beta_start}")
            logger.debug(f"  beta_end: {beta_end}")
            logger.debug(f"  timesteps: {timesteps}")
            
            # Call assembly kernel for each timestep
            for t in range(timesteps):
                # Create a temporary buffer for the output
                output_buffer = np.zeros(1, dtype=np.float32)
                
                # Call the kernel with the temporary buffer
                result = self._noise_scheduling_kernel(
                    output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int(schedule_type_int),
                    ctypes.c_float(beta_start),
                    ctypes.c_float(beta_end),
                    ctypes.c_int(timesteps),
                    ctypes.c_int(t)
                )
                
                logger.debug(f"  timestep {t}: result = {result}, output = {output_buffer[0]}")
                
                if result == 0:
                    logger.error(f"Kernel returned error at timestep {t}")
                    raise RuntimeError("Noise scheduling kernel returned error")
                    
                # Copy the result to the output array
                betas[t] = output_buffer[0]
                
        except Exception as e:
            logger.error(f"Assembly kernel failed: {e}")
            logger.info("Falling back to NumPy implementation")
            return self._numpy_get_beta_schedule(schedule_type, beta_start, beta_end, timesteps)
            
        return betas
        
    def _numpy_get_beta_schedule(self,
                                schedule_type: str,
                                beta_start: float,
                                beta_end: float,
                                timesteps: int) -> np.ndarray:
        """NumPy implementation of beta schedule computation."""
        if schedule_type == 'linear':
            return np.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps)
            alphas_cumprod = np.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule_type}") 