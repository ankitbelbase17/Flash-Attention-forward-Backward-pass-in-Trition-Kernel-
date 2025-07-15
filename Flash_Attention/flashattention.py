"""
This is a flash-attention implementation that only works with causal masks (where future tokens can't see past tokens).
It doesn't support other types of masks or no masks at all.

The forward pass follows the ideas from these two research papers:
https://arxiv.org/abs/2205.14135
https://arxiv.org/abs/2307.08691

The backward pass uses the faster triton implementation instead of the slower paper version:
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py

 Learnings from this code:
- How to call smaller functions from within kernels
- Why we use tl.exp2() instead of tl.exp() (it's faster)
- How Flash attention splits up the work across multiple processors
- How to use tl.static_assert() for compile-time checks
- How to organize work across multiple dimensions and why order matters
- When to calculate certain values in a separate kernel beforehand
- Using approximate values instead of exact calculations for speed

What this implementation is missing:
- Support for data types other than 32-bit floats and mixed precision
- Dropout functionality
- Probably other features I'm forgetting

Note: The performance tuning is set up but disabled (lists only have one option each).
If you want better performance, you could turn on the auto-tuning feature.
"""

import torch
import triton
import triton.language as tl
import math

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

#import os
#os.environ["TRITON_INTERPRET"] = "1"

@triton.jit
def _attn_fwd_inner(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    softmax_scale,
    stride_K_N, stride_V_N,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    N: tl.constexpr, Dh: tl.constexpr,
):
    """
    This function does the main attention computation for a block of queries.
    
    Think of it like this: each processor handles one horizontal strip of the attention matrix.
    The arrows show how each processor moves through the data:
    
    For normal attention:
                K & V tokens
                ------------>
                ------------>
    Q tokens    ------------>
                ------------>
                ------------>
    
    But with causal masking (where future tokens can't see past tokens), it looks like:
                K & V tokens
                >
                --->
    Q tokens    ------>
                --------->
                ------------>
    
    We handle the diagonal blocks separately because they need special masking:
                K & V tokens
                x
                   x
    Q tokens            x
                         x
                            x
    
    And the first call handles everything below the diagonal:
                K & V tokens
                
                -->
    Q tokens    ----->
                -------->
                ----------->
    """
    if DIAGONAL:
        # This is for blocks that sit on the diagonal where we transition from allowed to masked tokens
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        # Tell the compiler that lo is a multiple of BLOCK_SIZE_QO so it can optimize better
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
    else: 
        # This handles blocks below the diagonal where all tokens are allowed
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    # Go through blocks of K and V tokens and update our output as we go
    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        # Tell the compiler this is a multiple of BLOCK_SIZE_KV for better optimization
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)

        # Calculate (Q × K^T) / sqrt(head_dimension)
        mask_KV_N = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.) # shape (Dh, BLOCK_SIZE_KV)
        # The mask sets non-existent tokens (beyond sequence length) to zero
        S = tl.dot(Q, K_T) * softmax_scale # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        # Masked tokens create columns and rows of zeros at the edges of S

        if DIAGONAL: # If we're working on a block that contains the diagonal
            # Causal mask is True for positions where query can see the key (lower triangle + diagonal)
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            # Set upper triangle values to very negative number (will become 0 after softmax)
            S += tl.where(causal_mask, 0, -1.0e6) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        
        # Find the maximum value in each row for this block and compare with previous blocks
        M_new = tl.maximum(M, tl.max(S, axis=1)) # shape is (BLOCK_SIZE_QO)
        # Masked token rows will have max value of 0 since they only contain 0s and -inf
        
        # Subtract the max for numerical stability (standard softmax trick)
        S -= M_new[:, None] # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)

        # Calculate exponential of each score (this will be the numerator of softmax)
        P = tl.exp2(S) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        # We use base 2 instead of base e because it's faster and softmax result is the same
        # For masked tokens, this gives us 2^0 = 1 for valid positions
        
        # Sum the attention scores for each query (row)
        L_new = tl.sum(P, axis=1) # shape (BLOCK_SIZE_QO)
        
        # This correction factor adjusts our previous sum based on the new maximum
        alpha = tl.exp2(M - M_new) # shape (BLOCK_SIZE_Q)
        
        # Update our running sum with the correction factor
        L = L * alpha + L_new # shape (BLOCK_SIZE_QO)

        # Load the V values and update our output
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.) # shape (BLOCK_SIZE_KV, Dh)
        # Adjust previous output values based on the new maximum
        O = O * alpha[:, None] # shape (BLOCK_SIZE_QO, Dh)
        # Add the contribution from this block to our output
        O = tl.dot(P, V, acc=O) # shape (BLOCK_SIZE_QO, Dh)
        # Note: We're doing this before dividing by the softmax denominator, but that's okay
        # because matrix multiplication and division are associative at this level
        
        # Update our running maximum for the next iteration
        M = M_new

        # Move pointers to the next block
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M # We need these three values for the backward pass later


@triton.autotune( # This decorator finds the best settings automatically
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [16]#, 32, 64, 128]
        for BLOCK_SIZE_KV in [16]#, 32, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["Dh"],
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr,  V_ptr,               # each shape (B, H, N, Dh)
    O_ptr,                              # shape (B, H, N, Dh). where we store the final output
    LSE_ptr,                            # shape (B, H, N). stores max values first, then log-sum-exp values
    softmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, # batch size can vary at runtime
    # These are fixed at compile time
    H: tl.constexpr, N: tl.constexpr, 
    Dh: tl.constexpr, # head dimension, should be a power of 2
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
):
    # To use tl.exp2 instead of tl.exp (which is faster), we need to adjust our scaling factor
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    """
    Here's why this works mathematically:
    We want to show that e^x = 2^(x * rln2)
    
    Start with: e^x = (2^(log_2(e)))^x  because any number a = 2^(log_2(a))
    Using power rules: (2^(log_2(e)))^x = 2^(x * log_2(e))
    Since log_2(e) = 1/log_e(2), we get: e^x = 2^(x * 1/log_e(2)) 
    Therefore: e^x = 2^(x * rln2)
    
    We'll need to remember this change when computing gradients in the backward pass.
    """
    
    # This check happens at compile time, not runtime
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
    # Not sure why this is needed, but it doesn't hurt

    # Figure out which block of queries this processor should handle
    block_index_QO = tl.program_id(0)
    # Figure out which batch and head this processor should handle
    index_BH = tl.program_id(1)
    # Extract the batch index
    index_B = index_BH // H
    # Extract the head index within the batch
    index_H = index_BH % H

    # Move pointers to the right batch and head
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    # Create offset arrays for indexing
    # For sequence dimension, we split by processor, but for head dimension we keep everything
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)
    
    # Create specific offset patterns for each tensor
    Q_offsets = (offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh)
    # shape (BLOCK_SIZE_QO, Dh)
    
    # We transpose K while loading it (instead of using a separate transpose kernel)
    K_T_offsets = (offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N)
    # shape (Dh, BLOCK_SIZE_KV)
    
    V_offsets = (offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh)
    # shape (BLOCK_SIZE_KV, Dh)

    # Load the block of Q that this processor will use (stays in fast memory throughout)
    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.) # shape (BLOCK_SIZE_QO, Dh)
    # Mask sets tokens beyond sequence length to zero

    # Set up storage for intermediate results
    # Running maximum for each query (starts very negative so any real value will be larger)
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
    # Running sum for each query (starts at 1 because we'll use exponentials and e^0=1)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)
    # Output accumulator for this block of queries
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)

    # Process all blocks below the diagonal (where mask is all 1s)
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False, # Not processing diagonal blocks yet
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )

    # Process the diagonal block (where we need causal masking)
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True, # Now processing diagonal blocks with special masking
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )
    
    # Finally divide by the softmax denominator
    O = O / L[:, None] # shapes (BLOCK_SIZE_QO, Dh) / (BLOCK_SIZE_QO, 1) = (BLOCK_SIZE_QO, Dh)
    # We can do this division after the matrix multiplication because at this granular level,
    # the operations are associative (we're really doing individual dot products)

    # Calculate log-sum-exp for the backward pass
    # Instead of storing max and sum separately, we combine them using exponential math
    LSE = M + tl.math.log2(L) # shape (BLOCK_SIZE_QO)
    # This works because softmax(x_i) = exp(x_i - m_i) / l_i 
    #                                 = exp(x_i - m_i) / exp(log(l_i)) 
    #                                 = exp(x_i - m_i - log(l_i))

    # Save results back to main memory
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask) # shape (BLOCK_SIZE_QO)
    # Mask prevents saving values for tokens beyond sequence length
    
    O_offsets = (offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh)
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None]) # shape (BLOCK_SIZE_Q, Dh)
    # Mask prevents saving values for tokens beyond sequence length


@triton.autotune(
    [
        triton.Config({"PRE_BLOCK_SIZE_ROW": PRE_BLOCK_SIZE_ROW},
                        num_stages=num_stages, num_warps=num_warps,)
        for PRE_BLOCK_SIZE_ROW in [32]#, 64, 128, 256]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
    ],
    key=["Dh"],
)