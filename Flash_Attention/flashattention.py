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


@triton.jit
def attn_backward_preprocess(
    O_ptr, dLdO_ptr, Delta_ptr,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_dLdO_B, stride_dLdO_H, stride_dLdO_N, stride_dLdO_Dh,
    stride_Delta_B, stride_Delta_H, stride_Delta_N,
    N, Dh: tl.constexpr,
    PRE_BLOCK_SIZE_ROW: tl.constexpr,
):
    """
    This function calculates Delta values that will be needed in the backward pass.
    Delta represents the element-wise product of the output and its gradient, summed
    across the head dimension. We compute it once here to avoid redundant calculations
    in the later backward kernels.
    """
    index_BH = tl.program_id(1) # B * H number of pids
    row = tl.program_id(0) # N / BLOCK_SIZE_ROW number of pids

    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, Dh)
    mask = row_offsets < N

    # Load PRE_BLOCK_SIZE_ROW rows of O
    O_ptr += index_BH * stride_O_H # moves O_ptr to the correct batch & head for this pid.
    O_offsets = row_offsets[:, None] * stride_O_N + col_offsets[None, :] * stride_O_Dh
    O = tl.load(O_ptr + O_offsets, mask = mask[:, None], other=0.) # shape (PRE_BLOCK_SIZE_ROW, D)

    # Load PRE_BLOCK_SIZE_ROW rows of dLdO
    dLdO_ptr += index_BH * stride_dLdO_H
    dLdO_offsets = row_offsets[:, None] * stride_dLdO_N + col_offsets[None, :] * stride_dLdO_Dh
    dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask = mask[:, None], other=0.) # shape (PRE_BLOCK_SIZE_ROW, D) 

    # Delta is the dot product of O and dLdO along Dh, giving us a single scalar Delta_i per token in N
    # it will be useful in later parts of the backward pass
    Delta = tl.sum(dLdO.to(tl.float32) * O.to(tl.float32), axis=1) # shape (PRE_BLOCK_SIZE_ROW)
    Delta_ptr += index_BH * stride_Delta_H
    tl.store(Delta_ptr + row_offsets, Delta, mask = mask)


@triton.jit
def _attn_backward_KV(
    K, V, dLdK, dLdV,               # shape (BLOCK_SIZE_COL, D)
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr, 
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,   # no more _1 because this sub-kernel is the _1
    BLOCK_SIZE_COL: tl.constexpr, 
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    """
    This function computes gradients for Key and Value matrices. It works by taking
    a fixed chunk of K and V tokens, then iterating through different chunks of Q
    tokens to calculate how each K,V pair should be updated based on the attention
    computation. Think of it as fixing the columns of the attention matrix and
    sweeping through the rows to accumulate gradients.
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    # we transpose Q while loading it rather than in a separate kernel
    Q_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh

    for block_idx in range(num_steps):
        # Load data in an order that helps the compiler optimize memory access patterns
        # by grouping loads together before computations
        mask_N = offsets_ROW < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.) # shape (Dh, BLOCK_SIZE_ROW)
        LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_N, other=0.) # shape (BLOCK_SIZE_ROW)
        dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask_N[:, None], other=0.) # shape (BLOCK_SIZE_ROW, Dh)
        Delta = tl.load(Delta_ptr + offsets_ROW, mask=mask_N, other=0.) # shape (BLOCK_SIZE_ROW)

        # Recreate the attention scores and probabilities since storing them would use too much memory
        # We compute the transpose versions directly since that's what we need
        S_T = tl.dot(K, Q_T) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        # Convert scores to probabilities using the logsumexp values from forward pass
        P_T = tl.exp2(S_T - LSE[None, :]) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)

        if MASK: # Apply causal masking for autoregressive models
            # Create lower triangular mask (appears upper triangular due to transpose)
            mask = (offsets_COL[:, None] <= offsets_ROW[None, :]) # (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
            P_T = tl.where(mask, P_T, 0.)

        # Compute gradient for Value matrix
        dLdV = tl.dot(P_T, dLdO, acc=dLdV) # shape (BLOCK_SIZE_COL, Dh)

        # Compute gradients for Key matrix by working backwards through the attention computation
        dLdP_T = tl.dot(V, tl.trans(dLdO)) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdS_T = (P_T * (dLdP_T - Delta[None, :]) * ln2) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdK = tl.dot(dLdS_T, tl.trans(Q_T), acc=dLdK) # shape (BLOCK_SIZE_COL, D)

        # Move to next chunk of Q tokens
        offsets_ROW += BLOCK_SIZE_ROW
        Q_ptr += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr += BLOCK_SIZE_ROW * stride_N
    
    return dLdK, dLdV


@triton.jit
def _attn_backward_Q(
    dLdQ, Q, dLdO, LSE, 
    K_ptr, V_ptr, Delta_ptr,
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr, 
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    """
    This function computes gradients for the Query matrix. It takes a fixed chunk
    of Q tokens and iterates through different chunks of K and V tokens to calculate
    how each Q token should be updated. This is similar to the forward pass pattern
    where we fix Q and sweep through K,V pairs.
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    # we transpose V while loading it
    K_and_V_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N

    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW<N, other=0.) # shape (BLOCK_SIE_ROW)

    for block_idx in range(num_steps):
        K_T = tl.load(K_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[None, :], other=0.) 
            # shape (Dh, BLOCK_SIZE_COL)
        V_T = tl.load(V_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[None, :], other=0.) 
            # shape (Dh, BLOCK_SIZE_COL)

        # Recreate attention scores and probabilities for this Q,K,V block
        S = tl.dot(Q, K_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        P = tl.exp2(S - LSE) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        if MASK: # Apply causal masking
            mask = (offsets_ROW[:, None] >= offsets_COL[None, :]) # (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
            P = tl.where(mask, P, 0.) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        # Work backwards through attention computation to get Q gradients
        dLdP = tl.dot(dLdO, V_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        dLdS = (P * (dLdP - Delta[:, None]) * ln2) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        dLdQ += tl.dot(dLdS, tl.trans(K_T)) # shape (BLOCK_SIZE_ROW, Dh)
        
        # Move to next chunk of K,V tokens
        offsets_COL += BLOCK_SIZE_COL
        K_ptr += BLOCK_SIZE_COL * stride_N
        V_ptr += BLOCK_SIZE_COL * stride_N
    
    return dLdQ


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                        num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE_MICRO in [16]#, 32, 64]
        for BLOCK_SIZE_MACRO in [32]#, 64, 128]
        for num_stages in [3]#, 5, 7]
        for num_warps in [4]#, 8, 16]
        if BLOCK_SIZE_MACRO > BLOCK_SIZE_MICRO # could do >= but i wanna get mileage out of the loop code we wrote
    ],
    key=["Dh"],
)
@triton.jit
def attn_backward(
    Q_ptr, K_ptr, V_ptr, 
    dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr, 
    BLOCK_SIZE_MICRO: tl.constexpr,  #
    BLOCK_SIZE_MACRO: tl.constexpr,  #
):
    # Mathematical constants for efficient computation
    ln2: tl.constexpr = 0.6931471824645996  # natural logarithm of 2
    rln2: tl.constexpr = 1.4426950408889634  # reciprocal of ln(2)

    # Navigate to the correct position in our 4D tensors for this parallel execution
    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head = idx_batch_head % H 
    batch_head_jump = idx_batch * stride_B + idx_head * stride_H
    Q_ptr += batch_head_jump
    K_ptr += batch_head_jump
    V_ptr += batch_head_jump
    dLdO_ptr += batch_head_jump
    dLdQ_ptr += batch_head_jump
    dLdK_ptr += batch_head_jump
    dLdV_ptr += batch_head_jump

    # Navigate to correct position in 3D tensors (batch, head, sequence)
    batch_head_jump = idx_batch_head * N
    LSE_ptr += batch_head_jump
    Delta_ptr += batch_head_jump

    # Ensure our block sizes are compatible
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    ### STAGE 1: Compute gradients for Key and Value matrices
    # We use the opposite strategy from forward pass - fix K,V and iterate through Q

    # Set up block dimensions for this stage
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO

    # Start with diagonal blocks that need special causal masking
    pid = tl.program_id(0)
    start_COL = pid * BLOCK_SIZE_COL_1
    start_ROW = start_COL
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1
    
    # Load the K and V blocks that this parallel execution will work on
    offsets_COL_1 = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_Dh = tl.arange(0, Dh)
    KV_offsets = offsets_COL_1[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask = (offsets_COL_1[:, None] < N)
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.) # shape (BLOCK_SIZE_COL_1, Dh)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.) # shape (BLOCK_SIZE_COL_1, Dh)

    # Apply scaling factors once here instead of repeatedly in the loop
    K *= scale * rln2

    # Initialize gradient accumulators
    dLdK = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)

    # Handle diagonal blocks with causal masking
    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    # Handle off-diagonal blocks that don't need causal masking
    start_ROW += BLOCK_SIZE_COL_1
    N_adj = tl.cdiv(N, BLOCK_SIZE_COL_1) * BLOCK_SIZE_COL_1
    num_steps = (N_adj - start_ROW) // BLOCK_SIZE_ROW_1

    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )

    # Apply final scaling and store results
    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK, mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV, mask=KV_mask)

    ### STAGE 2: Compute gradients for Query matrix
    # Similar to forward pass - fix Q and iterate through K,V

    # Set up block dimensions for this stage
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO

    # Start with diagonal blocks
    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2

    # Load the Q block and related data for this parallel execution
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    QO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW = offsets_ROW < N
    Q = tl.load(Q_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.) # shape (BLOCK_SIZE_ROW_2, Dh) 
    Q *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.) # shape (BLOCK_SIZE_ROW_2, Dh) 
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_ROW, other=0.)[:, None] # shape (BLOCK_SIZE_ROW_2, 1) 

    # Initialize gradient accumulator
    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)

    # Handle diagonal blocks with causal masking
    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, LSE, 
        K_ptr, V_ptr, Delta_ptr, 
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    # Handle off-diagonal blocks without causal masking
    end_COL = start_COL
    start_COL = 0
    num_steps = end_COL // BLOCK_SIZE_COL_2
    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, LSE, 
        K_ptr, V_ptr, Delta_ptr, 
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )
    
    # Apply final scaling and store results
    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ, mask=mask_ROW[:, None])


class _flashattention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale): 
        assert q.shape == k.shape == v.shape
        assert q.shape[-1] <= 128, \
            f'flash attention only supports head dimension of 128 less but got {q.shape[-1]}'
        B, H, N, Dh = q.shape
        assert q.device == k.device and q.device == v.device
        assert q.dtype == k.dtype == v.dtype == torch.float32

        # Create output tensor and log-sum-exp storage
        O = torch.empty_like(q)
        LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        # Set up parallel execution grid - prioritize sequence dimension for memory locality
        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]),
            B * H,
        )

        attn_fwd[grid](
            q, k, v, O, LSE, 
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, N, Dh,
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.N, ctx.Dh = B, H, N, Dh
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        grid = ctx.grid
        scale = ctx.scale
        B, H, N, Dh = ctx.B, ctx.H, ctx.N, ctx.Dh

        # Create storage for gradients
        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)

        dLdO = dLdO.contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride() == dLdO.stride()

        # Precompute Delta values needed for backward pass
        Delta = torch.empty_like(LSE)
        pre_grid = lambda meta: (triton.cdiv(N, meta["PRE_BLOCK_SIZE_ROW"]), B * H)
        attn_backward_preprocess[pre_grid](
            O, dLdO, Delta,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dLdO.stride(0), dLdO.stride(1), dLdO.stride(2), dLdO.stride(3),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            N, Dh,
        )

        # Run main backward pass
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_MACRO"]), B * H) 
        attn_backward[grid](
            q, k, v,
            dLdO, dLdq, dLdk, dLdv,
            LSE, Delta,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, N, Dh,
        )

        return dLdq, dLdk, dLdv, None

triton_attention = _flashattention.apply
    

######### Step 1 #########
def test_flashattention_kernel(B, H, N, Dh, device=DEVICE, atol=5e-3):
    # Create random test data
    q = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device, requires_grad=True)
    sm_scale = 1/math.sqrt(Dh)
    
    # Test forward pass
    tri_out = triton_attention(q, k, v, sm_scale)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    """
    # Optional: Visual analysis of errors
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    actual = tri_out.detach().cpu().numpy()
    expected = ref_out.detach().cpu().numpy()
    abs_diff = np.abs(expected - actual)
    abs_fail_mask = (abs_diff > 1e-2).astype(np.int32)
    plt.figure(figsize=(8, 6))
    plt.imshow(abs_fail_mask[0][0], cmap="hot", aspect="auto")
    plt.xlabel("Model/Head Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar()
    plt.savefig('./out_heatmap.png')
    plt.close()
    """
    
    # Verify forward pass accuracy
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0) 
    print("passed fwd")

    # Test backward pass
    dLdout = 0.1 * torch.randn_like(q)
    tri_out.backward(dLdout, retain_graph=True)
    dLdq_tri, dLdk_tri, dLdv_tri = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None
    
    # Reference backward pass
    ref_out.backward(dLdout, retain_graph=True)
    dLdq_ref, dLdk_ref, dLdv_ref = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None

    """
    # Optional: Visual analysis of gradient errors
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    # Analysis code for dLdq, dLdk, dLdv...
    """

    # Verify backward pass accuracy
    torch.testing.assert_close(dLdq_tri, dLdq_ref, atol=atol, rtol=0)
    torch.testing.assert_close(dLdk_tri, dLdk_ref, atol=atol, rtol=0)
    torch.testing.assert_close(dLdv_tri, dLdv_ref, atol=atol, rtol=0)
    print("Passed bwd")
    

# Performance benchmarking configuration
configs = []
for mode in ["fwd", "bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[512 * i for i in range(1, 17)],
            line_arg="provider",
            line_vals=["torch", 'this_tutorial'],
            line_names=[
                "torch.nn.functional.scaled_dot_product_attention", 
                "This tutorial's implementation"
                ],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"attention-performance-{mode}",
            args={"mode": mode},
        ))

@triton.testing.perf_report(configs)
def bench_flash_attention(SEQ_LEN, mode, provider, device=DEVICE):
    """
    Benchmark function that measures performance in TFLOPS (trillion floating point operations per second)
    for both forward and backward passes of attention computation.
    """
    assert mode in ["fwd", "bwd"]
    dtype = torch.float32
    BATCH, N_HEADS = 32, 4
    HEAD_DIM = 128
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    
    if provider == 'torch':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    if provider == 'this_tutorial':
        fn = lambda: triton_attention(q, k, v, sm_scale)
    if mode == "bwd":
        O = fn()
        dLdO = torch.randn_like(O)
        fn = lambda: O.backward(dLdO, retain_graph=True)
        
    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul * 0.5
    if mode == "bwd":
        total_flops *= 2.5  # Account for backward pass complexity
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
     # Run comprehensive tests
    test_flashattention_kernel(1, 1, 128, 32) # without block masking
    test_flashattention_kernel(1, 1, 128, 64) # without block masking
    test_flashattention_kernel(1, 1, 128, 128) # without block masking
    test_flashattention_kernel(32, 8, 69, 128) # with block masking

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        bench_flash_attention.run(save_path='.', print_data=True)