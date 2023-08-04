import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch.autograd import Function

# input: (B, L, D)
@triton.jit
def fwd_sequential_scan(
    v,
    f1,
    hidden,
    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):
    
    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M        
    h1 = tl.zeros([BLOCK_M,], dtype=tl.float32)

    for _ in range(L):        
        x0 = tl.load(v + ptr).to(tl.float32)                
        decay1 = tl.load(f1 + ptr).to(tl.float32)
        h1 = (h1 - x0) * decay1 + x0
        tl.store(hidden + ptr, h1.to(hidden.dtype.element_ty) )
        ptr += C


@triton.jit
def fwd_sequential_scan_fused(
    v,
    f1,
    hidden,
    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):
    
    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M        
    h1 = tl.zeros([BLOCK_M,], dtype=tl.float32)

    for _ in range(L):        
        x0 = tl.load(v + ptr).to(tl.float32)                
        decay1 = tl.load(f1 + ptr).to(tl.float32)
        decay1 = tl.sigmoid(decay1)
        h1 = (h1 - x0) * decay1 + x0
        tl.store(hidden + ptr, h1.to(hidden.dtype.element_ty) )
        ptr += C


# input: (B, L, D)
@triton.jit
def bwd_sequential_scan(
    grad_output,
    
    v,
    f,

    h,

    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):
    
    
    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)    

    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + (L-1) * C + offset_n * BLOCK_M

    grad_h = tl.zeros([BLOCK_M,], dtype=tl.float32)

    for time_step in range(L-1, -1, -1):        

        grad = tl.load(grad_output + ptr).to(tl.float32)                    

        grad_h += grad

        decay = tl.load(f + ptr).to(tl.float32)
        input = tl.load(v + ptr).to(tl.float32)

        grad_v = (1 - decay) * grad_h
        tl.store(v + ptr, grad_v.to(v.dtype.element_ty))

        hidden_state = tl.load(h + ptr - C, mask= ptr >= (offset_b * L * C + C), other=0.0).to(tl.float32)

        grad_f = grad_h * (hidden_state - input)  
        
        tl.store(f + ptr, grad_f.to(f.dtype.element_ty))

        grad_h *= decay        



        ptr -= C        

# input: (B, L, D)
@triton.jit
def bwd_sequential_scan_fused(
    grad_output,
    
    v,
    f,

    h,

    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):
    
    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)    

    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + (L-1) * C + offset_n * BLOCK_M

    grad_h = tl.zeros([BLOCK_M,], dtype=tl.float32)

    for time_step in range(L-1, -1, -1):        

        grad = tl.load(grad_output + ptr).to(tl.float32)                    

        grad_h += grad

        decay = tl.load(f + ptr).to(tl.float32)
        decay = tl.sigmoid(decay)
        input = tl.load(v + ptr).to(tl.float32)

        grad_v = (1 - decay) * grad_h
        tl.store(v + ptr, grad_v.to(v.dtype.element_ty))

        hidden_state = tl.load(h + ptr - C, mask= ptr >= (offset_b * L * C + C), other=0.0).to(tl.float32)

        grad_f = grad_h * (hidden_state - input) * decay * (1 - decay)
        
        tl.store(f + ptr, grad_f.to(f.dtype.element_ty))

        grad_h *= decay        


        ptr -= C        


class TritonSequentialScan(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, v, f1):
        B,L,C = v.shape
        num_warps = 8
        assert C % 256 == 0
        v = v.contiguous()
        f1 = f1.contiguous()
        hidden =  torch.zeros_like(v).contiguous()
                                    
        fwd_sequential_scan[(B, int(C/256) )](
            v,
            f1,
            hidden,
            B,
            L,
            C, 
            BLOCK_M=256,
            num_warps=num_warps
        )

        ctx.save_for_backward(v, f1, hidden)    
        return hidden
            
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        v, f1, hidden = ctx.saved_tensors 
        B, L, C = v.shape
        
        num_warps = 8

        bwd_sequential_scan[(B,  int(C/256))](
            grad_output,                 
            v,
            f1,
            hidden,
            B,
            L,
            C, 
            BLOCK_M=256,
            num_warps=num_warps
        )
        return v, f1


class TritonSequentialScanFused(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, v, f1):
        B,L,C = v.shape
        num_warps = 8
        assert C % 256 == 0
        v = v.contiguous()
        f1 = f1.contiguous()
        hidden =  torch.zeros_like(v).contiguous()
                                    
        fwd_sequential_scan_fused[(B, int(C/256) )](
            v,
            f1,
            hidden,
            B,
            L,
            C, 
            BLOCK_M=256,
            num_warps=num_warps
        )

        ctx.save_for_backward(v, f1, hidden)    
        return hidden
            
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        v, f1, hidden = ctx.saved_tensors 
        B, L, C = v.shape
        
        num_warps = 8

        bwd_sequential_scan_fused[(B,  int(C/256))](
            grad_output,                 
            v,
            f1,
            hidden,
            B,
            L,
            C, 
            BLOCK_M=256,
            num_warps=num_warps
        )
        return v, f1


real_scan_tie_input_gate = TritonSequentialScan.apply


real_scan_tie_input_gate_fused = TritonSequentialScanFused.apply




