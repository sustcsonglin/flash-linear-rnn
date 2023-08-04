from linear_rnn.scan_triton import real_scan_tie_input_gate_fused
import torch

def naive_forward_fused(  v,                    
                          f                                       
                    ):    
    B, L, C = v.shape
    h = v.new_zeros(B, C)

    output = v.new_zeros(B, L, C)
    
    for i in range(L):
        input = v[:, i, :]
        decay = f[:, i, :].sigmoid()
        h = (h - input) * decay + input
        output[:, i, :] = h

    return output
    

def check_gradient():
    B = 4
    L = 1024
    C = 512
    v,f = torch.rand(B, L, C * 2).chunk(2, dim=-1)

    v = v.cuda().requires_grad_(True).contiguous()
    f = f.cuda().requires_grad_(True).contiguous()

    grad_output = torch.rand(B, L, C).cuda()
    
    output = naive_forward_fused(v, f)
    
    output.backward(grad_output)

    v_grad_clone =  v.grad.clone()
    f_grad_clone = f.grad.clone()
    
    v.grad.zero_()
    f.grad.zero_()

    output2 = real_scan_tie_input_gate_fused(v, f)
    output2.backward(grad_output)
    
    diff0 = (output2 - output).abs().max()

    diff1 = (v.grad - v_grad_clone).abs().max()
    diff2 = (f.grad - f_grad_clone).abs().max()
    print(diff0, diff1, diff2)
    breakpoint()



