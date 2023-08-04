from linear_rnn.triton_scan import complex_scan
import torch

def naive_forward(  v_real,
                    v_imag,             
                    f_real,
                    f_imag                   
                    ):    
    B, L, C = v_real.shape
    h_real = v_real.new_zeros(B, C)
    h_imag = v_real.new_zeros(B, C)

    output_real = v_real.new_zeros(B, L, C)
    output_imag = v_real.new_zeros(B, L, C)
    
    for i in range(L):
        input_real = v_real[:, i, :]
        input_imag = v_imag[:, i, :]
        decay_real = f_real[:, i, :]
        decay_imag = f_imag[:, i, :]   

        h_real_new =  (h_real * decay_real - h_imag * decay_imag) +  input_real        

        h_imag_new = (h_real * decay_imag + h_imag * decay_real) + input_imag
                
        output_real[:, i, :] = h_real_new
        output_imag[:, i, :] = h_imag_new 

        h_real = h_real_new        
        h_imag = h_imag_new
    
    return output_real, output_imag
    

def check_gradient():
    B = 4
    L = 1024
    C = 512
    v_real, v_imag, f_real, f_imag = torch.rand(B, L, C * 4).chunk(4, dim=-1)

    v_real = v_real.cuda().requires_grad_(True).contiguous()
    v_imag = v_imag.cuda().requires_grad_(True).contiguous()
    f_real = f_real.cuda().requires_grad_(True).contiguous()
    f_imag = f_imag.cuda().requires_grad_(True).contiguous()

    grad_output_real = torch.rand(B, L, C).cuda()
    grad_output_image = torch.rand(B, L, C).cuda()
    
    output1, output2 = naive_forward(v_real, v_imag, f_real, f_imag)
    
    (output1 * grad_output_real + output2 * grad_output_image).sum().backward()

    v_real_grad_clone = v_real.grad.clone()
    v_imag_grad_clone = v_imag.grad.clone()
    f_real_grad_clone = f_real.grad.clone()
    f_imag_grad_clone = f_imag.grad.clone()
    
    v_real.grad.zero_()
    v_imag.grad.zero_()
    f_real.grad.zero_()
    f_imag.grad.zero_()

    output3, output4 = complex_scan(v_real, v_imag, f_real, f_imag)

    (output3 * grad_output_real + output4 * grad_output_image).sum().backward()
    
    diff0 = (output3 - output1).abs().max()
    diff00 = (output4 - output2).abs().max()

    diff1 = (v_real.grad - v_real_grad_clone).abs().max()
    diff2 = (v_imag.grad - v_imag_grad_clone).abs().max()
    diff3 = (f_real.grad - f_real_grad_clone).abs().max()
    diff4 = (f_imag.grad -  f_imag_grad_clone).abs().max()
    print(diff0, diff00, diff1, diff2, diff3, diff4)
    breakpoint()


