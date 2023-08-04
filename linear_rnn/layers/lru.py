import numpy as np
import torch
import torch.nn as nn
from linear_rnn.scan_triton import complex_scan



class LRULayer(nn.Module):

    def __init__(
            self,
            d_model,
            dropout=0.2
        ):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(self.d_model, self.d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(2*self.d_model)
        self.out_proj = nn.Linear(2*self.d_model, self.d_model)

        nu_log, theta_log, gamma_log = self.initializer()
        self.nu_log = nn.Parameter(nu_log, requires_grad=True)
        self.theta_log = nn.Parameter(theta_log, requires_grad=True)
        self.gamma_log = nn.Parameter(gamma_log, requires_grad=True)

        self.swish =  nn.SiLU()

    def initializer(self):
        #https://arxiv.org/pdf/2303.06349.pdf Sect.3.2.2
        r_min, r_max = 0.9, 0.999
        u1 = np.random.random(self.d_model)
        u2 = np.random.random(self.d_model)
        nu_log = np.log(
            -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        theta_log = np.log(u2 * np.pi * 2)
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        
        return torch.Tensor(nu_log), torch.Tensor(theta_log), torch.Tensor(gamma_log)

    def forward(self, x):
        u = self.in_proj(x)
        v, o  = u.chunk(2,dim=-1)

        nu = torch.exp(-torch.exp(self.nu_log))
        theta = torch.exp(self.theta_log) 
        gamma = torch.exp(self.gamma_log)

        f_real = nu * torch.cos(theta)
        f_imag = nu * torch.sin(theta)
        
        input_real, input_imag = v.chunk(2, dim=-1)
        input_real = gamma[None, None, :] * input_real
        input_imag = gamma[None, None, :] * input_imag        
        
        f_real = f_real[None, None, :].expand_as(input_real)
        f_imag = f_imag[None, None, :].expand_as(input_real)
    
        output_real, output_imag = complex_rnn_scan(
            input_real.contiguous(), input_imag.contiguous(),
            f_real.contiguous(), f_imag.contiguous()
        )

        return self.out_proj( 
            self.layer_norm(
                self.dropout(
                torch.cat([output_real, output_imag], dim=-1) * self.swish(o)
                )
            )
        )
    
    
