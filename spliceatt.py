import torch
import os
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

model_stub = "neuralmagic/Meta-Llama-3.1-8B-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_stub)

# Load the model
model = SparseAutoModelForCausalLM.from_pretrained(model_stub, torch_dtype=torch.float16)

weights = model.state_dict()

att = {}



window_size = (16, 16) 
for name, param in weights.items():
    if any(layer in name for layer in ['.0', '15', '31']) and (
        name.endswith('self_attn.q_proj.weight') or 
        name.endswith('self_attn.k_proj.weight') or 
        name.endswith('self_attn.v_proj.weight') or 
        name.endswith('self_attn.o_proj.weight')
    ):
        # temp[name] = param.to(dtype=torch.float8_e4m3fn)
        att[name] = param
        print(att[name].dtype)

# Example usage
model = 'model.layers.0.self_attn.q_proj.weight'
weights = att[model]


def extract_exp_mantissa(fp8_tensor):
    if fp8_tensor.dtype != torch.float8_e4m3fn:
        raise ValueError("Input tensor must be of type torch.float8_e4m3fn")

    int_repr = fp8_tensor.view(torch.int8)
    int_repr = torch.flatten(int_repr, start_dim=0)


    # Masks for extracting mantissa and exponent
    mantissa_mask = 0b00000111  # 3 bits mask for mantissa
    exp_mask = 0b00001111       # 4 bits mask for exponent

    mantissas = int_repr & mantissa_mask
    exponents = (int_repr >> 3) & exp_mask

    max_mantissa, max_idx = torch.max(mantissas, dim=0)  # Use dim=0 for 1D result across columns
    # max_mantissa = max_mantissa.max(dim=0)
    corresponding_exponent = exponents[max_idx]


    return max_mantissa, corresponding_exponent  

def max_convolution(weights, window_size=(16, 16)):
    w_height, w_width = weights.shape
    k_height, k_width = window_size

    # Initialize an output tensor to store the max mantissas
    output_shape = (w_height - k_height + 1, w_width - k_width + 1)
    max_mantissas_output = torch.zeros(output_shape, dtype=torch.int8)
    cor_exp_output = torch.zeros(output_shape, dtype=torch.int8)  # If you plan to use it later

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            current_window = weights[i:i + k_height, j:j + k_width]
            
            # Extract mantissas from the current window
            max_mantissa, exp = extract_exp_mantissa(current_window)
    
            max_mantissas_output[i, j] = max_mantissa.item()

            cor_exp_output[i, j] = exp.item()  # Get the corresponding exponent
            # x = x+1

    return max_mantissas_output, cor_exp_output


mantissas,exps = max_convolution(weights, window_size)

# Save results
mantissas_flat = mantissas.numpy().flatten()
exps_flat = exps.numpy().flatten()

with open('result.txt', 'w') as f:
    for value, exp in zip(mantissas_flat, exps_flat):  
        f.write(f"{exp} {value}\n")  
