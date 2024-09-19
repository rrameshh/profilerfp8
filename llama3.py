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

# # Extract weights for attention and MLP layers
# selfattn_weights = {}
attention_weights = {}
mlp_weights = {}

for name, param in weights.items():

    if 'attention' in name and 'weight' in name:
        print(name)
        if param.dtype is torch.float8_e4m3fn:
            attention_weights[name] = param.to(dtype=torch.float8_e4m3fn)
        else:
            attention_weights[name] = param
    elif 'mlp' in name and 'weight' in name:
        if param.dtype is torch.float8_e4m3fn:
            mlp_weights[name] = param.to(dtype=torch.float8_e4m3fn)
        else:
            mlp_weights[name] = param

# for name in attention_weights:
#     print(f"{name}: {attention_weights[name].shape}")

# for name in mlp_weights:
#     print(f"{name}: {mlp_weights[name].shape}")

# Create a directory to save the weights
save_dir = "saved_weights"
os.makedirs(save_dir, exist_ok=True)

# def sparsity(weight):


def analyze_and_save_weights(weights, layer_type):
    for name, weight in weights.items():

        if weight.dtype == torch.float8_e4m3fn:
            weight = weight.to(torch.float32)

        weight_np = weight.detach().cpu().numpy()

        # Calculate sparsity
        num_zeroes = np.sum(weight_np == 0)
        total_elements = weight_np.size
        sparsity = num_zeroes / total_elements
        print(f"Sparsity of {name}: {sparsity:.4f} ({num_zeroes}/{total_elements})")

        # Histogram of weight distributions
        plt.hist(weight_np.flatten(), bins=50, alpha=0.75)
        plt.title(f"Weight Distribution for {name} ({layer_type})")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid()
        plt.savefig(os.path.join(save_dir, f"histogram_{layer_type}_{name.replace('/', '_')}.png"))
        plt.close()  # Close the figure to save memory

        # Low-order magnitude analysis
        low_order_count = np.sum(np.abs(weight_np) < 1e-3)  # Adjust threshold as needed
        print(f"Low-order magnitudes in {name}: {low_order_count} ({low_order_count / total_elements:.4f})")

        # Save weights
        torch.save(weight, os.path.join(save_dir, f"{layer_type}_{name.replace('/', '_')}.pt"))

# Analyze and save attention weights
print("Analyzing and saving attention weights:")
analyze_and_save_weights(attention_weights, "attention")

# Analyze and save MLP weights
print("\nAnalyzing and saving MLP weights:")
analyze_and_save_weights(mlp_weights, "mlp")
