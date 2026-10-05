from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > model_checker.py
"""

import torch

checkpoint_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/best_p3vae_model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("--- Checkpoint Summary ---")
print(f"Best Epoch Saved: {checkpoint.get('epoch', 'N/A')}")
print(f"Best Val Loss:   {checkpoint.get('val_loss', 'N/A'):.6f}")

# Optional: Inspect saved parameters or structure keys
print(f"Keys saved in file: {list(checkpoint.keys())}")