"""
jvlee_LIBS_ML>utils>post_processing.py

Function definitions for post processing of the trained model.

"""

# region Imports
# region plain
import torch 
# endregion

# region as
import numpy as np
from torch import nn
# endregion
# endregion


def run_spectral_inference(
        model: nn.Module,
        X_samples: np.ndarray,
        y_scaler,
        device: torch.device = torch.device('xpu'),
) -> np.ndarray:
    """
    Run model inference on a batch of samples and return predictions in
    *physical* (inverse-transformed) units.
 
    Returns:
        preds_physical: np.ndarray of shape (N, n_wavelengths)
    """
    model.eval()
    with torch.no_grad():
        X_t   = torch.from_numpy(X_samples.astype(np.float32)).to(device)
        preds = model(X_t).cpu().numpy()          # (N, n_wavelengths), scaled
 
    preds_physical = y_scaler.inverse_transform(preds)
    return preds_physical

def accuracy(outputs, labels):
    return (outputs.argmax(dim=1) == labels).float().mean().item() * 100
