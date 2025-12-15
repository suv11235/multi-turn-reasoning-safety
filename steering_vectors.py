import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import RepresentationData from the extraction module
# Assuming it's available in the python path or same directory
from representation_extraction import RepresentationData

class SteeringVectorGenerator:
    """Generates steering vectors from extracted representations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_vector(self, harmful_reps: List[RepresentationData], 
                      benign_reps: List[RepresentationData],
                      layer_name: str) -> Optional[torch.Tensor]:
        """
        Compute steering vector as Mean(Harmful) - Mean(Benign).
        returns a tensor of shape [hidden_dim]
        """
        if not harmful_reps or not benign_reps:
            self.logger.warning(f"Insufficient data for layer {layer_name}")
            return None

        # Helper to get flat features
        def get_features(reps):
            features = []
            for r in reps:
                if r.layer_name != layer_name:
                    continue
                # Average over sequence length (and batch if present)
                # rep.representation is numpy array
                if len(r.representation.shape) == 3:
                    feat = np.mean(r.representation, axis=(0, 1))
                elif len(r.representation.shape) == 2:
                    feat = np.mean(r.representation, axis=0)
                else:
                    feat = r.representation
                features.append(feat)
            return features

        harmful_feats = get_features(harmful_reps)
        benign_feats = get_features(benign_reps)

        if not harmful_feats or not benign_feats:
            return None

        # Compute means
        mu_harmful = np.mean(np.stack(harmful_feats), axis=0)
        mu_benign = np.mean(np.stack(benign_feats), axis=0)

        # Steering vector: direction away from harmful
        # Or direction towards safety: Benign - Harmful
        # Usually "Circuit Breaker" subtracts the harmful concept.
        # So Vector = Mu_Harmful - Mu_Benign
        # Then we SUBTRACT this vector * alpha.
        steering_vector = mu_harmful - mu_benign
        
        return torch.tensor(steering_vector, dtype=torch.float32)

    def compute_all_layers(self, harmful_reps: List[RepresentationData],
                          benign_reps: List[RepresentationData]) -> Dict[str, torch.Tensor]:
        """Compute vectors for all available layers."""
        layers = set(r.layer_name for r in harmful_reps) | set(r.layer_name for r in benign_reps)
        vectors = {}
        
        for layer in layers:
            vec = self.compute_vector(harmful_reps, benign_reps, layer)
            if vec is not None:
                vectors[layer] = vec
                
        return vectors


class CircuitBreaker:
    """Applies steering vectors to interrupt harmful processing."""

    def __init__(self, model: torch.nn.Module, 
                 steering_vectors: Dict[str, torch.Tensor],
                 alpha: float = 1.0):
        self.model = model
        self.vectors = steering_vectors
        self.alpha = alpha
        self.hooks = []
        self.logger = logging.getLogger(__name__)
        self.device = next(model.parameters()).device

    def _create_hook(self, vector: torch.Tensor):
        # Move vector to correct device/dtype
        vector = vector.to(self.device).to(self.model.dtype)
        
        def hook_fn(module, input, output):
            # Output is usually (hidden_states, ...) tuple or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                is_tuple = True
            else:
                hidden_states = output
                is_tuple = False
                
            # Subtract steering vector
            # hidden_states: [batch, seq, hidden]
            # vector: [hidden]
            # usage: hidden - alpha * vector
            
            # Subtraction happens in-place or returns new tensor
            # Ensure broadcasting works
            
            perturbation = self.alpha * vector.view(1, 1, -1)
            modified_hidden = hidden_states - perturbation
            
            if is_tuple:
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
                
        return hook_fn

    def verify_compatibility(self):
        """Check if target layers exist in model."""
        for layer_name in self.vectors:
            try:
                # Try to access layer
                parts = layer_name.split('.')
                curr = self.model
                for p in parts:
                    curr = getattr(curr, p)
            except AttributeError:
                self.logger.warning(f"Layer {layer_name} not found in model")

    def register(self):
        """Register all hooks."""
        self.remove() # Clear existing
        
        for layer_name, vector in self.vectors.items():
            try:
                parts = layer_name.split('.')
                curr = self.model
                for p in parts:
                    curr = getattr(curr, p)
                
                hook = curr.register_forward_hook(self._create_hook(vector))
                self.hooks.append(hook)
                self.logger.info(f"Circuit breaker registered on {layer_name}")
            except Exception as e:
                self.logger.error(f"Failed to register hook on {layer_name}: {e}")

    def remove(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.logger.info("Circuit breakers removed")

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
