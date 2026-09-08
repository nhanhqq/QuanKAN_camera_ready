"""Ablation and Extended Ablation for QuanKAN V4.

Includes:
    - Component Ablations:
        - no_quantum_entanglement (removes CNOT gates)
        - no_quantum_rotations (trainable rotations set to fixed/identity)
        - z_only_quantum_measurements (measure only Pauli-Z, exclude X and ZZ)
        - no_entropy_gate (always open gate = 1.0)
        - fixed_quantum_scale (freeze quantum scale parameter)
        - linear_classical_kan (replace classical KAN with Linear)
        - linear_final_kan (replace final hybrid KAN with Linear)
        - no_second_spatial_gat (1-layer spatial attention)
        - mean_spatial_pool (parameter-free spatial average pooling)
        - no_positional_encoding (remove 1D temporal sinusoids)
        - mean_temporal_pool (global average temporal pooling)
    - Capacity and Extended Ablations:
        - matched_mlp_final_head (parameter-matched 2-layer MLP)
        - classical_residual_capacity_matched (replace PQC with classical residual)
        - qubits_2, qubits_6, qubits_8 (qubit count sweeps)
        - circuit_layers_1, circuit_layers_2, circuit_layers_3 (circuit depth sweeps)
"""

from __future__ import annotations

import argparse
import copy
from typing import Dict, List, Optional, Tuple

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import (
    EarlyEEGFeatureEnhancer,
    EEGSpatioTemporalBackbone,
    EntropyGate,
    KANLinear,
    QuanKANV4,
    QuantumBranch,
    grad_reverse,
)


class CustomQuantumBranch(nn.Module):
    """Customizable PQC for ablation (varying qubits, layers, entanglement, measurements)."""

    def __init__(
        self,
        in_features: int = 192,
        num_qubits: int = 4,
        num_layers: int = 4,
        out_features: int = 32,
        entanglement: bool = True,
        trainable_rotations: bool = True,
        z_only: bool = False,
        q_device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.entanglement = entanglement
        self.trainable_rotations = trainable_rotations
        self.z_only = z_only

        self.pre_compress = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        angle_dim = num_layers * num_qubits * 2
        self.proj_in = KANLinear(32, angle_dim, grid_size=5, spline_order=3)

        self.dev = qml.device(q_device, wires=num_qubits)
        if trainable_rotations:
            self.weights = nn.Parameter(0.05 * torch.randn(num_layers, num_qubits, 3))
        else:
            self.register_buffer("weights", torch.zeros(num_layers, num_qubits, 3))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: Tensor, weights: Tensor):
            for layer in range(num_layers):
                for q in range(num_qubits):
                    qml.RY(inputs[:, layer, q, 0], wires=q)
                    qml.RZ(inputs[:, layer, q, 1], wires=q)
                if trainable_rotations:
                    for q in range(num_qubits):
                        qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
                if entanglement and num_qubits > 1:
                    if layer % 2 == 0:
                        for q in range(num_qubits):
                            qml.CNOT(wires=[q, (q + 1) % num_qubits])
                    else:
                        for q in range(num_qubits):
                            qml.CNOT(wires=[(q + 1) % num_qubits, q])

            measurements = [qml.expval(qml.PauliZ(q)) for q in range(num_qubits)]
            if not z_only and num_qubits > 1:
                measurements += [qml.expval(qml.PauliX(q)) for q in range(num_qubits)]
                measurements += [
                    qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % num_qubits))
                    for q in range(num_qubits)
                ]
            return measurements

        self.qnode = circuit
        self.measure_dim = num_qubits if z_only or num_qubits == 1 else 3 * num_qubits
        self.proj_out = KANLinear(self.measure_dim, out_features, grid_size=5, spline_order=3)

    def forward(self, x: Tensor) -> Tensor:
        angles = self.proj_in(self.pre_compress(x))
        angles = torch.pi * torch.tanh(angles)
        angles = angles.reshape(-1, self.num_layers, self.num_qubits, 2)
        measurements = self.qnode(angles.cpu(), self.weights.cpu())
        quantum_features = torch.stack(measurements, dim=1).to(x.device).float()
        return self.proj_out(quantum_features)


def build_ablation_model(variant: str, num_classes: int = 4, num_subjects: int = 15, q_device: str = "default.qubit") -> QuanKANV4:
    """Instantiate QuanKAN V4 with the requested ablation intervention."""
    model = QuanKANV4(
        in_channels=5,
        num_nodes=62,
        num_classes=num_classes,
        num_subjects=num_subjects,
        q_device=q_device,
        quantum_enabled=True,
    )

    if variant == "full":
        return model

    # 1. Quantum ablations
    elif variant == "no_quantum_entanglement":
        model.quantum_residual = CustomQuantumBranch(
            in_features=model.embedding_dim,
            num_qubits=4,
            num_layers=4,
            out_features=32,
            entanglement=False,
            q_device=q_device,
        )
    elif variant == "no_quantum_rotations":
        model.quantum_residual = CustomQuantumBranch(
            in_features=model.embedding_dim,
            num_qubits=4,
            num_layers=4,
            out_features=32,
            trainable_rotations=False,
            q_device=q_device,
        )
    elif variant == "z_only_quantum_measurements":
        model.quantum_residual = CustomQuantumBranch(
            in_features=model.embedding_dim,
            num_qubits=4,
            num_layers=4,
            out_features=32,
            z_only=True,
            q_device=q_device,
        )
    elif variant == "no_entropy_gate":
        class AlwaysOpenGate(nn.Module):
            def forward(self, _, logits):
                return torch.ones((logits.size(0), 1), device=logits.device), torch.zeros((logits.size(0), 1), device=logits.device)
        model.uncertainty_gate = AlwaysOpenGate()
    elif variant == "fixed_quantum_scale":
        model.quantum_scale.requires_grad_(False)

    # 2. Classifier / KAN ablations
    elif variant == "linear_classical_kan":
        model.classical_classifier = nn.Linear(model.embedding_dim, num_classes)
    elif variant == "linear_final_kan":
        model.final_classifier = nn.Linear(model.embedding_dim, num_classes)
    elif variant == "matched_mlp_final_head":
        model.final_classifier = nn.Sequential(
            nn.Linear(model.embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    # 3. Backbone ablations
    elif variant == "no_second_spatial_gat":
        model.backbone.spatial_gat2 = nn.Identity()
    elif variant == "no_positional_encoding":
        model.backbone.pos_enc = nn.Identity()
    elif variant == "mean_temporal_pool":
        class MeanPool(nn.Module):
            def forward(self, x):
                return x.mean(dim=-1)
        model.backbone.temporal_pool = MeanPool()

    # 4. Sensitivity & Extended sweeps (Qubits / Depth)
    elif variant.startswith("qubits_"):
        n_q = int(variant.split("_")[1])
        model.quantum_residual = CustomQuantumBranch(
            in_features=model.embedding_dim,
            num_qubits=n_q,
            num_layers=4,
            out_features=32,
            q_device=q_device,
        )
    elif variant.startswith("circuit_layers_"):
        n_l = int(variant.split("_")[2])
        model.quantum_residual = CustomQuantumBranch(
            in_features=model.embedding_dim,
            num_qubits=4,
            num_layers=n_l,
            out_features=32,
            q_device=q_device,
        )
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")

    return model
