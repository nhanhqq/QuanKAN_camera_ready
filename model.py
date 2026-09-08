"""QuanKAN V4: Spatio-Temporal Graph Attention Network with Quantum Residual and KAN Classifiers.

Architecture:
    Input: [B, T, 62, 5] (DE-LDS EEG Features)
    1. EarlyEEGFeatureEnhancer:
       - SpectrogramMobileNetV3Lite (Electrode-averaged STFT -> MobileNetV3 -> 64-D context)
       - Dynamic node/band conditioning + learnable temporal 1D conv + graph spatial smoothing
    2. EEGSpatioTemporalBackbone:
       - Dynamic Spatial Self-Attention (2 layers)
       - Spatial Token Pooling (62 nodes -> 8 latent tokens)
       - Improved 1D Sinusoidal Positional Encoding
       - Bottleneck Temporal Depthwise Separable Convolutions
       - Lightweight Temporal Attention Pooling -> 256-D embedding
    3. Classical Projection:
       - LayerNorm + Linear + GELU -> 192-D clean embedding
       - Classical KAN Classifier
    4. Quantum Residual Branch:
       - 4 Qubits, 4 Layers variational circuit executed on PennyLane 'default.qubit'
       - Alternating ring CNOT entanglement
       - 12 Pauli expectation measurements (Z, X, ZZ)
       - Output KAN projection + Entropy Uncertainty Gate
    5. Final Hybrid Classifier:
       - Fuses classical and gated quantum residual -> final emotion logits
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> Tuple[Tensor, None]:
        return -ctx.strength * gradient, None


def grad_reverse(x: Tensor, strength: float = 1.0) -> Tensor:
    """Identity in forward; scale and flip gradient sign in backward."""
    return GradientReversalFunction.apply(x, strength)


class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network (KAN) linear layer with B-spline bases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Sequence[float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        del grid_eps
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        self.base_activation = base_activation()

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * step + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = (
            nn.Parameter(torch.empty(out_features, in_features))
            if enable_standalone_scale_spline
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = torch.rand(self.grid_size + 1, self.in_features, self.out_features)
            noise = (noise - 0.5) * self.scale_noise / self.grid_size
            interior_grid = self.grid.T[self.spline_order : -self.spline_order]
            coefficients = self._curve2coeff(interior_grid, noise)
            self.spline_weight.copy_(self.scale_spline * coefficients)
        if self.spline_scaler is not None:
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: Tensor) -> Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        for order in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(order + 1)]) / (grid[:, order:-1] - grid[:, : -(order + 1)] + 1e-8)
            right = (grid[:, order + 1 :] - x) / (grid[:, order + 1 :] - grid[:, 1:-order] + 1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def _curve2coeff(self, x: Tensor, y: Tensor) -> Tensor:
        basis = self.b_splines(x).transpose(0, 1)
        targets = y.transpose(0, 1)
        solution = torch.linalg.lstsq(basis, targets).solution
        return solution.permute(2, 0, 1).contiguous()

    def scaled_spline_weight(self) -> Tensor:
        if self.spline_scaler is None:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x_flat), self.scale_base * self.base_weight)
        spline_output = F.linear(
            self.b_splines(x_flat).reshape(x_flat.size(0), -1),
            self.scaled_spline_weight().reshape(self.out_features, -1),
        )
        return (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)


class DynamicSpatialAttention(nn.Module):
    """Multi-head self-attention across EEG electrode channels."""

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if out_channels % num_heads:
            raise ValueError("out_channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.qkv = nn.Linear(in_channels, 3 * out_channels, bias=False)
        self.proj = nn.Linear(out_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, num_nodes, _ = x.shape
        head_dim = self.out_channels // self.num_heads
        qkv = self.qkv(x).reshape(batch_size, num_nodes, 3, self.num_heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)

        if mask is not None:
            pair_mask = mask.unsqueeze(0) * mask.unsqueeze(1)
            scores = scores.masked_fill(pair_mask[None, None] == 0, -1e9)

        attention = self.dropout(F.softmax(scores, dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch_size, num_nodes, self.out_channels)
        output = self.proj(output)
        residual = x + output if x.shape[-1] == output.shape[-1] else output
        return self.norm(residual)


class SpatialTokenPooling(nn.Module):
    """Pool 62 electrode channels into K learnable latent tokens."""

    def __init__(self, hidden_dim: int = 32, num_tokens: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.queries = nn.Parameter(0.02 * torch.randn(num_tokens, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        keys = F.normalize(self.norm(x), dim=-1)
        queries = F.normalize(self.queries, dim=-1)
        scores = torch.einsum("bnh,kh->bkn", keys, queries)
        attention = F.softmax(scores / math.sqrt(self.hidden_dim), dim=-1)
        return torch.einsum("bkn,bnh->bkh", attention, x)


class ImprovedPositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding along temporal dimension [B, C, T]."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        frequencies = torch.arange(0, channels, 2).float() / channels
        self.register_buffer("inv_freq", 1.0 / (10000**frequencies))

    def forward(self, x: Tensor) -> Tensor:
        positions = torch.arange(x.size(2), device=x.device).type_as(self.inv_freq)
        phase = torch.einsum("i,j->ij", positions, self.inv_freq)
        encoding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        output = torch.zeros(1, x.size(1), x.size(2), device=x.device, dtype=x.dtype)
        output[0, : encoding.shape[1], :] = encoding.T
        return x + output


class BottleneckTemporalConv(nn.Module):
    """Depthwise separable temporal convolution with residual connection."""

    def __init__(
        self,
        channels: int = 256,
        bottleneck: int = 96,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1, bias=False),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
            nn.Conv1d(
                bottleneck,
                bottleneck,
                kernel_size,
                padding=kernel_size // 2,
                groups=bottleneck,
                bias=False,
            ),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
            nn.Conv1d(bottleneck, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.block[-2].weight)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class LightweightTemporalAttentionPooling(nn.Module):
    """Learned scalar attention pooling over temporal length."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        attention = F.softmax(self.score(x), dim=-1)
        return torch.sum(x * attention, dim=-1)


class EEGSpatioTemporalBackbone(nn.Module):
    """Backbone extracting spatio-temporal dynamics from enhanced EEG."""

    def __init__(
        self,
        num_nodes: int = 62,
        in_channels: int = 5,
        hidden_dim: int = 32,
        num_classes: int = 4,
        spatial_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.spatial_gat1 = DynamicSpatialAttention(in_channels, hidden_dim, num_heads=8)
        self.spatial_gat2 = DynamicSpatialAttention(hidden_dim, hidden_dim, num_heads=8)
        self.spatial_pool = SpatialTokenPooling(hidden_dim, spatial_tokens)

        temporal_dim = spatial_tokens * hidden_dim
        self.pos_enc = ImprovedPositionalEncoding1D(temporal_dim)
        self.temporal_conv1 = BottleneckTemporalConv(temporal_dim, bottleneck=96, kernel_size=3, dropout=0.10)
        self.temporal_conv2 = BottleneckTemporalConv(temporal_dim, bottleneck=96, kernel_size=5, dropout=0.10)
        self.temporal_pool = LightweightTemporalAttentionPooling(temporal_dim)

    def extract_features(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, time_steps, num_nodes, num_bands = x.shape
        flat = x.reshape(batch_size * time_steps, num_nodes, num_bands)
        spatial = self.spatial_gat2(self.spatial_gat1(flat, mask), mask)
        tokens = self.spatial_pool(spatial).reshape(batch_size, time_steps, -1).permute(0, 2, 1)
        temporal = self.pos_enc(tokens)
        temporal = self.temporal_conv1(temporal)
        temporal = self.temporal_conv2(temporal)
        return self.temporal_pool(temporal)


class QuantumBranch(nn.Module):
    """Parameterized Quantum Circuit (PQC) executed with PennyLane default.qubit."""

    def __init__(
        self,
        in_features: int = 192,
        num_qubits: int = 4,
        num_layers: int = 4,
        out_features: int = 32,
        q_device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

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
        self.weights = nn.Parameter(0.05 * torch.randn(num_layers, num_qubits, 3))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: Tensor, weights: Tensor):
            for layer in range(self.num_layers):
                for qubit in range(self.num_qubits):
                    qml.RY(inputs[:, layer, qubit, 0], wires=qubit)
                    qml.RZ(inputs[:, layer, qubit, 1], wires=qubit)
                for qubit in range(self.num_qubits):
                    qml.Rot(weights[layer, qubit, 0], weights[layer, qubit, 1], weights[layer, qubit, 2], wires=qubit)
                if layer % 2 == 0:
                    for qubit in range(self.num_qubits):
                        qml.CNOT(wires=[qubit, (qubit + 1) % self.num_qubits])
                else:
                    for qubit in range(self.num_qubits):
                        qml.CNOT(wires=[(qubit + 1) % self.num_qubits, qubit])

            measurements = [qml.expval(qml.PauliZ(q)) for q in range(self.num_qubits)]
            measurements += [qml.expval(qml.PauliX(q)) for q in range(self.num_qubits)]
            measurements += [
                qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % self.num_qubits))
                for q in range(self.num_qubits)
            ]
            return measurements

        self.qnode = circuit
        self.measure_dim = 3 * num_qubits
        self.proj_out = KANLinear(self.measure_dim, out_features, grid_size=5, spline_order=3)

    def forward(self, x: Tensor) -> Tensor:
        angles = self.proj_in(self.pre_compress(x))
        angles = torch.pi * torch.tanh(angles)
        angles = angles.reshape(-1, self.num_layers, self.num_qubits, 2)
        measurements = self.qnode(angles.cpu(), self.weights.cpu())
        quantum_features = torch.stack(measurements, dim=1).to(x.device).float()
        return self.proj_out(quantum_features)


class EntropyGate(nn.Module):
    """Uncertainty gate weighting quantum residuals based on classical entropy."""

    def __init__(self, num_classes: int, gamma: float = 1.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, _: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:
        probabilities = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1, keepdim=True)
        gate = (entropy / math.log(self.num_classes)).clamp(0.0, 1.0)
        return gate.pow(self.gamma), entropy


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale), inplace=True)
        return x * scale


class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels: int, expanded_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.Hardswish(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                3,
                stride=stride,
                padding=1,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.Hardswish(inplace=True),
            SqueezeExcite(expanded_channels),
        )
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.project(self.depthwise(self.expand(x)))
        return x + output if self.use_residual else output


class SpectrogramMobileNetV3Lite(nn.Module):
    """STFT over electrode-averaged DE-LDS features processed by a MobileNetV3."""

    def __init__(self, in_bands: int = 5, embedding_dim: int = 64, n_fft: int = 16, hop_length: int = 4) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.features = nn.Sequential(
            nn.Conv2d(in_bands, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            MobileNetV3Block(16, 48, 24, 1),
            MobileNetV3Block(24, 72, 32, 2),
            MobileNetV3Block(32, 96, 48, 1),
            MobileNetV3Block(48, 144, 64, 2),
            MobileNetV3Block(64, 192, 64, 1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def make_spectrogram(self, x: Tensor) -> Tensor:
        valid = (x.abs().sum(dim=(2, 3)) > 0).to(x.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        signal = x.mean(dim=2).transpose(1, 2)
        mask = valid[:, None, :]
        mu = (signal * mask).sum(dim=-1, keepdim=True) / denom[:, None, :]
        var = ((signal - mu).pow(2) * mask).sum(dim=-1, keepdim=True) / denom[:, None, :]
        signal = ((signal - mu) / var.sqrt().clamp_min(1e-4)) * mask
        signal = F.avg_pool1d(signal, kernel_size=3, stride=1, padding=1) * mask

        b, c, t = signal.shape
        spec = torch.stft(
            signal.reshape(b * c, t),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(signal),
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs().square()
        spec = torch.log1p(spec).reshape(b, c, spec.size(-2), spec.size(-1))
        spec_mu = spec.mean(dim=(-2, -1), keepdim=True)
        spec_sd = spec.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
        return (spec - spec_mu) / spec_sd

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(self.make_spectrogram(x))).flatten(1)


class EarlyEEGFeatureEnhancer(nn.Module):
    """Front-end enhancement preserving the input shape [B, T, 62, 5]."""

    def __init__(
        self,
        num_nodes: int = 62,
        in_bands: int = 5,
        num_classes: int = 4,
        adj: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_bands = in_bands
        self.spectral_context = SpectrogramMobileNetV3Lite(in_bands=in_bands, embedding_dim=64)
        self.conditioner = nn.Linear(64, num_nodes + in_bands)
        nn.init.normal_(self.conditioner.weight, std=0.005)
        nn.init.zeros_(self.conditioner.bias)
        self.context_to_input = nn.Linear(64, num_nodes * in_bands)
        nn.init.normal_(self.context_to_input.weight, std=0.002)
        nn.init.zeros_(self.context_to_input.bias)
        self.aux_classifier = nn.Linear(64, num_classes)
        self.aux_logits = None

        self.temporal_filter = nn.Sequential(
            nn.Conv1d(in_bands, in_bands, 5, padding=2, groups=in_bands, bias=False),
            nn.Conv1d(in_bands, in_bands, 1, bias=False),
        )
        self.temporal_scale = nn.Parameter(torch.full((in_bands,), -2.944439))
        self.graph_scale = nn.Parameter(torch.zeros(in_bands))

        graph = (
            torch.eye(num_nodes, dtype=torch.float32)
            if adj is None
            else torch.as_tensor(adj, dtype=torch.float32).clone()
        )
        graph = graph / graph.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("graph", graph)

    def forward(self, x: Tensor) -> Tensor:
        valid = (x.abs().sum(dim=(2, 3), keepdim=True) > 0).to(x.dtype)
        context = self.spectral_context(x)
        self.aux_logits = self.aux_classifier(context)

        gains = 0.25 * torch.tanh(self.conditioner(context))
        node_gain = gains[:, : self.num_nodes, None]
        band_gain = gains[:, self.num_nodes :][:, None, None, :]
        conditioned = x * (1.0 + node_gain[:, None]) * (1.0 + band_gain)

        context_shift = 0.20 * torch.tanh(self.context_to_input(context))
        context_shift = context_shift.reshape(-1, 1, self.num_nodes, self.in_bands)
        conditioned = conditioned + context_shift * valid

        b, t, n, f = conditioned.shape
        temporal = conditioned.permute(0, 2, 3, 1).reshape(b * n, f, t)
        temporal = self.temporal_filter(temporal).reshape(b, n, f, t).permute(0, 3, 1, 2)
        temporal_weight = torch.sigmoid(self.temporal_scale)[None, None, None]

        graph_smoothed = torch.einsum("nm,btmf->btnf", self.graph, conditioned)
        graph_delta = graph_smoothed - conditioned
        graph_weight = (0.20 * torch.tanh(self.graph_scale))[None, None, None]

        enhanced = conditioned + temporal_weight * temporal + graph_weight * graph_delta
        return enhanced * valid


class QuanKANV4(nn.Module):
    """QuanKAN V4 full model with STFT front-end and quantum residual classifier."""

    def __init__(
        self,
        in_channels: int = 5,
        num_nodes: int = 62,
        num_classes: int = 4,
        num_subjects: int = 15,
        adj: Optional[torch.Tensor] = None,
        q_device: str = "default.qubit",
        quantum_enabled: bool = True,
        spatial_tokens: int = 8,
        spatial_hidden: int = 32,
        embedding_dim: int = 192,
        use_intensity_head: bool = False,
    ) -> None:
        super().__init__()
        self.frontend = EarlyEEGFeatureEnhancer(
            num_nodes=num_nodes, in_bands=in_channels, num_classes=num_classes, adj=adj
        )
        temporal_dim = spatial_tokens * spatial_hidden
        self.backbone = EEGSpatioTemporalBackbone(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=spatial_hidden,
            num_classes=num_classes,
            spatial_tokens=spatial_tokens,
        )
        self.quantum_enabled = quantum_enabled
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.classical_proj = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, embedding_dim),
            nn.GELU(),
        )
        self.head_dropout = nn.Dropout(0.20)
        self.classical_classifier = KANLinear(embedding_dim, num_classes)
        self.quantum_residual = QuantumBranch(
            in_features=embedding_dim,
            num_qubits=4,
            num_layers=4,
            out_features=32,
            q_device=q_device,
        )
        self.uncertainty_gate = EntropyGate(num_classes, gamma=1.5)
        self.quantum_proj = nn.Linear(32, embedding_dim)
        nn.init.normal_(self.quantum_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.quantum_proj.bias)
        self.quantum_scale = nn.Parameter(torch.tensor(0.0))
        self.final_classifier = KANLinear(embedding_dim, num_classes)
        self.quantum_aux_head = nn.Linear(32, num_classes)
        nn.init.zeros_(self.quantum_aux_head.weight)
        nn.init.zeros_(self.quantum_aux_head.bias)
        self.subject_head = nn.Linear(embedding_dim, num_subjects)
        self.intensity_head = (
            nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1))
            if use_intensity_head
            else None
        )

    def forward(
        self,
        x: Tensor,
        lambda_adv: float = 0.0,
        q_strength: float = 1.0,
    ) -> Tuple[Tensor, ...]:
        x_enhanced = self.frontend(x)
        temporal_features = self.backbone.extract_features(x_enhanced)
        clean_embedding = self.classical_proj(temporal_features)
        classical_embedding = self.head_dropout(clean_embedding)
        classical_logits = self.classical_classifier(classical_embedding)

        if self.quantum_enabled:
            quantum_features = self.quantum_residual(clean_embedding.detach())
            gate, entropy = self.uncertainty_gate(clean_embedding, classical_logits.detach())
            quantum_residual = self.quantum_proj(quantum_features)
            alpha = 0.25 * torch.sigmoid(self.quantum_scale) * q_strength
            hybrid_embedding = classical_embedding + gate * alpha * quantum_residual
            emotion_logits = self.final_classifier(hybrid_embedding)
            quantum_aux_logits = self.quantum_aux_head(quantum_features)
        else:
            batch_size = clean_embedding.size(0)
            quantum_features = clean_embedding.new_zeros(batch_size, 32)
            gate = clean_embedding.new_zeros(batch_size, 1)
            entropy = clean_embedding.new_zeros(batch_size, 1)
            quantum_aux_logits = torch.zeros_like(classical_logits)
            emotion_logits = self.final_classifier(classical_embedding)

        subject_logits = self.subject_head(grad_reverse(clean_embedding, lambda_adv))
        intensity_prediction = (
            self.intensity_head(clean_embedding).squeeze(-1)
            if self.intensity_head is not None
            else clean_embedding.new_zeros(clean_embedding.size(0))
        )
        return (
            emotion_logits,
            subject_logits,
            clean_embedding,
            classical_logits,
            quantum_aux_logits,
            quantum_features,
            gate,
            entropy,
            intensity_prediction,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
