# QuanKAN (Camera-Ready)

This folder provides a clean, standalone, Q1-journal-ready implementation of **QuanKAN V4** for EEG emotion recognition.

## Architecture
- **Front-End Enhancement**: Multi-scale STFT feature extraction with Spectrogram MobileNetV3-Lite and learnable temporal/spatial graph conditioning (`EarlyEEGFeatureEnhancer`).
- **Spatio-Temporal Backbone**: Dynamic Multi-Head Spatial Attention, Spatial Token Pooling (62 -> 8 tokens), 1D Sinusoidal Positional Encoding, and Bottleneck Temporal Convolutions.
- **Quantum-KAN Hybrid Residual**:
  - 4-qubit, 4-layer Parameterized Quantum Circuit with data re-uploading and alternating ring entanglement.
  - Executed on PennyLane `default.qubit` state-vector simulation.
  - Classical KAN classifier and Entropy-driven Uncertainty Gate controlling quantum corrections.

## Ablation & Extended Ablation Matrix
To evaluate component importance and extended sweeps, run using `--variant <name>`:
- `ablation.py` contains all ablation models.
- **Component Ablations**: `no_quantum_entanglement`, `no_quantum_rotations`, `z_only_quantum_measurements`, `no_entropy_gate`, `fixed_quantum_scale`, `linear_classical_kan`, `linear_final_kan`, `no_second_spatial_gat`, `no_positional_encoding`, `mean_temporal_pool`.
- **Extended Sweeps**: `matched_mlp_final_head`, `qubits_2`, `qubits_6`, `qubits_8`, `circuit_layers_1`, `circuit_layers_2`, `circuit_layers_3`.

## File Structure
- `model.py`: Complete, self-contained QuanKAN V4 architecture with zero third-party dependencies beyond PyTorch and PennyLane.
- `train_loso.py`: Strict Leave-One-Subject-Out (LOSO) cross-validation runner (150 epochs, CosineAnnealingLR, source-only normalization).
- `channel_62_pos.locs`: Standard 62-channel 3D electrode positions.

## Usage
To run 150-epoch strict LOSO cross-validation:
```bash
python3 train_loso.py --dataset seediv --epochs 150 --batch_size 32 --lr 3e-4 --output_dir results_quankan_loso
```
