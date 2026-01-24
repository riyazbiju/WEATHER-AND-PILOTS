# Set-Based Latent ODE for Weather Prediction

A complete implementation of a **Set-Based Latent ODE model** that works directly with individual weather observations (without profile grouping) while maintaining powerful temporal dynamics modeling.

## 🌟 Key Features

- **Observation-based architecture**: Works directly with raw observations, no profile creation needed
- **Permutation-invariant encoding**: Set encoder with attention-based aggregation
- **Neural ODE dynamics**: Captures temporal evolution in latent space
- **Spatially-conditioned decoding**: Incorporates spatial coordinates for predictions
- **Multiple split strategies**: Random, temporal, and station-based evaluation
- **Efficient context selection**: K-NN search in spacetime with causality enforcement

## 📁 Project Structure

```
ODE/
├── config.py                  # Configuration settings
├── data_preprocessing.py      # Data loading, cleaning, normalization
├── context_selection.py       # K-NN context selection in spacetime
├── models.py                  # Neural network architectures
│   ├── SetEncoder            # Permutation-invariant encoder
│   ├── ODEFunc               # Neural ODE dynamics
│   ├── SpatialDecoder        # Spatially-conditioned decoder
│   └── LatentODEModel        # Complete model
├── ode_solver.py              # ODE integration methods
├── dataset.py                 # PyTorch dataset and dataloaders
├── trainer.py                 # Training loop with ELBO loss
├── evaluator.py               # Evaluation with metrics
├── visualization.py           # Plotting utilities
├── main_train.py              # Main training script
├── inference.py               # Inference script
└── data_5_with_pressure.csv   # Weather data
```

## 🎯 Theoretical Framework

### Architecture Overview

```
1. CONTEXT SELECTION
   For each query point (t*, lat*, lon*, alt*):
   - Select K nearest neighbors in spacetime
   - Context = {(xi, ti, lati, loni, alti)} for i=1..K

2. SET ENCODER (Permutation-Invariant)
   - Process each context observation independently
   - Aggregate using attention mechanism
   - Output: z0_mean, z0_std (latent posterior)

3. LATENT ODE
   - Sample z0 ~ N(z0_mean, z0_std)
   - Evolve: z(t*) = ODESolve(f_θ, z0, t0 → t*)

4. SPATIAL DECODER
   - Condition on query coordinates (lat*, lon*, alt*)
   - Predict: x̂ = Decoder(z(t*), lat*, lon*, alt*)
```

### Mathematical Formulation

**Encoder (Recognition Network):**
```
q_φ(z_0 | C) = N(z_0; μ_φ(C), σ_φ(C))
```

**Latent Dynamics:**
```
dz/dt = f_θ(z, t)
z(t*) = z_0 + ∫[t_0 to t*] f_θ(z(τ), τ) dτ
```

**Decoder (Likelihood):**
```
p_ψ(x* | z(t*), s*) = N(x*; μ_ψ(z(t*), s*), σ²I)
```

**Loss Function (ELBO):**
```
L = E_q[log p(x*|z(t*), s*)] - KL[q(z_0|C) || p(z_0)]
```

## 🚀 Quick Start

### Installation

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn tqdm torchdiffeq
```

### Basic Training

```bash
# Random split (default)
python main_train.py --split random --epochs 50

# Temporal split (temporal generalization)
python main_train.py --split temporal --epochs 100

# Station split (spatial generalization)
python main_train.py --split station --epochs 100
```

### Custom Training

```bash
python main_train.py \
    --split random \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --k_neighbors 30 \
    --latent_dim 64
```

### Making Predictions

```bash
# Predict on all test samples
python inference.py --checkpoint checkpoints/best_model.pt

# Predict on subset
python inference.py --checkpoint checkpoints/best_model.pt --n_samples 1000 --output my_predictions.csv
```

## ⚙️ Configuration

Key parameters in [`config.py`](config.py):

### Data Parameters
- `FEATURE_COLS`: `['temperature', 'windspeed', 'winddirection', 'visibility', 'pressure']`
- `COORD_COLS`: `['latitude', 'longitude', 'altitude']`

### Context Selection
- `K_NEIGHBORS`: 30 (number of context neighbors)
- `TIME_WEIGHT`: 1.0 (temporal distance weight)
- `SPACE_WEIGHT`: 1.0 (spatial distance weight)

### Model Architecture
- `LATENT_DIM`: 64 (latent ODE dimension)
- `HIDDEN_DIM`: 128 (hidden layer size)
- `AGGREGATION`: 'attention' (set aggregation method)
- `ATTENTION_HEADS`: 4

### Training
- `BATCH_SIZE`: 128
- `LEARNING_RATE`: 1e-3
- `NUM_EPOCHS`: 100
- `KL_ANNEALING_EPOCHS`: 20

### ODE Solver
- `ODE_SOLVER`: 'dopri5' (Runge-Kutta 4-5)
- `ODE_RTOL`: 1e-3
- `ODE_ATOL`: 1e-4

## 📊 Data Format

Input CSV should have columns:
```
station, datetime, latitude, longitude, altitude,
temperature, windspeed, winddirection, visibility, pressure
```

The model automatically:
- Converts datetime to hours
- Normalizes all features and coordinates
- Creates spacetime index for context selection

## 🎓 Model Components

### 1. Set Encoder
```python
SetEncoder(
    input_dim=9,        # features (5) + coords (4)
    hidden_dim=128,
    latent_dim=64,
    aggregation='attention'
)
```

### 2. ODE Function
```python
ODEFunc(
    latent_dim=64,
    hidden_dim=64
)
# Computes dz/dt = f_θ(z, t)
```

### 3. Spatial Decoder
```python
SpatialDecoder(
    latent_dim=64,      # z(t*)
    spatial_dim=3,      # lat, lon, alt
    output_dim=5        # meteorological variables
)
```

## 📈 Training Process

1. **Data Preparation**
   - Load and clean observations
   - Split data (random/temporal/station)
   - Fit scalers on training set
   - Normalize all splits

2. **Context Selection**
   - Build KD-tree for efficient K-NN search
   - For temporal split: enforce causality (context from past only)

3. **Training Loop**
   - Sample batch of observations
   - Get K nearest neighbors as context
   - Encode context → latent posterior
   - Integrate ODE: z0 → z(t*)
   - Decode with spatial conditioning
   - Compute ELBO loss
   - KL annealing (0 → 1 over 20 epochs)

4. **Validation**
   - Monitor reconstruction and KL losses
   - Save best model based on validation loss
   - Learning rate scheduling

## 📉 Evaluation Metrics

The model reports metrics in **original scale** (not normalized):

### Per-Feature Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

### Stratified Analysis
- **By Altitude**: Performance across altitude bins
- **By Time**: Temporal evolution of errors

### Visualizations
- Training curves (loss, reconstruction, KL)
- Predictions vs targets scatter plots
- Residual distributions
- Altitude/time stratified performance

## 🔬 Three Split Strategies

### 1. Random Split (70/15/15)
```python
python main_train.py --split random
```
- **Use case**: General performance evaluation
- **Train**: Random 70% of observations
- **Test**: Random 15% of observations
- **Challenge**: IID assumption

### 2. Temporal Split
```python
python main_train.py --split temporal
```
- **Use case**: Future forecasting
- **Train**: Observations from t < t_cutoff
- **Test**: Observations from t >= t_cutoff
- **Challenge**: Temporal extrapolation
- **Note**: Enforces causal context (only past observations)

### 3. Station Split
```python
python main_train.py --split station
```
- **Use case**: New station predictions
- **Train**: Observations from 180 stations
- **Test**: Observations from 30 held-out stations
- **Challenge**: Spatial extrapolation

## 🔍 Key Differences from Profile-Based Approach

| Aspect | Profile-Based (Old) | Observation-Based (New) |
|--------|---------------------|-------------------------|
| **Input** | Pre-grouped profiles | Individual observations |
| **Context** | Implicit (within profile) | Explicit (K-NN selection) |
| **Sequence** | Fixed order | Permutation-invariant |
| **Flexibility** | Requires profile logic | Works with any sampling |
| **Scalability** | Limited by profile size | Adjustable context size |

## 💡 Usage Tips

### For Best Performance

1. **Tune K_NEIGHBORS**: Start with 20-50, adjust based on data density
2. **KL Annealing**: Essential for stable training, prevents posterior collapse
3. **Spatial Weights**: Balance time vs space in context selection
4. **ODE Solver**: Dopri5 for accuracy, RK4 for speed
5. **Batch Size**: Larger batches → more stable gradients

### For Temporal Forecasting

```python
config.SPLIT_TYPE = 'temporal'
config.ENFORCE_CAUSALITY = True  # Only use past observations
config.K_NEIGHBORS = 50  # More context for better predictions
```

### For Spatial Extrapolation

```python
config.SPLIT_TYPE = 'station'
config.SPACE_WEIGHT = 2.0  # Emphasize spatial similarity
config.LATENT_DIM = 128  # Larger latent space for spatial patterns
```

## 📚 Output Files

### Checkpoints (`checkpoints/`)
- `best_model.pt`: Best model based on validation loss
- `checkpoint_epoch_N.pt`: Regular checkpoints
- `training_history.json`: Loss curves data

### Results (`results/`)
- `metrics.json`: Comprehensive evaluation metrics
- `training_curves.png`: Loss evolution plots
- `predictions_vs_targets.png`: Scatter plots per feature
- `residuals.png`: Residual distributions

### Predictions
- `predictions.csv`: Query coords, true values, predictions, errors

## 🎯 Next Steps

### Model Improvements
- [ ] Add uncertainty quantification (predictive variance)
- [ ] Implement hierarchical latent ODE
- [ ] Multi-resolution ODE (different timescales)
- [ ] Graph-based context (station connections)

### Features
- [ ] Online learning (incremental updates)
- [ ] Multi-step forecasting
- [ ] Missing data imputation
- [ ] Anomaly detection

### Optimization
- [ ] Distributed training
- [ ] Mixed precision training
- [ ] Model compression/distillation
- [ ] Faster context search (approximate NN)

## 🐛 Troubleshooting

### OOM (Out of Memory)
- Reduce `BATCH_SIZE`
- Reduce `K_NEIGHBORS`
- Use gradient accumulation

### Poor Convergence
- Increase `KL_ANNEALING_EPOCHS`
- Lower learning rate
- Check data normalization

### Slow Training
- Use simpler ODE solver ('rk4' instead of 'dopri5')
- Reduce `K_NEIGHBORS`
- Use fewer ODE integration steps

## 📖 References

1. **Latent ODEs for Irregularly-Sampled Time Series** (Rubanova et al., 2019)
2. **Neural Ordinary Differential Equations** (Chen et al., 2018)
3. **Deep Sets** (Zaheer et al., 2017)
4. **Conditional Neural Processes** (Garnelo et al., 2018)

## 📧 Citation

If you use this code, please cite:

```bibtex
@software{latent_ode_weather,
  title={Set-Based Latent ODE for Weather Prediction},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/latent-ode-weather}
}
```

---

## 🎉 Acknowledgments

Built on the theoretical framework combining:
- **Latent ODE** for temporal dynamics
- **Set-based models** for permutation invariance
- **Spatially-conditioned decoding** for heterogeneous observations

Perfect for irregular, multi-source weather data! 🌦️
