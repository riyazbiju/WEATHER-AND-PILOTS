"""
Neural network modules for Latent ODE model:
- Set Encoder with permutation-invariant aggregation
- ODE dynamics function
- Spatially-conditioned decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, activation='relu', dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Build layers
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            # No activation/dropout on last layer
            if i < num_layers - 1:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'elu':
                    layers.append(nn.ELU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class SetEncoder(nn.Module):
    """
    Encodes a set of context observations into latent posterior distribution.
    Permutation-invariant via aggregation.
    """
    
    def __init__(self, config, n_features=None):
        super().__init__()
        
        self.config = config
        self.latent_dim = config.LATENT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        
        # Determine number of features (account for circular vars encoded as sin/cos)
        if n_features is None:
            n_features = len(config.FEATURE_COLS)
            # Add extra dimension for each circular variable (sin/cos = 2 dims instead of 1)
            for var in config.CIRCULAR_VARS:
                if var in config.FEATURE_COLS:
                    n_features += 1  # one extra (since we replace 1 var with 2)
        
        # Input: [features, coords (4)] dimensions
        input_dim = n_features + 4
        
        # Individual observation encoder
        self.obs_encoder = MLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=config.ENCODER_LAYERS
        )
        
        # Aggregation method
        self.aggregation = config.AGGREGATION
        
        if self.aggregation == 'attention':
            self.attention = MultiHeadAttention(
                embed_dim=self.hidden_dim,
                num_heads=config.ATTENTION_HEADS
            )
        
        # Posterior network (aggregated representation -> latent params)
        self.fc_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
    
    def forward(self, context_features, context_coords, context_mask):
        """
        Args:
            context_features: [B, K, F] context observation features
            context_coords: [B, K, 4] context coordinates (time, lat, lon, alt)
            context_mask: [B, K] mask (1 for valid, 0 for padding)
            
        Returns:
            z0_mean: [B, latent_dim]
            z0_logvar: [B, latent_dim]
        """
        B, K, F = context_features.shape
        
        # Concatenate features and coordinates
        context_input = torch.cat([context_features, context_coords], dim=-1)  # [B, K, F+4]
        
        # Encode each observation independently
        obs_embeddings = self.obs_encoder(context_input)  # [B, K, hidden_dim]
        
        # Mask out padding
        obs_embeddings = obs_embeddings * context_mask.unsqueeze(-1)
        
        # Aggregate (permutation-invariant)
        if self.aggregation == 'mean':
            # Masked mean
            sum_embeddings = obs_embeddings.sum(dim=1)  # [B, hidden_dim]
            count = context_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            aggregated = sum_embeddings / count  # [B, hidden_dim]
            
        elif self.aggregation == 'max':
            # Masked max (set padding to very negative value)
            masked_embeddings = obs_embeddings.clone()
            masked_embeddings[context_mask == 0] = -1e4
            aggregated = masked_embeddings.max(dim=1)[0]  # [B, hidden_dim]
            
        elif self.aggregation == 'attention':
            # Self-attention aggregation
            aggregated = self.attention(obs_embeddings, context_mask)  # [B, hidden_dim]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Compute posterior parameters
        z0_mean = self.fc_mean(aggregated)
        z0_logvar = self.fc_logvar(aggregated)
        
        return z0_mean, z0_logvar


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for set aggregation"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask):
        """
        Args:
            x: [B, K, embed_dim]
            mask: [B, K]
            
        Returns:
            aggregated: [B, embed_dim]
        """
        B, K, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, K, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, K, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, K, K]
        
        # Apply mask
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
        scores = scores.masked_fill(mask_expanded == 0, -1e4)
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, K, head_dim]
        out = out.transpose(1, 2).reshape(B, K, D)  # [B, K, embed_dim]
        
        # Project
        out = self.out(out)
        
        # Global mean pooling (masked)
        out = out * mask.unsqueeze(-1)
        aggregated = out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        return aggregated


class ODEFunc(nn.Module):
    """
    Neural ODE dynamics function: dz/dt = f(z, t)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.latent_dim = config.LATENT_DIM
        self.hidden_dim = config.ODE_HIDDEN_DIM
        
        # Input: latent state + time
        self.net = MLP(
            input_dim=self.latent_dim + 1,  # z + t
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
            num_layers=3,
            activation='tanh'
        )
    
    def forward(self, t, z):
        """
        Args:
            t: scalar time
            z: [B, latent_dim] latent state
            
        Returns:
            dz_dt: [B, latent_dim] time derivative
        """
        # Create time features
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:  # scalar tensor
                t_vec = t.expand(z.shape[0], 1)
            elif t.dim() == 1:  # [B] tensor
                t_vec = t.unsqueeze(1)  # [B, 1]
            else:  # already [B, 1]
                t_vec = t
        else:
            t_vec = torch.ones(z.shape[0], 1, device=z.device) * t
        
        # Concatenate z and t
        zt = torch.cat([z, t_vec], dim=-1)
        
        # Compute derivative
        dz_dt = self.net(zt)
        
        return dz_dt


class SpatialDecoder(nn.Module):
    """
    Decodes latent state + spatial coordinates to meteorological variables
    """
    
    def __init__(self, config, n_features=None):
        super().__init__()
        
        self.latent_dim = config.LATENT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        
        # Determine output dimension (account for circular vars as sin/cos)
        if n_features is None:
            self.output_dim = len(config.FEATURE_COLS)
            for var in config.CIRCULAR_VARS:
                if var in config.FEATURE_COLS:
                    self.output_dim += 1  # one extra for sin/cos encoding
        else:
            self.output_dim = n_features
        
        # Input: latent state + spatial coords (lat, lon, alt)
        self.net = MLP(
            input_dim=self.latent_dim + 3,  # z + spatial coords
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=config.DECODER_LAYERS
        )
    
    def forward(self, z, spatial_coords):
        """
        Args:
            z: [B, latent_dim] latent state at query time
            spatial_coords: [B, 3] spatial coordinates (lat, lon, alt)
            
        Returns:
            predictions: [B, output_dim] predicted meteorological variables
        """
        # Concatenate latent state with spatial coords
        features = torch.cat([z, spatial_coords], dim=-1)
        
        # Decode
        predictions = self.net(features)
        
        return predictions


class LatentODEModel(nn.Module):
    """
    Complete Latent ODE model:
    1. Encode context set -> latent posterior
    2. Sample latent state z0
    3. Evolve via ODE: z0 -> z(t*)
    4. Decode with spatial conditioning
    """
    
    def __init__(self, config, n_features=None):
        super().__init__()
        
        self.config = config
        self.latent_dim = config.LATENT_DIM
        
        # Determine feature dimension
        if n_features is None:
            self.n_features = len(config.FEATURE_COLS)
            for var in config.CIRCULAR_VARS:
                if var in config.FEATURE_COLS:
                    self.n_features += 1
        else:
            self.n_features = n_features
        
        # Components
        self.encoder = SetEncoder(config, n_features=self.n_features)
        self.ode_func = ODEFunc(config)
        self.decoder = SpatialDecoder(config, n_features=self.n_features)
        
        # Prior: standard normal
        self.register_buffer('prior_mean', torch.zeros(self.latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(self.latent_dim))
    
    def encode(self, context_features, context_coords, context_mask):
        """Encode context to latent posterior"""
        return self.encoder(context_features, context_coords, context_mask)
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, spatial_coords):
        """Decode latent state to predictions"""
        return self.decoder(z, spatial_coords)
    
    def forward(self, context_features, context_coords, context_mask, 
                query_coords, ode_solver=None):
        """
        Full forward pass
        
        Args:
            context_features: [B, K, F]
            context_coords: [B, K, 4]
            context_mask: [B, K]
            query_coords: [B, 4] (time, lat, lon, alt)
            ode_solver: Optional ODE solver (if None, no ODE integration)
            
        Returns:
            predictions: [B, F] predicted meteorological variables
            z0_mean, z0_logvar: Latent posterior parameters
        """
        # 1. Encode context
        z0_mean, z0_logvar = self.encode(context_features, context_coords, context_mask)
        
        # 2. Sample latent state
        z0 = self.reparameterize(z0_mean, z0_logvar)
        
        # 3. Evolve via ODE (if solver provided)
        if ode_solver is not None:
            # Extract time coordinates
            t0 = context_coords[:, :, 0].mean(dim=1)  # Mean context time
            t_query = query_coords[:, 0]  # Query time
            
            # Integrate ODE from t0 to t_query
            z_t = ode_solver.integrate(self.ode_func, z0, t0, t_query)
        else:
            # No temporal evolution (for debugging/testing)
            z_t = z0
        
        # 4. Decode with spatial conditioning
        spatial_coords = query_coords[:, 1:]  # [B, 3] (lat, lon, alt)
        predictions = self.decode(z_t, spatial_coords)
        
        return predictions, z0_mean, z0_logvar
    
    def compute_loss(self, predictions, targets, z0_mean, z0_logvar, kl_weight=1.0):
        """
        Compute ELBO loss
        
        Args:
            predictions: [B, F] model predictions
            targets: [B, F] ground truth
            z0_mean, z0_logvar: Latent posterior parameters
            kl_weight: Weight for KL term (for annealing)
            
        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(predictions, targets, reduction='mean')
        
        # KL divergence: KL[q(z|x) || p(z)]
        # For multivariate Gaussian: 0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + z0_logvar - z0_mean.pow(2) - z0_logvar.exp())
        kl_loss = kl_loss / predictions.shape[0]  # Average over batch
        
        # Total loss (ELBO)
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
