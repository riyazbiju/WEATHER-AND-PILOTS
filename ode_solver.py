"""
ODE solvers for latent dynamics integration
"""

import torch
from torchdiffeq import odeint


class ODESolver:
    """Wrapper for ODE integration"""
    
    def __init__(self, method='dopri5', rtol=1e-3, atol=1e-4):
        """
        Args:
            method: ODE solver method ('dopri5', 'rk4', 'euler', 'midpoint')
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol
    
    def integrate(self, ode_func, z0, t0, t1):
        """
        Integrate ODE from t0 to t1
        
        Args:
            ode_func: ODE dynamics function dz/dt = f(z, t)
            z0: [B, latent_dim] initial state
            t0: [B] or scalar - initial time(s)
            t1: [B] or scalar - final time(s)
            
        Returns:
            z1: [B, latent_dim] state at time t1
        """
        # Handle scalar vs vector times
        if isinstance(t0, torch.Tensor) and t0.ndim == 1:
            # Different times for each batch element - need batched integration
            return self._integrate_batch(ode_func, z0, t0, t1)
        else:
            # Same time for all batch elements - standard integration
            return self._integrate_single(ode_func, z0, t0, t1)
    
    def _integrate_single(self, ode_func, z0, t0, t1):
        """Integration with same time span for all batch elements"""
        # Convert to tensors
        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0, device=z0.device, dtype=z0.dtype)
        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1, device=z0.device, dtype=z0.dtype)
        
        # If t0 == t1, no integration needed
        if torch.allclose(t0, t1):
            return z0
        
        # Time span
        t_span = torch.stack([t0, t1])
        
        # Integrate
        z = odeint(
            ode_func,
            z0,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )
        
        # Return final state
        return z[-1]
    
    def _integrate_batch(self, ode_func, z0, t0, t1):
        """Integration with different time spans for each batch element"""
        B = z0.shape[0]
        results = []
        
        for i in range(B):
            # Integrate each trajectory separately
            if torch.allclose(t0[i], t1[i]):
                results.append(z0[i:i+1])
            else:
                t_span = torch.stack([t0[i], t1[i]])
                z = odeint(
                    ode_func,
                    z0[i:i+1],
                    t_span,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol
                )
                results.append(z[-1])
        
        return torch.cat(results, dim=0)


class SimpleODESolver:
    """
    Simple ODE solver using Euler or RK4 method.
    Useful when torchdiffeq is not available.
    """
    
    def __init__(self, method='rk4', n_steps=10):
        """
        Args:
            method: 'euler' or 'rk4'
            n_steps: Number of integration steps
        """
        self.method = method
        self.n_steps = n_steps
    
    def integrate(self, ode_func, z0, t0, t1):
        """Integrate ODE from t0 to t1"""
        # Convert to tensors
        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0, device=z0.device, dtype=z0.dtype)
        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1, device=z0.device, dtype=z0.dtype)
        
        # Handle scalar times
        if t0.ndim == 0:
            t0 = t0.unsqueeze(0).expand(z0.shape[0])
        if t1.ndim == 0:
            t1 = t1.unsqueeze(0).expand(z0.shape[0])
        
        # If t0 == t1, no integration needed
        if torch.allclose(t0, t1):
            return z0
        
        # Time step
        dt = (t1 - t0) / self.n_steps
        
        # Integration
        z = z0
        t = t0
        
        for _ in range(self.n_steps):
            if self.method == 'euler':
                z = z + dt.unsqueeze(1) * ode_func(t, z)
            elif self.method == 'rk4':
                k1 = ode_func(t, z)
                k2 = ode_func(t + dt / 2, z + dt.unsqueeze(1) * k1 / 2)
                k3 = ode_func(t + dt / 2, z + dt.unsqueeze(1) * k2 / 2)
                k4 = ode_func(t + dt, z + dt.unsqueeze(1) * k3)
                z = z + dt.unsqueeze(1) * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            t = t + dt
        
        return z
