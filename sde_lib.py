"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import diff
from utils import Reshape, Flatten, Merge

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  @abc.abstractmethod
  def PDE(self, score_fn,  
          normalize=True, reduction='mean', 
          list_dim=[], time_est='approx', div_est='approx'):
      """Fokker Planck system of PDEs"""
      pass


  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()



class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.name = 'vpsde'
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):    
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G

  def PDE(self, score_fn, 
          wgt='constant',  
          normalize=True, 
          reduction='mean', 
          list_dim=[], 
          time_est='approx', 
          div_est='approx', 
          train=True, 
          scalar_fp="False",
          m=1):
      
      sde_fn = self.sde
      flat_model = Merge(score_fn)
      class score_FP(self.__class__):
          def __init__(self):
              self.flat_model = flat_model
              self.normalize = normalize
              self.reduction = reduction 
              self.list_dim = list_dim
              self.time_est = time_est
              self.div_est = div_est       
              self.train = train
              self.wgt = wgt
              self.scalar_fp = scalar_fp
              self.m = m
              
          def fp(self, x, t):
            x = x.reshape((x.shape[0], -1))
            x.requires_grad = True
            t.requires_grad = True
            D = x.shape[-1]
            if self.list_dim == []:        
                self.list_dim = range(D)
            s = self.flat_model(x, t)                 
            if self.div_est == 'exact':
                div_s = diff.batch_div(self.flat_model, x, t)
            elif self.div_est == 'approx':    
                div_s = diff.hutch_div(self.flat_model, x, t)
            s_l22 = torch.linalg.norm(s, 2, dim=1, keepdim=False)**2    
            
            "Computing RHS"
            _, diffusion = sde_fn(x, t)

            if self.scalar_fp in ("True"):
                g_pow = diffusion**2 
                f_dot_s = torch.einsum('bs,bs->b', x, s)
                RHS = (g_pow/2) * ( div_s + s_l22 + f_dot_s )   
                RHS = RHS if self.train else RHS.cpu().detach().numpy()
                res = torch.linalg.norm(RHS, ord=2) if self.train \
                    else np.linalg.norm(RHS, ord=2) 
            elif self.scalar_fp in ("False"): 
                print("scalar_fp", self.scalar_fp)
                g_pow = (diffusion[:, None])**2           
                f_dot_s = torch.einsum('bs,bs->b', x, s) 
                RHS = (g_pow/2) * diff.gradient( div_s + s_l22 + f_dot_s, x)
                RHS = RHS if self.train else RHS.cpu().detach().numpy()
                
                "Computing LHS"
                if self.time_est == 'exact':
                    res = torch.zeros_like(t) if self.train else np.zeros_like(t.cpu().detach().numpy())   
                    for j in list_dim:
                        dsdt = diff.partial_t_j(self.flat_model, x, t, j)      
                        dsdt = dsdt if self.train else dsdt.cpu().detach().numpy()
                        residue = torch.clip((dsdt - RHS[0:,j])**2, max=1.0e+30) if self.train \
                            else np.clip((dsdt - RHS[0:,j])**2, a_min=0.0, a_max=1.0e+30)
                        res += residue # batch square-sum (of each coordinate); shape: [B,]
                    res = torch.sqrt(res) if self.train else np.sqrt(res)
                elif self.time_est == 'approx':
                    dsdt = diff.t_finite_diff(self.flat_model, x, t)
                    dsdt = dsdt if self.train else dsdt.cpu().detach().numpy()
                    error = (dsdt - RHS)
                    res = torch.linalg.norm(error, ord=2, dim=1) if self.train \
                        else np.linalg.norm((dsdt - RHS), ord=2, axis=1)
                    print("dsdt", torch.linalg.norm(dsdt, ord=2, dim=1).mean()) 
                    print("RHS", torch.linalg.norm(RHS, ord=2, dim=1).mean())  
    
                    res = res ** self.m
                if self.wgt == 'convention':
                    res = diffusion * res
                elif self.wgt == 'll':
                    g_2 = diffusion**2
                    res = g_2 * res
                elif self.wgt == 'constant':
                    res = res
                else:
                    print("Undefined time weighting...")                
                    
                if self.reduction == 'mean':
                    res = torch.mean(res) if self.train else np.mean(res)
                elif self.reduction == 'sum':
                    res = res.sum() if self.train else np.sum(res)
                elif self.reduction == 'batchwise':
                    res = res
                elif self.reduction == 'pointwise':
                    res = error                    
                else:
                    print("Undefined reduction method...")
                if self.normalize:
                    res = res/(D ** self.m)   
            
            elif self.scalar_fp in ("both"):
                g_pow = diffusion**2 
                f_dot_s = torch.einsum('bs,bs->b', x, s)
                RHS = (g_pow/2) * ( div_s + s_l22 + f_dot_s )  
                RHS = RHS if self.train else RHS.cpu().detach().numpy()
                scalar_res = torch.linalg.norm(RHS, ord=2) if self.train else np.linalg.norm(RHS, ord=2)       

                print("scalar_fp", self.scalar_fp)
                g_pow = (diffusion[:, None])**2           
                f_dot_s = torch.einsum('bs,bs->b', x, s)                             
                D = x.shape[-1]
                if self.list_dim == []:        
                    self.list_dim = range(D)
                RHS = (g_pow/2) * diff.gradient( div_s + s_l22 + f_dot_s, x)       
            
                "Computing LHS"
                dsdt = diff.t_finite_diff(self.flat_model, x, t)
                error = (dsdt - RHS)
                res = torch.linalg.norm(error, ord=2, dim=1)  
                res = res ** self.m
                if self.wgt == 'convention':
                    res = diffusion * res
                elif self.wgt == 'll':
                    g_2 = diffusion**2
                    res = g_2 * res
                elif self.wgt == 'constant':
                    res = res
                else:
                    print("Undefined time weighting...")
            
                if self.reduction == 'mean':
                    res = torch.mean(res) 
                elif self.reduction == 'sum':
                    res = res.sum() 
                elif self.reduction == 'batchwise':
                    res = res
                elif self.reduction == 'pointwise':
                    res = error 
                else:
                    print("Undefined reduction method...")
                    
                if self.normalize:
                    res = res/(D ** self.m)                  
                vec_res = res                 

            x.requires_grad = False
            t.requires_grad = False
            
            
            if self.scalar_fp in ("True", "False"):
                print("FP type {}: {}".format(self.scalar_fp, res.detach().cpu().numpy()), flush=True)
                return res
            elif self.scalar_fp in ("both"):
                return scalar_res, vec_res            
    
      return score_FP()


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.name = 'vesde'
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):   
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G

  def PDE(self, score_fn, 
          wgt='constant',
          normalize=True, 
          reduction='mean', 
          list_dim=[], 
          time_est='approx', 
          div_est='approx', 
          train=True, 
          scalar_fp="both",
          m=1):
      
      sde_fn = self.sde
      flat_model = Merge(score_fn)
      class score_FP(self.__class__):
          def __init__(self):
              self.flat_model = flat_model
              self.normalize = normalize
              self.reduction = reduction 
              self.list_dim = list_dim
              self.time_est = time_est
              self.div_est = div_est       
              self.train = train
              self.scalar_fp = scalar_fp
              self.wgt = wgt
              self.m = m
              
          def fp(self, x, t):
            x = x.reshape((x.shape[0], -1))
            x.requires_grad = True
            t.requires_grad = True
            s = self.flat_model(x, t)                 
            if self.div_est == 'exact':
                div_s = diff.batch_div(self.flat_model, x, t)
            elif self.div_est == 'approx':    
                div_s = diff.hutch_div(self.flat_model, x, t)
            s_l22 = torch.linalg.norm(s, 2, dim=1, keepdim=False)**2    
            
            # "Computing RHS"
            _, diffusion = sde_fn(x, t)
            
            if self.scalar_fp in ("True"):
                g_pow = diffusion**2 
                RHS = (g_pow/2) * ( div_s + s_l22 )   
                RHS = RHS if self.train else RHS.cpu().detach().numpy()
                res = torch.linalg.norm(RHS, ord=2) if self.train \
                    else np.linalg.norm(RHS, ord=2) 
            elif self.scalar_fp in ("False"):            
                g_pow = (diffusion[:, None])**2 
                D = x.shape[-1]
                if self.list_dim == []:        
                    self.list_dim = range(D)
                RHS = (g_pow/2) * diff.gradient( div_s + s_l22, x) 
            
                "Computing LHS"
                if self.time_est == 'exact':
                    res = torch.zeros_like(t)  
                    for j in self.list_dim:
                        dsdt = diff.partial_t_j(self.flat_model, x, t, j)  
                        residue = torch.clip((dsdt - RHS[0:,j])**2, max=1.0e+30)
                        res += residue 
                    res = torch.sqrt(res) 
                elif self.time_est == 'approx':
                    dsdt = diff.t_finite_diff(self.flat_model, x, t)
                    error = dsdt - RHS
                    res = torch.linalg.norm(error, ord=2, dim=1) 
                    res = res ** self.m
                if self.wgt == 'convention':
                    res = diffusion * res
                elif self.wgt == 'll':
                    g_2 = diffusion**2
                    res = g_2 * res
                elif self.wgt == 'constant':
                    res = res
                else:
                    print("Undefined time weighting...")
    
                if self.reduction == 'mean':
                    res = torch.mean(res) 
                elif self.reduction == 'sum':
                    res = res.sum()                
                else:
                    print("Undefined reduction method...")

                if self.normalize:
                    res = res/(D ** self.m)  
                
            
            elif self.scalar_fp in ("both"):
                g_pow = diffusion**2 
                RHS = (g_pow/2) * ( div_s + s_l22 )   
                RHS = RHS if self.train else RHS.cpu().detach().numpy()
                scalar_res = torch.linalg.norm(RHS, ord=2) if self.train else np.linalg.norm(RHS, ord=2)       
                  
                g_pow = (diffusion[:, None])**2 
                D = x.shape[-1]
                if self.list_dim == []:        
                    self.list_dim = range(D)
                RHS = (g_pow/2) * diff.gradient( div_s + s_l22, x)   
            
                "Computing LHS"
                dsdt = diff.t_finite_diff(self.flat_model, x, t)
                res = torch.linalg.norm((dsdt - RHS), ord=2, dim=1)  
                res = res ** self.q
                if self.wgt == 'convention':
                    res = diffusion * res
                elif self.wgt == 'll':
                    g_2 = diffusion**2
                    res = g_2 * res
                elif self.wgt == 'constant':
                    res = res
                else:
                    print("Undefined time weighting...")
       
                if self.reduction == 'mean':
                    res = torch.mean(res) 
                elif self.reduction == 'sum':
                    res = res.sum() 
                else:
                    print("Undefined reduction method...")
                if self.normalize:
                    res = res/(D ** self.m)                  
                vec_res = res 
                        
                    
            x.requires_grad = False
            t.requires_grad = False
            
            
            if self.scalar_fp in ("True", "False"):
                return res
            elif self.scalar_fp in ("both"):
                return scalar_res, vec_res
    
      return score_FP()
  


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.