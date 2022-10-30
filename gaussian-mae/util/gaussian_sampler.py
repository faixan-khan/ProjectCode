# ==============================================================================
# This file includes helper methods to use the gaussian sampler
# ==============================================================================
# This source  code is licensed under the license found in the  LICENSE  file in 
# the root directory of this source tree.
# ------------------------------------------------------------------------------
import torch
from  torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import torch.distributions as D

# ==============================================================================
# Save time and memory with singleton
# ==============================================================================
class _Singleton_Points:
    _instance = None
    
    def __init__(self, N, patches_h, patches_w, device):
        i_rng = torch.arange(patches_h, device=device)
        j_rng = torch.arange(patches_w, device=device)
        points = torch.cartesian_prod(i_rng, j_rng).reshape(patches_h,
                                                            patches_w,-1)
        points = points.type(torch.float32)
        self._instance = points.unsqueeze(dim=-2).expand(-1,-1,N,-1)


def Points(N, patches_h, patches_w, device):
    if _Singleton_Points._instance is None:
        _Singleton_Points._instance = _Singleton_Points(N, patches_h, patches_w,
                                                                         device)
    return _Singleton_Points._instance._instance

def get_mean_dim(m_dim, win, dim_size):
    """returns the mean point for a particular  dimension for our gaussian gene-
    ration to make sure the distribution remains within the image"""
    min_m_dim = win//2
    max_m_dim = dim_size-min_m_dim
    m_dim = torch.clip(m_dim, min_m_dim, max_m_dim)
    return m_dim

def generate_gaussian_2D_1D(h: int, w: int, m_h: torch.Tensor, m_w: torch.Tensor,
                            win: int, device):
    """generates a 2D gaussian distribution in a matrix and 
    
    - h: 2D grid height
    - w: 2D grid width
    - m_h: the mean points with respect to the height axis
    - m_w: the mean points with respect to the width axis
    - win: window size
    
    ----------------------------------------------------------------------------
    NOTE: the returned probability distribution is not normalized
    """
    N = m_h.shape[0]
    # we scale the range of values as the number of points is  different to have
    # an indentity covariance matrix
    min_dim = min(h, w)
    h_ratio = h/min_dim
    w_ratio = w/min_dim

    # define the range of values accroding to the mean h and w
    S_H = -2*m_h/h
    E_H = 2-2*m_h/h
    S_W = -2*m_w/w
    E_W = 2-2*m_w/w


    # ==========================================================================
    # TODO: modify the code to generate N 2D gaussians
    dist = torch.empty(size=(N, h*w), device= device)
    i = -1
    for s_h, e_h, s_w, e_w in zip(S_H, E_H, S_W, E_W):
        i += 1
        s_h, e_h, s_w, e_w = s_h.item(), e_h.item(), s_w.item(), e_w.item()
        # now, generate the grid
        x, y = torch.meshgrid(torch.linspace(s_h*h_ratio,e_h*h_ratio,h),
                              torch.linspace(s_w*w_ratio,e_w*w_ratio,w),
                              indexing='ij')
        d = torch.sqrt(x*x+y*y)
        # define the sigma according to the window size
        # ----------------------------------------------------------------------
        # we divide win/min_dim to be 1 when we sample from the whole  image, so
        # sigma would span the whole window in such case. We further divide over
        # 3, so~%100 of the distribution will be within the range of -1,1 (which
        # is our window).
        # ----------------------------------------------------------------------
        sigma = win/min_dim/3
        # now, generate the 2D gaussian
        g = torch.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
        # return the normalized version of it
        dist[i,:] = torch.reshape(g, (-1,))
    return dist #means_mask

def sample_indices_1D(p_mask, num_samples):
    """The method samples patches from a 1D mask and return their indices"""
    idx = p_mask.multinomial(num_samples=num_samples, replacement=False)
    # return the indices
    return idx

def generate_binary_mask_1D(mask_shape, idx):
    """The method generates a binary mask according to the samples
    
    - mask_shape: the shape of the generated mask
    - idx: a tensor the includes the indices of the selected patches
    """
    b_mask = torch.zeros(mask_shape, dtype=torch.uint8)
    b_mask[((torch.arange(b_mask.shape[0]).unsqueeze(1).repeat(1,idx.shape[1])), idx)] = 1
    assert (b_mask.sum(dim=1) == idx.shape[1]).prod() == 1
    return b_mask

def mean_generation(window_size, N, n_gaussian, max_dims, device):
    """the method randomly picks mean points for our gaussian generation

    - window_size: the size of the generated gaussians
    - N: number of patch grids
    - n_gaussian: the number of gaussians per a single image
    - max_dims: a tuple of the size of the 2D grid
    - device: the device of the means
    """
    patches_h, patches_w = max_dims
    # all the windows are of the same shape
    min_h = window_size//2
    max_h = patches_h - min_h + 1

    min_w = window_size//2
    max_w = patches_w - min_w + 1
    
    # --------------------------------------------------------------------------
    # generate different permutations
    # --------------------------------------------------------------------------
    # We generate the means  in such way to make sure the means do not duplicate
    # for a single image
    # --------------------------------------------------------------------------
    mean_indx = torch.randperm(patches_h*patches_w, device= device).expand((N,-1))
    # p_h = (torch.randperm(max_h-min_h)+min_h).expand((N,-1))
    # p_w = (torch.randperm(max_w-min_w)+min_w).expand((N,-1))

    noise = torch.rand(mean_indx.shape, device= device)

    # noise_h = torch.rand(p_h.shape)
    # noise_w = torch.rand(p_w.shape)

    ids_shuffle = torch.argsort(noise, dim=1)
    # ids_h_shuffle = torch.argsort(noise_h, dim=1)
    # ids_w_shuffle = torch.argsort(noise_w, dim=1)

    mean_indices = torch.gather(mean_indx, dim=1, index=ids_shuffle)[:,:n_gaussian]

    m_h = torch.div(mean_indices,patches_w,rounding_mode='floor')
    m_w = torch.remainder(mean_indices,patches_w)
    # m_h = torch.gather(p_h, dim=1, index=ids_h_shuffle)[:,:n_gaussian]
    # m_w = torch.gather(p_w, dim=1, index=ids_w_shuffle)[:,:n_gaussian]

    # ==========================================================================
    # make sure the shapes are correct
    # ==========================================================================
    # print(m_h.shape, N, n_gaussian, 'what ')
    assert m_h.shape == (N,n_gaussian)
    assert m_w.shape == (N,n_gaussian)

    return m_h, m_w

# ==============================================================================
# Here, we will generate multiple  independent gaussian distributions  and merge
# them to have a mixture of gaussians
def mixture_gaussians_1D(N, n_gaussian, window_size, patches_h, patches_w,
                         device, enable_out=True, mean_mask = True):
    # Randomly set the mean
    # ==========================================================================
    # Make sure the window of the gaussian fits into the grid
    assert window_size < patches_h and window_size < patches_w
    m_h , m_w = mean_generation(window_size, N, n_gaussian, 
                                (patches_h,patches_w), device)
    # to make sure we do not go out of the image while generating
    if not enable_out:
        m_h = get_mean_dim(m_h, window_size, patches_h)
        m_w = get_mean_dim(m_w, window_size, patches_w)
    m = torch.stack((m_h,m_w),dim=-1)
    # ==========================================================================
    # Now generate the gaussians
    means_mask = None
    if mean_mask:
        means_mask = torch.zeros(N, patches_h*patches_w, dtype=torch.uint8,
                                                                 device=device)
        means_indx = m_h*patches_w + m_w
        means_mask[((torch.arange(means_mask.shape[0]).unsqueeze(1).expand(-1,
                                        means_indx.shape[1])), means_indx)] = 1
    mix = D.Categorical(torch.ones(N, n_gaussian, device= device))
    sgma = window_size/3
    cov = torch.eye(2, device= device).unsqueeze(0).unsqueeze(1).expand(\
                                                      (N,n_gaussian,-1,-1))*sgma
    dist_1 = MultivariateNormal(loc=m, covariance_matrix=cov)
    dist = MixtureSameFamily(mix, dist_1)
    points = Points(N, patches_h, patches_w, device)
    p_mask = torch.exp(dist.log_prob(points)).permute(2,0,1).reshape(N,-1)
    if mean_mask:
        p_mask = (1-means_mask)*(p_mask+1e-6)
        # We have N*n_gaussian number of gaussians
        assert means_mask.nonzero().shape[0]==N*n_gaussian
        # Make sure that all the means have 0 probability
        assert (((1-means_mask)*p_mask)==p_mask).prod().item() == 1
    # Normalize your probability distribution
    p_mask = p_mask / p_mask.sum(dim=1).reshape(-1,1)
    return p_mask, means_mask

# ==============================================================================
# Here, we will generate multiple  independent gaussian distributions  and merge
# them to have a mixture of gaussians
def _mixture_gaussians_1D(N, n_gaussian, window_size, patches_h, patches_w,
                         device, enable_out=True, mean_mask = True):
    # Randomly set the mean
    # ==========================================================================
    # Make sure the window of the gaussian fits into the grid
    assert window_size < patches_h and window_size < patches_w
    m_h , m_w = mean_generation(window_size, N, n_gaussian, 
                                (patches_h,patches_w), device)
    # to make sure we do not go out of the image while generating
    if not enable_out:
        m_h = get_mean_dim(m_h, window_size, patches_h)
        m_w = get_mean_dim(m_w, window_size, patches_w)
    # ==========================================================================
    # Now generate the gaussians
    p_mask = torch.zeros(N, patches_h*patches_w, device= device)
    means_mask = None
    if mean_mask:
        means_mask = torch.zeros(p_mask.shape, dtype=torch.uint8, device= device)
        means_indx = m_h*patches_w + m_w
        means_mask[((torch.arange(means_mask.shape[0]).unsqueeze(1).repeat(1,
                                        means_indx.shape[1])), means_indx)] = 1
    for gaussian_indx in range(n_gaussian):
        p = generate_gaussian_2D_1D(patches_h, patches_w,
                                    m_h[:,gaussian_indx], 
                                    m_w[:,gaussian_indx],
                                    window_size, device)
        p_mask += p
    if mean_mask:
        
        p_mask = (1-means_mask)*p_mask
        # We have N*n_gaussian number of gaussians
        assert means_mask.nonzero().shape[0]==N*n_gaussian
        # Make sure that all the means have 0 probability
        assert (((1-means_mask)*p_mask)==p_mask).prod().item() == 1
    # Normalize your probability distribution
    p_mask = p_mask / p_mask.sum(dim=1).reshape(-1,1)
    return p_mask, means_mask


# ==============================================================================
# Sample excution
# ==============================================================================
# N = 16
# new_h = 14
# new_w = 14
# depth = 786
# rand_data = torch.rand(N, new_h, new_w, depth)
# rand_data.shape

# n_gaussian = 6
# window_sizes = torch.ones(n_gaussian, dtype=torch.uint8)*9
# p_mask, means_mask = mixture_gaussians_1D(N,n_gaussian, window_sizes, new_h,
#                                           new_w)
# means_idx = means_mask.nonzero()[:,1].reshape(N,-1)
# num_samples = 18
# idx = sample_indices_1D(p_mask, num_samples)
# sample_mask = generate_binary_mask_1D(p_mask.shape, idx)
# assert (sample_mask*means_mask).sum() == 0
# ------------------------------------------------------------------------------
# description
# ------------------------------------------------------------------------------
# means_mask: binary 2D mask having 1 for the selected patches for anchor tokens
# sample_mask: binary 2D mask having 1 for sampled patches according to the gau-
#           ssian [for reconstruction]
# idx: the indices of the sampled patches
# means_idx: the indices of the means; they are sorted according to position
# ==============================================================================