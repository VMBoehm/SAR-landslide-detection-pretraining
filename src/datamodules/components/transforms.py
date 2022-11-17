import numpy as np
import scipy
from scipy.spatial import KDTree as KDTree
from src.utils.loss_kernels import gaussian_kernel
from scipy.signal import convolve2d

class KDE_smoothing():
    
    def __init__(self, sigma, norm=False):
        
        self.sigma      = sigma
        self.norm       = norm
        self.kernel     = gaussian_kernel(size=5*sigma,sigma=sigma)
        self.kernel     = self.kernel/self.kernel.sum()

    def __call__(self,data):
        shape = data.shape
        data = convolve2d(np.squeeze(data),self.kernel, mode='same')
        if self.norm:
            data/=np.sum(data)
        return np.reshape(data, shape)


class Ignore_multiples():
    
    def __init__(self, threshold=0., new_val = 1.0):
        self.threshold = threshold
        self.new_val   = new_val
    
    def __call__(self,data):
        data = (data>self.threshold).astype(np.float32)*self.new_val
        return data


class Log_transform():
    
    def __init__(self, stats_dict, offset=1e-4, log=np.log10, name_tag='log'):
        """

        """
        self.offset     = offset
        self.log        = log
        self.stats_dict = stats_dict
        self.name_tag   = name_tag


    def __call__(self,data, channel):
        data = np.nan_to_num(data)
        if min(self.stats_dict[f'min_{channel}'],np.min(data))<=0.:
            data+=abs(self.stats_dict[f'min_{channel}'])
            data+=self.offset
        data = self.log(data)
        
        return data, self.name_tag
    
#TODO: adapt
class Standardize():
    
    def __init__(self, stats_dict, name_tag='stded'):
        self.stats_dict     = stats_dict
        self.name_tag       = name_tag
    
    # VB uses stas_dict for all channels together (ig broadcasting is correct)
    def __call__(self,data, channel):
        
        if 'log' in channel:
            mean = self.stats_dict[f'log_mean_{channel}']
            std  = self.stats_dict[f'log_std_{channel}']
        else:
            mean = self.stats_dict[f'mean_{channel}']
            std  = self.stats_dict[f'std_{channel}']            

        data = np.nan_to_num(data)
        data = (data-mean)/std

        
        return data, self.name_tag
    
    
class Identity_transform():
    
    def __init__(self):
        pass
    
    def __call__(self,data):
        return data
    
    
    
