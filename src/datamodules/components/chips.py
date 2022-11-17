#from osgeo import gdal, osr, ogr # Python bindings for GDAL
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import json
from rlxutils import subplots
from .utils import pimshow

class Chipset:
    
    def __init__(self, chipset_folder=None, data=None, metadata=None):

        self.folder = chipset_folder
        self.chip_fnames = os.listdir(chipset_folder)
                
        chip_ids_npz = [i.split(".")[0] for i in self.chip_fnames if i.endswith(".npz")]
        chip_ids_pkl = [i.split(".")[0] for i in self.chip_fnames if i.endswith("metadata.pkl")]
        # keep the chips with both metadata and data
        self.chip_ids = [i for i in chip_ids_npz if i in chip_ids_pkl]
     
    def get_chip(self, chip_id):
        if not chip_id in self.chip_ids:
            raise ValueError(f"{chip_id} does not exist")
            
        return Chip(self.folder, chip_id)
    
    def random_chip(self):
        chip_id = self.chip_ids[np.random.randint(len(self.chip_ids))]
        return Chip(self.folder, chip_id)

    def chips(self):
        for chip_id in self.chip_ids:
            yield Chip(self.folder, chip_id)        
        
class Chip:
    def __init__(self, chipset_folder, chip_id, data=None, metadata=None):
        self.chipset_folder = chipset_folder
        self.chip_id = chip_id

        assert (data is None and metadata is None) or (data is not None and metadata is not None), "'data' and 'metadata' must be both set or unset"

        if data is None:
            with np.load(f"{self.chipset_folder}/{self.chip_id}.npz") as my_file:
                self.data = my_file['arr_0']
                my_file.close()
            with open(f"{self.chipset_folder}/{self.chip_id}.metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
                f.close()
        else:
            self.data = data
            self.metadata = metadata

    def clone(self):
        return self.__class__(self.chipset_folder, self.chip_id, self.data.copy(), self.metadata.copy())
            
    def get_varnames(self, exceptions=[]):
        r = self.get_timestep_varnames() + self.get_timepair_varnames() + self.get_static_varnames()
        r = [i for i in r if not i in exceptions]
        return r

    def get_time_varnames(self):
        return self.get_timestep_varnames() + self.get_timepair_varnames()

    def get_timestep_varnames(self):
        return list(np.unique([i.split("::")[1] for i in self.metadata['variables'] if i.startswith("TS::")]))            

    def get_timesteps(self):
        return list(np.unique([i.split("::")[2] for i in self.metadata['variables'] if i.startswith("TS::")]))            
    
    def get_timepair_varnames(self):
        return list(np.unique([i.split("::")[1] for i in self.metadata['variables'] if i.startswith("TP::")]))            

    def get_timepairs(self):
        return list(np.unique([i.split("::")[2] for i in self.metadata['variables'] if i.startswith("TP::")]))            

    def get_static_varnames(self):
        return list(np.unique([i.split("::")[1] for i in self.metadata['variables'] if i.startswith("ST::")]))            
        
    def apply_with_other(self, other, func):
        """
        applies a function element-wise to the data of two chips
        which must have exactly the same vars.
        
        the function must have signature func(x,y) with x,y and the return value
        must all be arrays of the same size
        """
        thisvars = self.get_varnames()
        othervars = other.get_varnames()

        if self.get_varnames() != other.get_varnames():
            raise ValueError("chips must have the same number of variables")
        
        r = self.clone()
        
        r.data = func(self.data, other.data)
        
        for name in self.metadata['variables']:
            self.metadata['variables']
            
        return r
    
    def apply(self, var_name, func, args):
        """
        applies a function element-wise to the data of two chips
        which must have exactly the same vars.
        
        the function must have signature func(x,y) with x,y and the return value
        must all be arrays of the same size
        """
        try:
            assert(var_name in self.get_varnames())
        except:
            raise ValueError(f'{var_name} not in chip')
            
        r                = self.sel([var_name])

        r.data, name_tag = func(np.nan_to_num(r.data), **args) 
        
        if name_tag:
            r.metadata['variables']= [name.replace(var_name,var_name+'_'+name_tag) for name in r.metadata['variables']]

        return r

    def apply_across_time(self, func):
        """
        applies a function across the time dimension rendering time variables into static
        returns a new chip
        """

        tsvars = self.get_timestep_varnames()
        tpvars = self.get_timepair_varnames()
        selected_vars = tsvars + tpvars
        new_data = np.vstack([np.apply_over_axes(func, self.sel(v).data,0) for v in selected_vars])
        new_metadata = self.metadata.copy()
        # all variables now become static
        new_metadata['variables'] = [f"ST::{i}__{func.__name__}" for i in tsvars] + [f"ST::{i}__{func.__name__}" for i in tpvars]
        new_chip_id = self.chip_id + f"_{np.random.randint(1000000):07d}"
        new_metadata['chip_id'] = new_chip_id
        return self.__class__(self.chipset_folder, new_chip_id, new_data, new_metadata)
    
    def diff_channels(self, tag1, tag2):
        var_names = self.get_varnames()
        
        new_metadata = self.metadata.copy()

        new_metadata['variables'] =[]
        new_data = []
        new_vars = []
        for varname in var_names:
            if tag1 in varname:
                v, idx = self.get_array_idxs(varnames=[varname.replace(tag1, tag2), varname])
                new_data.append(np.expand_dims(np.squeeze(self.data[idx[0]]-self.data[idx[1]]),axis=0))

                if tag1 in v[0]:
                    v = v[0].replace(tag1,'diff')
                else:
                    v = v[0].replace(tag2,'diff')

                new_metadata['variables'].append(v)
            elif tag2 in varname:
                pass
            else:
                v, idx = self.get_array_idxs(varnames=[varname])
                new_data.append(np.expand_dims(np.squeeze(self.data[idx]),0))
                new_metadata['variables'].append(v[0])
        
        new_data = np.vstack(new_data)
        new_chip_id = self.chip_id + f"_{np.random.randint(1000000):07d}"
        new_metadata['chip_id'] = new_chip_id

        return self.__class__(self.chipset_folder, new_chip_id, new_data, new_metadata)
            
    def get_array_idxs(self, varnames=None, start_date=None, end_date=None):
        if varnames is None:
            varnames = self.get_varnames()
        elif not type(varnames)==list:
            varnames = [varnames]
        vspecs = self.metadata['variables']
        selected_idxs = []
        selected_vars = []
        for i in range(len(vspecs)):
            vspec = vspecs[i]
            if vspec.startswith('TS::'):
                _,vname,vdate = vspec.split("::")
                if vname in varnames\
                   and (start_date is None or start_date<=vdate)\
                   and (end_date is None or end_date>=vdate):
                    selected_idxs.append(i)
                    selected_vars.append(vspec)
            elif vspec.startswith('TP::'):
                _,vname,vdate = vspec.split("::")
                vdate1, vdate2 = vdate.split("_")
                if vname in varnames\
                   and (start_date is None or (start_date<=vdate1 and start_date<=vdate2))\
                   and (end_date is None or (end_date>=vdate2 and end_date>=vdate2)):
                    selected_idxs.append(i)
                    selected_vars.append(vspec)
            elif vspec.startswith('ST::'):
                _, vname = vspec.split("::")
                if vname in varnames:
                    selected_idxs.append(i)
                    selected_vars.append(vspec)
        return selected_vars, selected_idxs
    
    def get_array(self, varnames=None, start_date=None, end_date=None):
        _, selected_idxs = self.get_array_idxs(varnames, start_date, end_date)
        return self.data[selected_idxs]    

    def plot(self, overlay=None, log=False, **kwargs):
        if not 'n_cols' in kwargs:
            kwargs['n_cols'] = 5
        if not 'usizex' in kwargs:
            kwargs['usizex'] = 4
        if not 'usizey' in kwargs:
            kwargs['usizey'] = 3

        for ax,i in subplots(len(self.data), **kwargs):
            if log and np.nanmin(self.data[i])>=0:
                x = np.log10(self.data[i]+1e-4)
            else:
                x = self.data[i]
            pimshow(x)
            if np.sum(overlay):
                pimshow(np.squeeze(overlay), alpha=0.2)
            plt.colorbar()
            varname = self.metadata['variables'][i].split("::")
            if len(varname)==2:
                tit = varname[1]
            else:
                tit = f"{varname[1]}\n{varname[2]}"

            plt.title(tit)

        plt.tight_layout()

    def sel(self, varnames=None, start_date=None, end_date=None):
        selected_vars, selected_idxs = self.get_array_idxs(varnames, start_date, end_date)
        new_data = self.data[selected_idxs]
        new_metadata = self.metadata.copy()
        new_metadata['variables'] = selected_vars
        new_chip_id = self.chip_id + f"_{np.random.randint(1000000):07d}"
        new_metadata['chip_id'] = new_chip_id
        return self.__class__(self.chipset_folder, new_chip_id, new_data, new_metadata)

    def save_as_geotif(self, dest_folder):
        from osgeo import gdal, osr, ogr
        
        def getGeoTransform(extent, nlines, ncols):
            resx = (extent[2] - extent[0]) / ncols
            resy = (extent[3] - extent[1]) / nlines
            return [extent[0], resx, 0, extent[3] , 0, -resy]

        # Define the data extent (min. lon, min. lat, max. lon, max. lat)
        extent = list(self.metadata['bounds'].values()) # South America

        # Export the test array to GeoTIFF ================================================

        # Get GDAL driver GeoTiff
        driver = gdal.GetDriverByName('GTiff')

        data = self.data
        
        # Get dimensions
        nlines = data.shape[1]
        ncols = data.shape[2]
        nbands = len(data)
        data_type = gdal.GDT_Float32 # gdal.GDT_Float32

        # Create a temp grid
        #options = ['COMPRESS=JPEG', 'JPEG_QUALITY=80', 'TILED=YES']
        grid_data = driver.Create('grid_data', ncols, nlines, nbands, data_type)#, options)

        # Write data for each bands
        for i in range(len(data)):
            grid_data.GetRasterBand(i+1).WriteArray(self.data[i])

        # Lat/Lon WSG84 Spatial Reference System
        import os
        import sys
        proj_lib = "/".join(sys.executable.split("/")[:-2]+['share', 'proj'])

        os.environ['PROJ_LIB']=proj_lib
        srs = osr.SpatialReference()
        #srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        srs.ImportFromEPSG(int(self.metadata['crs'].split(':')[1]))

        # Setup projection and geo-transform
        grid_data.SetProjection(srs.ExportToWkt())
        grid_data.SetGeoTransform(getGeoTransform(extent, nlines, ncols))

        # Save the file
        file_name = f'{dest_folder}/{self.chip_id}.tif'
        print(f'saved {file_name} with {nbands} bands')
        driver.CreateCopy(file_name, grid_data, 0)  

        # Close the file
        driver = None
        grid_data = None

        # Delete the temp grid
        import os                
        os.remove('grid_data')
        #===========================        
        
    @staticmethod
    def concat_static(chip_list,sufixes=None):
        """
        concats two chips containing only static variables
        """

        if sufixes is None:
            sufixes = [""] * len(chip_list)

        assert len(chip_list)==len(sufixes), f"you have {len(chip_list)} chips but {len(sufixes)} sufixes"
        
        c = chip_list[0]
        p = sufixes[0]
            
        # all variables must be static
        assert len(c.get_time_varnames())==0, "chips can only contain static variables"
                
        r = c.clone()
        r.chip_id += "_concat"
        r.metadata['variables'] = [f"{i}{p}" for i in r.metadata['variables']]
        
        for i in range(1,len(chip_list)):
            c = chip_list[i]
            p = sufixes[i]
            
            if c.metadata['bounds'] != r.metadata['bounds']:
                raise ValueError("all chips must have the same bounds")
                
            if c.metadata['crs'] != r.metadata['crs']:
                raise ValueError("all chips must have the same crs")
        
            r.data = np.vstack([r.data, c.data])
            r.metadata['variables'] += [f"{i}{p}" for i in c.metadata['variables']]
        
        if len(np.unique( r.metadata['variables'] )) != sum([len(i.metadata['variables'] ) for i in chip_list] ):
            raise ValueError("there were overlapping variable names in the chips. use 'sufixes'")

        return r