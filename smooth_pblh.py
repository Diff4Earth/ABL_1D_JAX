# This is the python only version of the Hanning filter function

import xarray as xr
#import jax
#import jax.numpy as jnp
#jax.config.update('jax_platforms', 'cpu')
import numpy as np

def smooth_pblh(pvar2d, msk):
  ''' 2D Hanning filter on atmospheric PBL height'''
  # Hanning filter
  smth_a = 1 / 8
  smth_b = 1 / 4
  zumsk = jnp.zeros_like(msk)
  zumsk = zumsk.at[:-1].set(msk[:-1] * msk[1:])
  zdX = np.zeros_like(pvar2d)
  zdX[:-1] = (pvar2d[1:] - pvar2d[:-1]) * zumsk[:-1]
  
  zvmsk = np.zeros_like(msk)
  zvmsk[:, :-1] = msk[:, :-1] * msk[:, 1:]
  zdY = np.zeros_like(pvar2d)
  zdY[:, :-1] = (pvar2d[:, 1:] - pvar2d[:, :-1]) * zvmsk[:, :-1]
    
  zfY = np.zeros_like(zdY)
  print(np.shape(zdY))
  print(np.shape(zdX))
  
  zFY = np.zeros_like(pvar2d)
  zFY[1:-1, :-1] = 
        zdY[1:-1, :-1] + 
        smth_a * ((zdX[1:-1, 1:] - zdX[:-2, 1:]) - 
                  (zdX[1:-1, :-1] - zdX[:-2, :-1]))
  
  zFX = np.zeros_like(pvar2d)
  zFX[:-1, 1:-1] = 
        zdX[:-1, 1:-1] + 
        smth_a * ((zdY[1:, 1:-1] - zdY[1:, :-2]) - 
                  (zdY[:-1, 1:-1] - zdY[:-1, :-2]))
  
  result = pvar2d.copy()
  result[1:-1, 1:-1] = result[1:-1, 1:-1] + (
        msk[1:-1, 1:-1] * smth_b * (
            (zFX[1:-1, 1:-1] - zFX[:-2, 1:-1]) +
            (zFY[1:-1, 1:-1] - zFY[1:-1, :-2])
        )
    )
    
  return result
    

def ABL_1D_computation(filename, var_name, mask_name, output_name):
    ds = xr.open_dataset(filename)
    pvar2d = np.array(ds[var_name][0])

    msk = np.array(ds[mask_name][0])
    msk_fix = np.zeros_like(msk)
    msk_fix[msk==0] = 1

    result = smooth_pblh(pvar2d, msk_fix)
    
    data_jax = xr.Dataset(
    data_vars=dict(blh_smoox=(["latitude", "longitude"],result), msk=(["latitude", "longitude"],msk_fix)),
    coords={
        "latitude": ds.latitude.values,
        "longitude": ds.longitude.values
    }
)
    data_jax.to_netcdf(output_name)

if __name__ == '__main__':
    #Define parameters
    filename = 'ERA5_bhl.nc'
    var_name = 'blh'
    mask_name = 'lsm'
    output_name = 'jax_hanning_py.nc'

    ABL_1D_computation(filename, var_name, mask_name, output_name)
