import xarray as xr
import jax
import jax.numpy as jnp
jax.config.update('jax_platforms', 'cpu')


def smooth_pblh(pvar2d, msk):
  ''' 2D Hanning filter on atmospheric PBL height'''
  # Hanning filter
  smth_a = 1 / 8
  smth_b = 1 / 4
  zumsk = jnp.zeros_like(msk)
  zumsk = zumsk.at[:-1].set(msk[:-1] * msk[1:])
  zdX = jnp.zeros_like(pvar2d)
  zdX = zdX.at[:-1].set((pvar2d[1:] - pvar2d[:-1]) * zumsk[:-1])
  
  zvmsk = jnp.zeros_like(msk)
  zvmsk = zvmsk.at[:, :-1].set(msk[:, :-1] * msk[:, 1:])
  zdY = jnp.zeros_like(pvar2d)
  zdY = zdY.at[:, :-1].set((pvar2d[:, 1:] - pvar2d[:, :-1]) * zvmsk[:, :-1])
    
  zfY = jnp.zeros_like(zdY)
  print(jnp.shape(zdY))
  print(jnp.shape(zdX))
  
  zFY = jnp.zeros_like(pvar2d)
  zFY = zFY.at[1:-1, :-1].set(
        zdY[1:-1, :-1] + 
        smth_a * ((zdX[1:-1, 1:] - zdX[:-2, 1:]) - 
                  (zdX[1:-1, :-1] - zdX[:-2, :-1]))
    )
  
  zFX = jnp.zeros_like(pvar2d)
  zFX = zFX.at[:-1, 1:-1].set(
        zdX[:-1, 1:-1] + 
        smth_a * ((zdY[1:, 1:-1] - zdY[1:, :-2]) - 
                  (zdY[:-1, 1:-1] - zdY[:-1, :-2]))
    )
  
  result = pvar2d.copy()
  result = result.at[1:-1, 1:-1].add(
        msk[1:-1, 1:-1] * smth_b * (
            (zFX[1:-1, 1:-1] - zFX[:-2, 1:-1]) +
            (zFY[1:-1, 1:-1] - zFY[1:-1, :-2])
        )
    )
    
  return result
    

def ABL_1D_computation(filename, var_name, mask_name, output_name):
    ds = xr.open_dataset(filename)
    pvar2d = jnp.array(ds[var_name][0])

    msk = jnp.array(ds[mask_name][0])
    msk_fix = jnp.zeros_like(msk)
    msk_fix = msk_fix.at[msk==0].set(1)

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
    output_name = 'jax_hanning_cpu_loopy.nc'

    ABL_1D_computation(filename, var_name, mask_name, output_name)
