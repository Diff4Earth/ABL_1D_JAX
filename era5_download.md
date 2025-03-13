# ERA5 source

The file `ERA5_bhl.nc` used in the scripts one timestep of hourly ERA5 data for planetary boundary layer height and land-sea mask. The file was taken from the Copernicus Climate Data Store. Python code to download the file using the CDS API is below.

See <href>https://cds.climate.copernicus.eu/how-to-api</href> for instructions [last accessed 13-03-2025].

```
import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "boundary_layer_height",
        "land_sea_mask"
    ],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
```
