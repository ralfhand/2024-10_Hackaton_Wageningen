#!/usr/bin/env python3

# This script preprocesses the ICON data to create input files for the firedanger tool written by Daniel Steinfeld. 
# The preprocessing includes the selection of subregions, the time period, and variables and the interpolation from Healpix format to a lon-lat grid. 
# Customization can be done in the customization section below.
# 
# In the current configuration the script computes wind_speed from daily mean u- and v-wind (uas and vas from P1D), daily max temperature (max of tas from PT3H), 
# and extracts daily mean precipitation rates (pr from P1D). Minimum relative humidity was computed from daily mean specific humidity (qv2m from P1D) and daily max
# temperature (as above).

import sys

YYYY=sys.argv[1]
maxpoolsize=int(sys.argv[2])


def customization():

    global exp_id, zoom, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, dir_out, dir_out, filename_out, time_min, time_max

    ####################################################################################################################
    ####################################################################################################################

    # c u s t o m i z a t i o n   s e c t i o n

    ####################################################################################################################
    ####################################################################################################################


    # in the following lines you can customize parameters according to your needs


    exp_id = "ngc4008"
    zoom=9                     # zoom level (should +/- fit with the resolution of the output)

    lon_min = -125.            # region to be selected (as lon/lat, use negative values for western/southern hemisphere)
    lon_max = -115.
    lat_min = 30.
    lat_max = 40.

    interpol_method="nearest"  # experimental. Use only nearest by now, otherwise missvals at the boundaries might cause terrible problems.  
                               # to be moodified to choose one out of nearest, linear, cubic  

    res_out_x = .25            # resolution of the interpolated files in degree lon
    res_out_y = res_out_x      # resolution of the interpolated files in degree lat

    dir_out="/work/bb1153/m300363/fireweather_data/California"
    filename_out= dir_out + "/inputvars_ICON_ngc4008_California_025deg_" + YYYY + ".nc"   # keep YYYY somewhere in the filename 
                                                                                         # as this script will run multiple times
                                                                                         # called by one SLURM job per year 

    # end of customization section (usually there should be no need to change anything below this line

    ######################################################################################################################
    ######################################################################################################################

    match YYYY:
       case 2020:
          time_min = YYYY + "-01-02"
       case 1990:
          time_min = YYYY + "-01-01"    # first date to be selected as YYYY-MM-DD (do not use 2020-01-01)
       case _:
          time_min = YYYY + "-01-01"
 
    time_max = YYYY + "-12-31"    # last date to be selected as YYYY-MM-DD
 
    print("writing output to ",filename_out, flush=True)

    return exp_id, zoom, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, dir_out, dir_out, filename_out, time_min, time_max


#####################
# import libraries  #
#####################

import os
import intake
import dask
from dask.diagnostics import ProgressBar
from multiprocessing import Pool, Manager, current_process
import pandas as pd
import functools
import itertools
# import cmocean
import metpy.calc
from metpy.units import units
import healpy
import xarray as xr
import numpy as np
from scipy import interpolate
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# import logging
from collections.abc import Iterable

####################
# define functions #
####################

# def default_kwargs(**defaultKwargs):
#     def actual_decorator(fn):
#         @functools.wraps(fn)
#         def g(*args, **kwargs):
#             defaultKwargs.update(kwargs)
#             return fn(*args, **defaultKwargs)
#         return g
#     return actual_decorator

def get_nest(dx):
    return dx.crs.healpix_order == "nest"

def get_nside(dx):
    return dx.crs.healpix_nside

def attach_coords(ds):
    lons, lats = healpy.pix2ang(
        get_nside(ds), np.arange(ds.dims["cell"]), nest=get_nest(ds), lonlat=True
    )
    return ds.assign_coords(
        lat=(("cell",), lats, {"units": "degree_north"}),
        lon=(("cell",), lons, {"units": "degree_east"}),
    )

def compute_lhour(ds):
          
   lhour1 = 0 * ds.tas + ds.time.dt.hour + np.round(24/360*ds.lon)
   lhour2 = lhour1.where(lhour1>=0,lhour1+24)
   lhour3 = lhour2.where(lhour2<24,lhour2-24)
    
   return lhour3


def compute_noonvals(ds,arrayname,varname):

    # print("select noon values for ds: " + str(ds) + ", varname: " + varname +". output will be appended to " + arrayname)
    
    lhour=compute_lhour(ds)
    sel = np.where((lhour == 11) | (lhour == 12) | (lhour == 13),1,0)
    var_sel = eval("ds." + varname + "* sel")
    arrayname[varname] = var_sel.resample(time="D").sum()
    
    return arrayname[varname]


def compute_daymax(ds,arrayname,varname):

    # print("select noon values for ds: " + str(ds) + ", varname: " + varname +". output will be appended to " + arrayname)

    # var_sel = eval("ds." + varname + "* sel")
    arrayname[varname] = ds[varname].resample(time="D").max()

    return arrayname[varname]



# @default_kwargs(interpol_method="linear")
def interpolate_healpy2lonlat(input_array,output_array,varname,inlon,inlat,outlon,outlat,*args,**kwargs):
 
    print("interpolation method: ",interpol_method)    
    
    outgrid=np.meshgrid(outlon,outlat)

    values_interpolated=np.empty((np.shape(input_array)[0],len(outlat),len(outlon)))

    i=0

    for date in input_array.time:

        _day=datetime.strptime(np.datetime_as_string(date["time"],unit="D"),"%Y-%m-%d").day
        _month=datetime.strptime(np.datetime_as_string(date["time"],unit="D"),"%Y-%m-%d").month
        _year=datetime.strptime(np.datetime_as_string(date["time"],unit="D"),"%Y-%m-%d").year

        if ( _day == 1 ):
           print("interpolating data for ",varname," MM-YYYY = ", _month,"-",_year,flush=True)                                                        

        values_tsel=input_array.sel(time=date) # .drop(time)
        values_interpolated[i][:][:] =  interpolate.griddata((inlon,inlat), values_tsel, (outgrid[0][:][:],outgrid[1][:][:]), method=interpol_method)
        i += 1

    # print("interpolation to lon-lat for ",varname)    # without mp
    print("interpolation to lon-lat for ",varname, " computed on ", current_process(),flush=True)    # with mp

    output_array[varname] = values_interpolated
    
    return output_array[varname]


def main():

    customization()
    
    ##########################
    # load datasets (ICON)   #
    ##########################

    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    experiment = cat.ICON[exp_id]

    ds_3h = experiment(time="PT3H", zoom=zoom, chunks="auto").to_dask().pipe(attach_coords)   # for 2m temperature, precipitation, surface winds
    ds_1d = experiment(time="P1D", zoom=zoom, chunks="auto").to_dask().pipe(attach_coords)    # for 2m spec. humidity

    print("datasets loaded", flush=True)

    ########################################################################################
    # select region & pick (almost) noon-values for 3-hourly data                          #
    ########################################################################################

    # disclaimer: for time zones that do not have 12:00:00 values be relaxed by +/-1 hour  #

    ds_3h.lon[ds_3h.lon>180]=ds_3h.lon[ds_3h.lon>180]-360
    ds_3h_reg=ds_3h.sel(time=slice(time_min,time_max)).where((ds_3h.lon > lon_min) & (ds_3h.lon < lon_max) & (ds_3h.lat > lat_min) & (ds_3h.lat < lat_max),drop=True)

    ds_1d.lon[ds_1d.lon>180]=ds_1d.lon[ds_1d.lon>180]-360
    ds_1d_reg=ds_1d.sel(time=slice(time_min,time_max)).where((ds_1d.lon > lon_min) & (ds_1d.lon < lon_max) & (ds_1d.lat > lat_min) & (ds_1d.lat < lat_max),drop=True)

    print("select noon values", flush=True)

    # maxpoolsize = 4
    #
    # varlist = ['tas','pr','uas','vas','qv2m','pres_sfc']
    # poolsize = max(len(varlist),maxpoolsize)
    
    manager=Manager()
    noonvals=manager.dict()

    # with Pool(poolsize) as p:
    #
    #     iterable = itertools.zip_longest(itertools.repeat(ds_3h_reg,len(varlist)),itertools.repeat(noonvals,len(varlist)),varlist)
    #     p.starmap(compute_noonvals,iterable)
    #
    # print("noon values selected",flush=True)
    
    ########################################################################################
    # compute Tmax from 3-hourly data                                                      #
    ########################################################################################

    compute_daymax(ds_3h_reg,noonvals,"tas")

    print("computed daily maximum temperature from 3-hourly data", flush=True)

    ###################################################
    # compute wind speed from mean daily uas and vas  #
    ###################################################

    noonvals['uas'] = ds_1d_reg['uas']
    noonvals['vas'] = ds_1d_reg['vas']

    uas_unit = units(ds_1d_reg.uas.attrs.get("units"))
    vas_unit = units(ds_1d_reg.vas.attrs.get("units"))
    wind_speed_unit = uas_unit 

    noonvals['wind_speed'] = metpy.calc.wind_speed(noonvals['uas'] * uas_unit ,noonvals['vas'] * vas_unit)

    print("computed noon values for wind speed",flush=True)

    ################
    # get daily pr #
    ################

    pr_unit = units(ds_1d_reg.pr.attrs.get("units"))
    noonvals['pr'] = ds_1d_reg['pr'] * 86400
    pr_unit = "mm"

    print("extracted daily wind_speed and precip",flush=True)

    #####################################################
    # compute relative humidity from specific humidity  #
    #####################################################

    huss_unit = units(ds_1d_reg.qv2m.attrs.get("units"))
    noonvals['huss'] = ds_1d_reg['qv2m']

    pres_sfc_unit = units(ds_1d_reg.pres_sfc.attrs.get("units"))
    noonvals['pres_sfc'] = ds_1d_reg['pres_sfc']

    tas_unit = units(ds_1d_reg.tas.attrs.get("units"))

    noonvals['hurs'] = metpy.calc.relative_humidity_from_specific_humidity(noonvals['pres_sfc'] * pres_sfc_unit, noonvals['tas'] * tas_unit , noonvals['huss'] * huss_unit) * 100
    noonvals['hurs'] = noonvals['hurs'].where((noonvals['hurs'] < 100 ),100)
    hurs_unit = "%"

    print("computed noon values for relative humidity",flush=True)
     
    ########################
    # convert tas to degC  #
    ########################

    noonvals["tas"] =  noonvals["tas"] -273.15
    tas_unit = "C"

    wind_speed_unit="m/s"

    ######################################
    # interpolate data to lon-lat grid   #
    ######################################

    inlon = np.asarray(ds_1d_reg.lon)
    inlat = np.asarray(ds_1d_reg.lat)
    outlon = np.arange(lon_min,lon_max + res_out_x, res_out_x, dtype=float)
    outlat = np.arange(lat_min,lat_max + res_out_y, res_out_y, dtype=float)

    maxpoolsize = 4

    varlist = ['tas','pr','wind_speed','hurs']

    poolsize = max(len(varlist),maxpoolsize)

    manager=Manager()
    outvars_interpolated=manager.dict()

    with Pool(poolsize) as p:

        iterable = itertools.zip_longest(
            [ noonvals[var] for var in varlist ],
            itertools.repeat(outvars_interpolated,len(varlist)),
            [ var for var in varlist ],
            itertools.repeat(inlon,len(varlist)),
            itertools.repeat(inlat,len(varlist)),
            itertools.repeat(outlon,len(varlist)),
            itertools.repeat(outlat,len(varlist)),
            itertools.repeat(print("interpol_method=" + interpol_method),len(varlist))
        )
    
        p.starmap(interpolate_healpy2lonlat,iterable)

    outvars_interpolated['hurs'] = np.where((outvars_interpolated['hurs'] < 100),outvars_interpolated['hurs'],100)

    #######################
    # create output file  #
    #######################

    print("writting outdata to",filename_out,flush="True")

    dsout = xr.Dataset({
            "tas": (("time", "lat", "lon"), outvars_interpolated["tas"] , {"units": str(tas_unit),  "long_name": "2m temperature"}),
            "wind_speed": (("time", "lat", "lon"), outvars_interpolated["wind_speed"], {"units": str(wind_speed_unit),  "long_name": "10m wind speed"}),
            "pr": (("time", "lat", "lon"), outvars_interpolated["pr"], {"units": str(pr_unit),  "long_name": "precipitation rate"}),
            "hurs": (("time", "lat", "lon"), outvars_interpolated["hurs"], {"units": str(hurs_unit),  "long_name": "surface relative humidity"})},
            coords={
            "time": [ datetime.strptime(np.datetime_as_string(x,unit="D") + "T12:00:00", "%Y-%m-%dT%H:%M:%S") for x in noonvals["tas"].time ], 
            "lat": ("lat", outlat, {"units": "degree_north",  "long_name": "latitude"}),
            "lon": ("lon", outlon, {"units": "degree_east",  "long_name": "longitude"}),
            },)
    dsout.to_netcdf(filename_out,mode="w")


if __name__ == "__main__":
    main()
