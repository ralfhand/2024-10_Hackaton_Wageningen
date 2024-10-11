#!/usr/bin/env python
 
# This script preprocesses the IFS data to create input files for the firedanger tool written by Daniel Steinfeld.
# The preprocessing includes the selection of subregions, the time period, and variables and the interpolation from Healpix format to a lon-lat grid.
# Customization can be done in the customization section below.
#
# In the current configuration the script computes wind_speed from daily mean u- and v-wind (daily mean of hourly u10 and v10), daily max temperature
# (daily max of hourly 2t), daily mean precipitation (daily mean of hourly tprate) and daily min relative humidity from daily mean dewpoint tempreature (daily mean of 2d)
# and daily max temperature (as above). All hourly variables were taken from 2D_hourly_0.25deg.

import sys
 
YYYY=sys.argv[1]
maxpoolsize=int(sys.argv[2])

def customization():

    global exp_id, time, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, dir_out, dir_out, filename_out, time_min, time_max

    ####################################################################################################################
    ####################################################################################################################
 
    # c u s t o m i z a t i o n   s e c t i o n 
 
    ####################################################################################################################
    ####################################################################################################################
 
 
    # in the following lines you can customize parameters according to your needs
 
    exp_id = "IFS_9-FESOM_5-production"
 
    time = "2D_hourly_0.25deg"
 
    lon_min = -125.            # region to be selected (as lon/lat, use negative values for western/southern hemisphere)
    lon_max = -115.
    lat_min = 30.
    lat_max = 40.
 
    interpol_method="nearest"  # experimental. Use only nearest by now, otherwise missvals at the boundaries might cause terrible problems.  
                               # to be moodified to choose one out of nearest, linear, cubic 
 
    res_out_x = .25            # resolution of the interpolated files in degree lon
    res_out_y = res_out_x      # resolution of the interpolated files in degree lat
 
    dir_out="/work/bb1153/m300363/fireweather_data/California"
    filename_out= dir_out + "/inputvars_" + exp_id + "_California_025deg_" + YYYY + ".nc"   # keep YYYY somewhere in the filename 
                                                                                         # as this script will run multiple times
                                                                                         # called by one SLURM job per year 
 
    # end of customization section (usually there should be no need to change anything below this line
 
    ######################################################################################################################
    ######################################################################################################################
 
 
    match YYYY:
       case 2020 | 1990:
          time_min = YYYY + "-01-01"    # first date to be selected as YYYY-MM-DD (do not use 2020-01-01)
       case _:
          time_min = YYYY + "-01-01"
 
    time_max = YYYY + "-12-31"    # last date to be selected as YYYY-MM-DD
 
    print("writing output to ",filename_out, flush=True)

    return exp_id, time, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, dir_out, dir_out, filename_out, time_min, time_max

#####################
# import libraries  #
#####################

import os
import intake
import dask
from dask.diagnostics import ProgressBar
from multiprocessing import Pool, Manager, Process, current_process
import pandas as pd
import functools
import itertools
import gribscan
import eccodes
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

def attach_coords(ds):
    model_lon = ds.lon.values
    model_lat = ds.lat.values
    return ds.assign_coords(
        lat = (("value",), model_lat, {"units": "degree_north"}),
        lon = (("value",), model_lon, {"units": "degree_east"}),
    )

def compute_lhour(ds):
          
   lhour1 = 0 * ds.tp + ds.time.dt.hour + np.round(24/360*ds.lon)
   lhour2 = lhour1.where(lhour1>=0,lhour1+24)
   lhour3 = lhour2.where(lhour2<24,lhour2-24)
    
   return lhour3


def compute_noonvals(ds,arrayname,varname):

    # currently obsolete. Kept for eventuaöl later usage
 
    # print("select noon values for ds: " + str(ds) + ", varname: " + varname +". output will be appended to " + arrayname)
    
    lhour=compute_lhour(ds)
    print("lhour computed")
    sel = np.where((lhour == 12),1,0)
    # var_sel = eval("ds." + varname + "* sel")
    var_sel = ds[varname] * sel
    arrayname[varname] = var_sel.resample(time="D").sum()

    del lhour,sel,var_sel

    # print("noonvals for ",varname)    # without mp
    print("noonvals for ",varname, " computed on ", current_process(),flush=True)    # with mp
    
    return arrayname[varname]

def compute_daymax(ds,arrayname,varname):
 
    # print("select noon values for ds: " + str(ds) + ", varname: " + varname +". output will be appended to " + arrayname)
 
    # var_sel = eval("ds." + varname + "* sel")
    arrayname[varname] = ds[varname].resample(time="D").max()

    # print("noonvals for ",varname)    # without mp
    print("daymax for ",varname, " computed on ", current_process(),flush=True)    # with mp
    
    return arrayname[varname]

def compute_daymin(ds,arrayname,varname):
 
    # print("select noon values for ds: " + str(ds) + ", varname: " + varname +". output will be appended to " + arrayname)
 
    # var_sel = eval("ds." + varname + "* sel")
    arrayname[varname] = ds[varname].resample(time="D").min()

    # print("noonvals for ",varname)    # without mp
    print("daymin for ",varname, " computed on ", current_process(),flush=True)    # with mp
    
    return arrayname[varname]

def compute_daymean(ds,arrayname,varname):
 
    # print("select noon values for ds: " + str(ds) + ", varname: " + varname +". output will be appended to " + arrayname)
 
    # var_sel = eval("ds." + varname + "* sel")
    arrayname[varname] = ds[varname].resample(time="D").mean()

    # print("noonvals for ",varname)    # without mp
    print("daymean for ",varname, " computed on ", current_process(),flush=True)    # with mp
    
    return arrayname[varname]


# @default_kwargs(interpol_method="linear")
def interpolate_healpy2lonlat(input_array,output_array,varname,inlon,inlat,outlon,outlat,*args,**kwargs):
 
    print("interpolation method: ",interpol_method, flush=True)    
    
    outgrid=np.meshgrid(outlon,outlat)

    values_interpolated=np.empty((np.shape(input_array)[0],len(outlat),len(outlon)))

    i=0

    for date in input_array.time:

        _day=datetime.strptime(np.datetime_as_string(date["time"],unit="D"),"%Y-%m-%d").day
        _month=datetime.strptime(np.datetime_as_string(date["time"],unit="D"),"%Y-%m-%d").month
        _year=datetime.strptime(np.datetime_as_string(date["time"],unit="D"),"%Y-%m-%d").year

        if ( _day == 2 ):
           print("interpolating data for ",varname," MM-YYYY = ", _month,"-",_year,flush=True)                                                        

        values_tsel=input_array.sel(time=date) # .drop(time)
        values_interpolated[i][:][:] =  interpolate.griddata((inlon,inlat), values_tsel, (outgrid[0][:][:],outgrid[1][:][:]), method=interpol_method)
        i += 1

    output_array[varname] = values_interpolated

    del values_interpolated

    # print("interpolation to lon-lat for ",varname)    # without mp
    print("interpolation to lon-lat for ",varname, " computed on ", current_process(),flush=True)    # with mp
    
    return output_array[varname]

def main():

    customization()
    
    ##########################
    # load datasets (IFS)   #
    ##########################

    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    experiment = cat.IFS[exp_id][time]


    ds = experiment(chunks="auto").to_dask().pipe(attach_coords)
    print("datasets loaded", flush=True)

    ########################################################################################
    # select region & pick noon-values                      #
    ########################################################################################

    ds.lon[ds.lon>180]=ds.lon[ds.lon>180]-360
    ds_reg=ds.sel(time=slice(time_min,time_max)).where((ds.lon > lon_min) & (ds.lon < lon_max) & (ds.lat > lat_min) & (ds.lat < lat_max),drop=True)
    
    poolsize = maxpoolsize 
    
    varlist = ['2d','tprate','10u','10v']

    # poolsize = (len(varlist),maxpoolsize)
    
    manager=Manager()
    noonvals=manager.dict()

#    for var in varlist:
#        compute_noonvals(ds_reg,noonvals,var)
 
#    jobs=[]
 
#    for var in varlist:
#        p = Process(target=compute_noonvals,args=(ds_reg,noonvals,var))
#        jobs.append(p)
#        p.start()
 
#    for proc in jobs:
#        proc.join()
 
#    with Pool(poolsize) as pool:
#        iterable = itertools.zip_longest(itertools.repeat(ds_reg,len(varlist)),itertools.repeat(noonvals,len(varlist)),varlist)
#        # chunks = chunk(list_a, multiprocessing.cpu_count())    
#        for instance in pool.starmap(compute_noonvals, [ step for step in iterable ],chunksize=poolsize):
#            output = instance
#            noonvals[instance[0]] = instance[1]
#            del output

    with Pool(poolsize) as p:
        
        iterable = itertools.zip_longest(itertools.repeat(ds_reg,len(varlist)),itertools.repeat(noonvals,len(varlist)),varlist)
        p.starmap(compute_daymean,iterable)
#        
    compute_daymax(ds_reg,noonvals,"2t")

    print("noon values selected",flush=True)

    ########################################
    # compute wind speed from uas and vas  #
    ########################################

    uas_unit = units(ds_reg['10u'].attrs.get("units"))
    vas_unit = units(ds_reg['10v'].attrs.get("units"))

    noonvals['wind_speed'] = metpy.calc.wind_speed(noonvals['10u'] * uas_unit ,noonvals['10v'] * vas_unit)

    print("computed noon values for wind speed",flush=True)


    ########################################
    # convert tprate to mm/day             #
    ########################################

    pr_unit = units(ds_reg['tp'].attrs.get("units"))

    noonvals['tprate'] = noonvals['tprate'] * 86400
    pr_unit = "mm"

    #####################################################
    # compute relative humidity from specific humidity  #
    #####################################################

    dewpoint_unit = units(ds_reg['2d'].attrs.get("units"))
    tas_unit = units(ds_reg['2t'].attrs.get("units"))

    noonvals['hurs'] = metpy.calc.relative_humidity_from_dewpoint(noonvals['2t'] * tas_unit, noonvals['2d'] * dewpoint_unit)
    noonvals['hurs'] = noonvals['hurs'].where((noonvals['hurs'] < 1 ),1) * 100

    hurs_unit="%"

    print("computed value for relative humidity",flush=True)

 
    #####################################################
    # compute relative humidity from specific humidity  #
    #####################################################

    noonvals["2t"] = noonvals["2t"] - 273.15
    tas_unit = "C" 

    wind_speed_unit="m/s"

    print("converted all units",flush=True)

    ######################################
    # interpolate data to lon-lat grid   #
    ######################################

    inlon = np.asarray(ds_reg.lon)
    inlat = np.asarray(ds_reg.lat)
    outlon = np.arange(lon_min,lon_max + res_out_x, res_out_x, dtype=float)
    outlat = np.arange(lat_min,lat_max + res_out_y, res_out_y, dtype=float)

    poolsize = maxpoolsize

    # varlist = ['2t','tp','10u','10v','wind_speed','hurs']
    varlist = ['2t','tprate','wind_speed','hurs']

    manager=Manager()
    outvars_interpolated=manager.dict()

## fallback: non parallelized loop
#    for var in varlist:
#        interpolate_healpy2lonlat(noonvals[var],outvars_interpolated,var,inlon,inlat,outlon,outlat,print("interpol_method=" + interpol_method),len(varlist))
    
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

    print("writting outdata to",filename_out,flush="True")

    dsout = xr.Dataset({
        "tas": (("time", "lat", "lon"), outvars_interpolated["2t"] , {"units": str(tas_unit)}),
        "wind_speed": (("time", "lat", "lon"), outvars_interpolated["wind_speed"], {"units": str(wind_speed_unit)}),
        "pr": (("time", "lat", "lon"), outvars_interpolated["tprate"], {"units": str(pr_unit)}),
        "hurs": (("time", "lat", "lon"), outvars_interpolated["hurs"], {"units": str(hurs_unit)})},
        coords={
        "time": [ datetime.strptime(np.datetime_as_string(x,unit="D") + "T12:00:00", "%Y-%m-%dT%H:%M:%S") for x in noonvals["2t"].time ],
        "lat": ("lat", outlat, {"units": "degree_north"}),
        "lon": ("lon", outlon, {"units": "degree_east"}),
        },)
 
    dsout.to_netcdf(filename_out)

if __name__ == "__main__":
    main()

