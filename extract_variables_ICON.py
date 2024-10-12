#!/usr/bin/env python3

"""

This script extracts variables from the ICON simulations and prepares them for further processing in cdo 
It selects regions and interpolates the input data to a regular grid.

It should be called as follows:

extract_variables_ICON.py YYYY expid lon_min lon_max lat_min lat_max res_out outfile maxpoolsize

where:

YYYY          Year
expid         Experiment ID as in catalog (e.g. ngc4008)
lon_min       first longitude
lon_max       last longitude
lat_min       first latitude
lat_max       last latitude
res_out       resolution of the output
outfile       name of the output file (including path)
maxpoolsize   maximum size of the multiprocessing pools

author: Ralf Hand (ralf.hand@unibe.ch)

"""


#####################
# import libraries  #
#####################

import sys
import os
import intake
import dask
# from dask.diagnostics import ProgressBar
from multiprocessing import Pool, Manager, current_process
# import pandas as pd
# import functools
import itertools
# import cmocean
import metpy.calc
from metpy.units import units
import healpy
import xarray as xr
import numpy as np
from scipy import interpolate
from datetime import datetime
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import logging
from collections.abc import Iterable

####################
# define functions #
####################

def read_args():

    """
    Read arguments from input
    """
    
    global YYYY, exp_id, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, outfile, maxpoolsize, time_min, time_max, zoom

    YYYY = int(sys.argv[1])
    exp_id = str(sys.argv[2])
    lon_min = float(sys.argv[3])
    lon_max = float(sys.argv[4])
    lat_min = float(sys.argv[5])
    lat_max = float(sys.argv[6])
    res_out = float(sys.argv[7])
    outfile = str(sys.argv[8])
    maxpoolsize=int(sys.argv[9])

    res_out_x = res_out
    res_out_y = res_out

    zoom = 9
    interpol_method="nearest"

    match YYYY:
       case 2020:
          time_min = str(YYYY) + "-01-01"
       case _:
          time_min = str(int(YYYY) - 1) + "-12-31"

    match YYYY:
       case 2049:
          time_max = str(YYYY) + "-12-31"
       case _:
          time_max = str(int(YYYY) + 1) + "-01-01"

    print("read the following arguments from input:",flush=True) 
    print("        YYYY = ",YYYY,flush=True)
    print("      exp_id = ", exp_id,flush=True)
    print("     lon_min = ", lon_min," lon_max=", lon_max, " lat_min=", lat_min, " lat_max=", lat_max,flush=True)
    print("     res_out = ", res_out,flush=True)
    print("     outfile = ", outfile,flush=True)
    print(" maxpoolsize = ", maxpoolsize,flush=True)

    return  YYYY, exp_id, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, outfile, maxpoolsize, time_min, time_max, zoom


def get_nest(dx):
    return dx.crs.healpix_order == "nest"



def get_nside(dx):
    return dx.crs.healpix_nside



def attach_coords(ds):

    """
    Attach ccordinates to input data
    """
    
    lons, lats = healpy.pix2ang(
        get_nside(ds), np.arange(ds.dims["cell"]), nest=get_nest(ds), lonlat=True
    )
    return ds.assign_coords(
        lat=(("cell",), lats, {"units": "degree_north"}),
        lon=(("cell",), lons, {"units": "degree_east"}),
    )

def set_localtime(ds,outname,var):

    """
    Corrects the time of var from UTC to local time
    """
    
    timeshift = 0 * ds.sel(time=slice(str(YYYY - 1) + "-12-31",str(YYYY) + "-12-31"))[var][:,:] + np.round(24/360*ds.lon)
    print("timeshift.shape:",timeshift.shape)

    i=0

    timeshift_check_list = [-9,-6,-3,0,3,6,9,12] 
    # timeshift_check_list = range(-9,-3,3)

    data_local=np.empty((len(timeshift_check_list),len(timeshift_check_list),ds.sel(time=slice(str(YYYY-1)+"-12-31",str(YYYY)+"-12-31"))[var].shape[0],ds[var].shape[1]))
    
    for timeshift_check in timeshift_check_list:

        print("calculating localtime for GMT",timeshift_check,"hours.")
 
        # if timeshift_check < 0:
        #   data_timsel = ds.sel(time=slice(str(int(YYYY) - 1) + "-12-31-" + str(24 + timeshift_check) + ":00:00",
        #                                    str(YYYY) + "-12-31-" + str(23 + timeshift_check) + ":59:59"))
        # elif timeshift_check == 0:
        #    data_timesel = ds.sel(time=slice(str(YYYY) + "-01-01",str(YYYY) + "-12-31"))
        # else:
        #    data_timsel = ds.sel(time=slice(str(YYYY) + "-01-01-" + str(timeshift_check) + ":00:00",
        #                                    str(int(YYYY) + 1) + "-01-01-" + str(timeshift_check - 1 ) + ":59:59"))

        if timeshift_check < 0:
            data_timsel = ds.sel(time=slice(str(int(YYYY) - 1) + "-12-30-" + str(24 + timeshift_check) + ":00:00",
                                            str(YYYY) + "-12-31-" + str(23 + timeshift_check) + ":59:59"))
        elif timeshift_check == 0:
            data_timesel = ds.sel(time=slice(str(YYYY - 1) + "-12-31",str(YYYY) + "-12-31"))
        else:
            data_timsel = ds.sel(time=slice(str(YYYY - 1) + "-12-31-" + str(timeshift_check) + ":00:00",
                                            str(int(YYYY) + 1) + "-01-01-" + str(timeshift_check - 1 ) + ":59:59"))


        data=data_timsel[var]

        sel = np.where(((timeshift == timeshift_check - 1) | (timeshift == timeshift_check) | (timeshift == timeshift_check + 1)),1,0)

        print("shape of sel",sel.shape)
        # print("sel:", sel)

        data_local[i,:,:] = data * sel

        # print("data_local for ",var,": ", data_local[i,:,:])

        i=+1

    # print("shape of data.local: ",data_local.shape)
    # print("data.local: ", data_local)

    outname[var] = data_local.sum(axis=0)[0,:,:]

    # print("return value outvar for", var,": ",outname[var])
 
    # print("local time set for ",var)    # without mp
    print("local time set for ",var, "computed on ", current_process(),flush=True)    # with mp
    
    if (var == "tas"):
        outname["timeshift"] = timeshift.values
        return outname["timeshift"], outname[var]
    else:
        return outname[var]

# @default_kwargs(interpol_method="linear")
def interpolate_healpy2lonlat(input_array,output_array,varname,inlon,inlat,outlon,outlat,*args,**kwargs):

    """
    Interpolate from Healpix to lonlat
    """
    
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

    read_args()
    
    ##########################
    # load datasets (ICON)   #
    ##########################

    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    experiment = cat.ICON[exp_id]

    ds = experiment(time="PT3H", zoom=zoom, chunks="auto").to_dask().pipe(attach_coords)

    print("datasets loaded", flush=True)

    ##########################
    # select region          #
    ##########################

    # disclaimer: for time zones that do not have 12:00:00 values be relaxed by +/-1 hour  #

    ds.lon[ds.lon>180]=ds.lon[ds.lon>180]-360
    ds_reg=ds.where((ds.lon > lon_min) & (ds.lon < lon_max) & (ds.lat > lat_min) & (ds.lat < lat_max),drop=True)

    varlist=["uas","vas","qv2m","pr","tas","pres_sfc"]

    poolsize=min(len(varlist),maxpoolsize)
    
    manager=Manager()
    localtime_vars=manager.dict()

    ###############################
    # correct time to local time  #
    ###############################
     
    iterable = itertools.zip_longest(
        itertools.repeat(ds_reg,len(varlist)),
        itertools.repeat(localtime_vars,len(varlist)),
        [ var for var in varlist ])

#    for item in iterable:
#        print(item)

    with Pool(poolsize) as p:
        p.starmap(set_localtime,iterable)    

    localtime_shape=localtime_vars["uas"].shape
    print("localtime_shape: ",localtime_shape)

    #############################################
    # compute wind speed from mean uas and vas  #
    #############################################

    uas_unit = units(ds_reg.uas.attrs.get("units"))
    vas_unit = units(ds_reg.vas.attrs.get("units"))
    wind_speed_unit = uas_unit 

    localtime_vars['wind_speed'] = metpy.calc.wind_speed(localtime_vars['uas'] * uas_unit ,localtime_vars['vas'] * vas_unit)

    wind_speed_unit="m/s"

    print("computed wind speed from uas and vas",flush=True)

    ################
    # pr to mm/das #
    ################

    pr_unit = units(ds_reg.pr.attrs.get("units"))
    localtime_vars['pr'] = localtime_vars['pr'] * 86400
    pr_unit = "mm"

    print("extracted daily wind_speed and precip",flush=True)

    #####################################################
    # compute relative humidity from specific humidity  #
    #####################################################

    huss_unit = units(ds_reg.qv2m.attrs.get("units"))
    pres_sfc_unit = units(ds_reg.pres_sfc.attrs.get("units"))
    tas_unit = units(ds_reg.tas.attrs.get("units"))

    localtime_vars['hurs'] = metpy.calc.relative_humidity_from_specific_humidity(
        localtime_vars['pres_sfc'] * pres_sfc_unit,
        localtime_vars['tas'] * tas_unit ,
        localtime_vars['qv2m'] * huss_unit) * 100

#    del localtime_vars["qv2m"], localtime_vars['pres_sfc']
#    
#    localtime_vars['hurs'] = localtime_vars['hurs'].where((localtime_vars['hurs'] < 100 ),100)
#
    hurs_unit = "%"
    
    print("computed relative humidity",flush=True)
     
    ########################
    # convert tas to degC  #
    ########################

    localtime_vars["tas"] =  localtime_vars["tas"] - 273.15
    tas_unit = "C"

    localtime_vars_xr=xr.Dataset(
            data_vars=dict(
                uas=(["time", "idx"], localtime_vars["uas"]),
                vas=(["time", "idx"], localtime_vars["vas"]),
                wind_speed=(["time", "idx"], localtime_vars["wind_speed"]),
                hurs=(["time", "idx"], localtime_vars["hurs"]),
                pr=(["time", "idx"], localtime_vars["pr"]),
                tas=(["time", "idx"], localtime_vars["tas"]),
                pres_sfc=(["time", "idx"], localtime_vars["pres_sfc"]),
                timeshift=(["time", "idx"], localtime_vars["timeshift"]),
                ),
            coords={
               "time": ds_reg.sel(time=slice(str(YYYY - 1) + "-12-31",str(YYYY) + "-12-31")).time,
               },
            )


    ######################################
    # interpolate data to lon-lat grid   #
    ######################################

    inlon = np.asarray(ds_reg.lon)
    inlat = np.asarray(ds_reg.lat)
    outlon = np.arange(lon_min,lon_max + res_out_x, res_out_x, dtype=float)
    outlat = np.arange(lat_min,lat_max + res_out_y, res_out_y, dtype=float)

    varlist = ["tas","pr","wind_speed","hurs","timeshift"]
    poolsize = min(len(varlist),maxpoolsize)

    manager=Manager()
    outvars=manager.dict()

    with Pool(poolsize) as p:

        iterable = itertools.zip_longest(
            [ localtime_vars_xr[var] for var in varlist ],
            itertools.repeat(outvars,len(varlist)),
            [ var for var in varlist ],
            itertools.repeat(inlon,len(varlist)),
            itertools.repeat(inlat,len(varlist)),
            itertools.repeat(outlon,len(varlist)),
            itertools.repeat(outlat,len(varlist)),
            itertools.repeat(print("interpol_method=" + interpol_method),len(varlist))
        )
    
        p.starmap(interpolate_healpy2lonlat,iterable)
#
#    outvars['hurs'] = np.where((outvars_interpolated['hurs'] < 100),outvars_interpolated['hurs'],100)
#
#    #######################
#    # create output file  #
#    #######################
#
    print("writting outdata to",outfile,flush="True")
#
    dsout = xr.Dataset({
            "tas": (("time", "lat", "lon"), outvars["tas"] , {"units": str(tas_unit),  "long_name": "2m temperature"}),
            "wind_speed": (("time", "lat", "lon"), outvars["wind_speed"], {"units": str(wind_speed_unit),  "long_name": "10m wind speed"}),
            "pr": (("time", "lat", "lon"), outvars["pr"], {"units": str(pr_unit),  "long_name": "precipitation rate"}),
            "hurs": (("time", "lat", "lon"), outvars["hurs"], {"units": str(hurs_unit),  "long_name": "surface relative humidity"}),
            "timeshift": (("time", "lat", "lon"), outvars["timeshift"], {"units": str(hurs_unit),  "long_name": "timeshift s.r.t. GMT"}),
            },
            coords={
            "time": ds_reg.sel(time=slice(str(YYYY - 1) + "-12-31",str(YYYY) + "-12-31")).time, 
            "lat": ("lat", outlat, {"units": "degree_north",  "long_name": "latitude"}),
            "lon": ("lon", outlon, {"units": "degree_east",  "long_name": "longitude"}),
            },)
    dsout.to_netcdf(outfile,mode="w")

if __name__ == "__main__":
    main()
