#!/usr/bin/env python3

"""

This script extracts variables from the ICFS simulations and prepares them for further processing in cdo 
It selects regions and interpolates the input data to a regular grid.

It should be called as follows:

extract_variables_IFS.py YYYY expid lon_min lon_max lat_min lat_max res_out outfile maxpoolsize

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
import itertools
import metpy.calc
from metpy.units import units
import healpy
import xarray as xr
import numpy as np
from scipy import interpolate
from datetime import datetime
# import logging
from collections.abc import Iterable

# from prepare_input_firedanger_functions import *

####################
# define functions #
####################

def read_args():

    """
    Read arguments from input
    """
   
    global YYYY, exp_id, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, outfile, maxpoolsize, dateout_min, dateout_max, zoom, time, model, exp

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

    zoom=9                    # for ICON
    time="2D_hourly_0.25deg"  # for IFS

    interpol_method="nearest"

    class exp:
        match exp_id:
            case "IFS_9-FESOM_5-production-hist":
                fyear = 1990
                lyear = 2019
            case "IFS_9-FESOM_5-production":
                fyear = 2020
                lyear = 2049
            case _:
                raise ValueError("ex_pid " + exp_id + " unknown. quit.")
        model="IFS"

#    match exp_id:
#        case "IFS_9-FESOM_5-production-hist" | "IFS_9-FESOM_5-production":
#            model = "IFS"
#        case "ngc4008":
#            model="ICON"

    if (( int(YYYY) < exp.fyear) | (int(YYYY) > exp.lyear)):
        raise ValueError(str(YYYY) + " is outside allowed year range for exp_id " + exp_id)

    match YYYY:
       case exp.fyear:
          dateout_min = str(YYYY) + "-01-02"
       case _:
          dateout_min = str(int(YYYY) - 1) + "-12-31"

    match YYYY:
       case exp.lyear:
          dateout_max = str(YYYY) + "-12-30"
       case _:
          dateout_max = str(YYYY) + "-12-31"

    print("read the following arguments from input:",flush=True) 
    print("        YYYY = ",YYYY,flush=True)
    print("      exp_id = ", exp_id,flush=True)
    print("     lon_min = ", lon_min," lon_max=", lon_max, " lat_min=", lat_min, " lat_max=", lat_max,flush=True)
    print("     res_out = ", res_out,flush=True)
    print("     outfile = ", outfile,flush=True)
    print(" maxpoolsize = ", maxpoolsize,flush=True)

    return  YYYY, exp_id, lon_min, lon_max, lat_min, lat_max, interpol_method, res_out_x, res_out_y, outfile, maxpoolsize, dateout_min, dateout_max, zoom, time, exp

def get_nest(dx):
    return dx.crs.healpix_order == "nest"



def get_nside(dx):
    return dx.crs.healpix_nside



def attach_coords(ds):

    """
    Attach ccordinates to input data
    """ 
   
    print("attaching coordinates to", exp.model, "output")

    match exp.model:

        case "ICON":

            lons, lats = healpy.pix2ang(
                get_nside(ds), np.arange(ds.dims["cell"]), nest=get_nest(ds), lonlat=True
                )

            return ds.assign_coords(
                lat=(("cell",), lats, {"units": "degree_north"}),
                lon=(("cell",), lons, {"units": "degree_east"}),
                )

        case "IFS":
            model_lon = ds.lon.values
            model_lat = ds.lat.values
            return ds.assign_coords(
                lat = (("value",), model_lat, {"units": "degree_north"}),
                lon = (("value",), model_lon, {"units": "degree_east"}),
                )
    

def set_localtime(ds,outname,var):

    """
    Corrects the time of var from UTC to local time
    """
    
    timeshift = 0 * ds.sel(time=slice(dateout_min,dateout_max))[var][:,:] + np.round(24/360*ds.lon)
    print("timeshift.shape:",timeshift.shape)

    i=0

    # speed optimization: avoid computation of non-used time zones

    ftimezone=np.round(24/360*lon_min)
    ltimezone=np.round(24/360*lon_max)

    match exp.model:
        case "ICON":
            timeshift_step=3
            timeshift_check_list = [-9,-6,-3,0,3,6,9,12] 
        case "IFS":
            timeshift_step=1
            timeshift_check_list = [-8]
    
    ftimezone=int(max( timeshift_step - 12 , ftimezone)) # do not use UTC-12h
    ltimezone=int(min( ltimezone, 12 + timeshift_step))  # do not use UTC+12h

    timeshift_check_list = range(ftimezone,ltimezone + timeshift_step, timeshift_step)
    
    print("extracting the following timezones (UTC + x):", timeshift_check_list)

    data_local=np.empty((len(timeshift_check_list),len(timeshift_check_list),ds.sel(time=slice(dateout_min,dateout_max))[var].shape[0],ds[var].shape[1]))

    for timeshift_check in timeshift_check_list:

        print("calculating localtime for GMT",timeshift_check,"hours.")
 
        match YYYY:

            case exp.fyear:
                if timeshift_check < 0:
                    data_timsel = ds.sel(time=slice(str(YYYY) + "-01-01-" + str(24 + timeshift_check) + ":00:00",
                                                    str(YYYY) + "-12-31-" + str(23 + timeshift_check) + ":59:59"))
                elif timeshift_check == 0:
                    data_timesel = ds.sel(time=slice(str(YYYY) + "-01-02",str(YYYY) + "-12-31"))
                else:
                    data_timsel = ds.sel(time=slice(str(YYYY) + "-01-02-" + str(timeshift_check) + ":00:00",
                                                    str(int(YYYY) + 1) + "-01-01-" + str(timeshift_check - 1 ) + ":59:59"))

            case exp.lyear:
                if timeshift_check < 0:
                    data_timsel = ds.sel(time=slice(str(int(YYYY) - 1) + "-12-30-" + str(24 + timeshift_check) + ":00:00",
                                                    str(YYYY) + "-12-30-" + str(23 + timeshift_check) + ":59:59"))
                elif timeshift_check == 0:
                    data_timesel = ds.sel(time=slice(str(YYYY - 1) + "-12-30",str(YYYY) + "-12-31"))
                else:
                    data_timsel = ds.sel(time=slice(str(YYYY - 1) + "-12-31-" + str(timeshift_check) + ":00:00",
                                                    str(int(YYYY)) + "-12-31-" + str(timeshift_check - 1 ) + ":59:59"))

            case _:
                if timeshift_check < 0:
                    data_timsel = ds.sel(time=slice(str(int(YYYY) - 1) + "-12-30-" + str(24 + timeshift_check) + ":00:00",
                                                    str(YYYY) + "-12-31-" + str(23 + timeshift_check) + ":59:59"))
                elif timeshift_check == 0:
                    data_timesel = ds.sel(time=slice(str(YYYY - 1) + "-12-31",str(YYYY) + "-12-31"))
                else:
                    data_timsel = ds.sel(time=slice(str(YYYY - 1) + "-12-31-" + str(timeshift_check) + ":00:00",
                                                    str(int(YYYY) + 1) + "-01-01-" + str(timeshift_check - 1 ) + ":59:59"))



        data=data_timsel[var]

        match exp.model:
            case "ICON":
                sel = np.where(((timeshift == timeshift_check - 1) | (timeshift == timeshift_check) | (timeshift == timeshift_check + 1)),1,0)
            case "IFS":
                sel = np.where((timeshift == timeshift_check),1,0)

        print("shape of sel",sel.shape)
        # print("sel:", sel)

        data_local[i,:,:] = data * sel

        # print("data_local for ",var,": ", data_local[i,:,:])

        i=+1

    outname[var] = data_local.sum(axis=0)[0,:,:]

    print("local time set for ",var, "computed on ", current_process(),flush=True)    # with mp
    
    if ((var == "tas") | (var == "2t")):
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




def write_outfile():

    """
    create output file 
    """

    print("writting outdata to",outfile,flush="True")

    dsout = xr.Dataset({
            "tas": (("time", "lat", "lon"), outvars["tas"] , {"units": str(tas_unit),  "long_name": "2m temperature"}),
            "wind_speed": (("time", "lat", "lon"), outvars["wind_speed"], {"units": str(wind_speed_unit),  "long_name": "10m wind speed"}),
            "pr": (("time", "lat", "lon"), outvars["pr"], {"units": str(pr_unit),  "long_name": "precipitation rate"}),
            "hurs": (("time", "lat", "lon"), outvars["hurs"], {"units": str(hurs_unit),  "long_name": "surface relative humidity"}),
            "timeshift": (("time", "lat", "lon"), outvars["timeshift"], {"units": str(hurs_unit),  "long_name": "timeshift s.r.t. GMT"}),
            },
            coords={
            "time": ds_reg.sel(time=slice(dateout_min,dateout_max)).time,
            "lat": ("lat", outlat, {"units": "degree_north",  "long_name": "latitude"}),
            "lon": ("lon", outlon, {"units": "degree_east",  "long_name": "longitude"}),
            },)
    dsout.to_netcdf(outfile,mode="w")


def main():

    read_args()
    
    ##########################
    # load datasets (IFS)    #
    ##########################

    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    experiment = cat.IFS[exp_id][time]

    ds = experiment(chunks="auto").to_dask().pipe(attach_coords)
    print("datasets loaded", flush=True)

    ##########################
    # select region          #
    ##########################

    # disclaimer: for time zones that do not have 12:00:00 values be relaxed by +/-1 hour  #

    ds.lon[ds.lon>180]=ds.lon[ds.lon>180]-360
    ds_reg=ds.where((ds.lon > lon_min) & (ds.lon < lon_max) & (ds.lat > lat_min) & (ds.lat < lat_max),drop=True)

    varlist = ['2d','tprate','10u','10v','2t']

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

    localtime_shape=localtime_vars["tprate"].shape
    print("localtime_shape: ",localtime_shape)

    #############################################
    # compute wind speed from mean uas and vas  #
    #############################################

    uas_unit = units(ds_reg['10u'].attrs.get("units"))
    vas_unit = units(ds_reg['10v'].attrs.get("units"))

    localtime_vars['wind_speed'] = metpy.calc.wind_speed(localtime_vars['10u'] * uas_unit , localtime_vars['10v'] * vas_unit)

    wind_speed_unit="m/s"

    print("computed wind speed from uas and vas",flush=True)

    #################
    # pr to mm/days #
    #################

    pr_unit = units(ds_reg.tprate.attrs.get("units"))
    localtime_vars['pr'] = localtime_vars['tprate'] * 86400
    pr_unit = "mm"

    print("converted tprate to mm/day",flush=True)

    #####################################################
    # compute relative humidity from specific humidity  #
    #####################################################

    dewpoint_unit = units(ds_reg['2d'].attrs.get("units"))
    tas_unit = units(ds_reg['2t'].attrs.get("units"))

    localtime_vars['hurs'] = metpy.calc.relative_humidity_from_dewpoint(localtime_vars['2t'] * tas_unit, localtime_vars['2d'] * dewpoint_unit)
    # localtime_vars['hurs'] = localtime_vars['hurs'].where((localtime_vars['hurs'] < 1 ),1) * 100

    hurs_unit="%"

    print("computed relative humidity",flush=True)
     
    ########################
    # convert tas to degC  #
    ########################

    localtime_vars["tas"] =  localtime_vars["2t"] - 273.15
    tas_unit = "C"

    localtime_vars_xr=xr.Dataset(
            data_vars=dict(
                uas=(["time", "idx"], localtime_vars["10u"]),
                vas=(["time", "idx"], localtime_vars["10v"]),
                wind_speed=(["time", "idx"], localtime_vars["wind_speed"]),
                hurs=(["time", "idx"], localtime_vars["hurs"]),
                pr=(["time", "idx"], localtime_vars["pr"]),
                tas=(["time", "idx"], localtime_vars["tas"]),
                timeshift=(["time", "idx"], localtime_vars["timeshift"]),
                ),
            coords={
               "time": ds_reg.sel(time=slice(dateout_min, dateout_max)).time,
               },
            )


    ######################################
    # interpolate data to lon-lat grid   #
    ######################################

    inlon = np.asarray(ds_reg.lon)
    inlat = np.asarray(ds_reg.lat)
    outlon = np.arange(lon_min,lon_max + res_out_x, res_out_x, dtype=float)
    outlat = np.arange(lat_min,lat_max + res_out_y, res_out_y, dtype=float)

    varlist = ["tas","pr","wind_speed","hurs"] # ,"timeshift"]
    poolsize = min(len(varlist),maxpoolsize)

    manager=Manager()
    outvars=manager.dict()

    print("time shape:", ds_reg.sel(time=slice(dateout_min,dateout_max)).time.shape)
    print("outlon shape",outlon.shape)
    print("outlat shape",outlat.shape)
    for var in varlist:
        print(var, "shape",localtime_vars_xr[var].shape)

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

#    outvars['hurs'] = np.where((outvars_interpolated['hurs'] < 100),outvars_interpolated['hurs'],100)

    # create output file

    write_outfile()

if __name__ == "__main__":
    main()
