import pandas as pd
from datetime import timedelta
import sys
import pytz
from pyresample import create_area_def
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from satpy import Scene
import cartopy.crs as ccrs

data_dir = './data/'

def get_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                     central_latitude=38.5,
                                     standard_parallels=(38.5, 38.5),
                                     globe=ccrs.Globe(semimajor_axis=6371229,
                                                      semiminor_axis=6371229))
    return lcc_proj

def pick_temporal_smoke(smoke_shape, t_0, t_f):
    use_idx = []
    fmt = '%Y%j %H%M'
    for idx, row in smoke_shape.iterrows():
        start = pytz.utc.localize(datetime.strptime(row['Start'], fmt))
        end = pytz.utc.localize(datetime.strptime(row['End'], fmt))
        # the ranges overlap if:
        if t_0-timedelta(minutes=10)<= end and start-timedelta(minutes=10) <= t_f:
            use_idx.append(idx)
    rel_smoke = smoke_shape.loc[use_idx]
    return rel_smoke

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

def get_fns(dt):
    hr, dn, yr = get_dt_str(dt)
    goes_dir = data_dir + 'goes/'
    fns = glob(goes_dir + '*C0[123]*_s{}{}{}*'.format(yr,dn,hr))
    return fns

def get_RGB(scn, composite):
    data = scn.save_dataset(composite, compute=False)
    R = data[0][0]
    G = data[0][1]
    B = data[0][2]
    # reorder before computing for plotting
    RGB = np.dstack([R, G, B])
    RGB = RGB.compute() # computationally expensive part
    return RGB

def get_scn(fns, to_load, extent, res=3000, proj=get_proj()):
    print(fns)
    scn = Scene(filenames=fns)
    scn.load(to_load, generate=False)
    my_area = create_area_def(area_id='my_area',
                              projection=proj,
                              resolution=res,
                              area_extent=extent
                              )
    new_scn = scn.resample(my_area) # resamples datasets and resturns a new scene object
    return new_scn


def plot_data(dt):

    state_shape = './data/shape_files/cb_2018_us_state_500k.shp'
    states = geopandas.read_file(state_shape)

    lcc_proj = get_proj()
    fns = get_fns(dt)
    extent=[-2.4e6, -1.5e6, 2.3e6, 1.4e6]
    composite = ['cimss_true_color_sunz_rayleigh']
    scn = get_scn(fns,composite, extent, proj=lcc_proj)
    crs = scn[composite[0]].attrs['area'].to_cartopy_crs() # the crs object will have the area extent for plotting

    RGB = get_RGB(scn, composite[0])

    scan_start = scn[composite[0]].attrs['start_time']
    scan_end = scn[composite[0]].attrs['end_time']

    smoke_shape_fn = data_dir + 'smoke/hms_smoke{:%Y%m%d}.shp'.format(scan_start)
    smoke = geopandas.read_file(smoke_shape_fn)

    states = states.to_crs(crs)
    smoke = smoke.to_crs(crs)

    t_0 = pytz.utc.localize(scan_start)
    t_f = pytz.utc.localize(scan_end)

    smoke = pick_temporal_smoke(smoke, t_0, t_f)
    low_smoke = smoke.loc[smoke['Density'] == 'Light']
    med_smoke = smoke.loc[smoke['Density'] == 'Medium']
    high_smoke = smoke.loc[smoke['Density'] == 'Heavy']

    fig = plt.figure(figsize=(15, 12))
    ax = plt.axes(projection=crs)
    plt.imshow(RGB, transform=crs, extent=crs.bounds, origin='upper')
    states.boundary.plot(ax=ax, edgecolor='white', linewidth=.25)
    high_smoke.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.5)
    med_smoke.plot(ax=ax, facecolor='none', edgecolor='orange', linewidth=2.5)
    low_smoke.plot(ax=ax, facecolor='none', edgecolor='yellow', linewidth=2.5)
    plt.axis('off')
    fig.tight_layout()
    #plt.savefig('conus.png')
    plt.show()

def main(dt):
    plot_data(dt)

if __name__ == '__main__':
    #dt = sys.argv[1]
    dt_str = '2024/06/25 13:40'
    dt = pytz.utc.localize(datetime.strptime(dt_str, '%Y/%m/%d %H:%M'))
    main(dt)
