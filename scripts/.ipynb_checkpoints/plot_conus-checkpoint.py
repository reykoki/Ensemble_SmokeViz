import pandas as pd
import sys
import pytz
from pyresample import create_area_def
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from satpy import Scene

data_dir = './data/'

def check_bounds(x, y, bounds):
    if bounds['minx'] > np.min(x) and bounds['maxx'] < np.max(x) and bounds['miny'] > np.min(y) and bounds['maxy'] < np.max(y):
        return True
    else:
        return False

def get_norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def pick_out_shapes(smoke_shape, t_0, t_f, x, y, lcc_proj):
    #print(smoke_shape)
    high_idx = []
    med_idx = []
    low_idx = []
    bounds = smoke_shape.bounds
    fmt = '%Y%j %H%M'
    for idx, row in smoke_shape.iterrows():
        start = pytz.utc.localize(datetime.strptime(row['Start'], fmt))
        end = pytz.utc.localize(datetime.strptime(row['End'], fmt))
        # the ranges overlap if:
        in_bounds = check_bounds(x, y, bounds.loc[idx])
        d = row['Density']
        if in_bounds and t_0 <= end and start <= t_f:
            if d == 'Light':
                low_idx.append(idx)
            elif d == 'Medium':
                med_idx.append(idx)
            elif d == 'Heavy':
                high_idx.append(idx)
    high_smoke = smoke_shape.loc[high_idx]
    med_smoke = smoke_shape.loc[med_idx]
    low_smoke = smoke_shape.loc[low_idx]
    high_smoke = high_smoke.to_crs(lcc_proj)
    med_smoke = med_smoke.to_crs(lcc_proj)
    low_smoke = low_smoke.to_crs(lcc_proj)
    return high_smoke, med_smoke, low_smoke

def get_dt_str(dt):
    hr = dt.hour
    hr = str(hr).zfill(2)
    tt = dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    yr = dt.year
    return hr, dn, yr

def get_dt(input_dt):
    fmt = '%Y/%m/%d %H:%M'
    dt = datetime.strptime(input_dt, fmt)
    dt = pytz.utc.localize(dt)
    return dt

def get_fns(dt):
    hr, dn, yr = get_dt_str(dt)
    goes_dir = data_dir + 'goes_temp/'
    fns = glob(goes_dir + '*C0[123]*_s{}{}{}*'.format(yr,dn,hr))
    print(fns)
    return fns

def get_rgb(corr_data):
    # corrected is in RGB order
    R = corr_data[0][0]
    G = corr_data[0][1]
    B = corr_data[0][2]
    R = get_norm(R)
    G = get_norm(G)
    B = get_norm(B)
    RGB = np.dstack([R, G, B])
    return RGB



def plot_data(input_dt):
    dt = get_dt(input_dt)
    fns = get_fns(dt)

    scn = Scene(reader='abi_l1b', filenames=fns)

    composite = 'cimss_true_color_sunz_rayleigh'
    scn.load([composite])

    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                              resolution=500,
                              #entire US
                              area_extent=[-2.4e6, -1.5e6, 2.3e6, 1.4e6],
                              nprocs=4)

    new_scn = scn.resample(my_area, cache_dir='cache_dir')
    lcc_proj = new_scn[composite].attrs['area'].to_cartopy_crs()
    geos_proj = scn['C02'].attrs['area'].to_cartopy_crs()
    #print(geos_proj)

    corr_data = new_scn.save_dataset('cimss_true_color_sunz_rayleigh', compute=False)
    RGB = get_rgb(corr_data)



    scan_start = new_scn[composite].attrs['start_time']
    scan_end = new_scn[composite].attrs['end_time']
    print('start of satellite scan: ', scan_start)
    print('end of satellite scan: ', scan_end)

    smoke_shape_fn = data_dir + 'smoke/hms_smoke{:%Y%m%d}.shp'.format(scan_start)

    state_shape = './data/shape_files/cb_2018_us_state_500k.shp'

    states = geopandas.read_file(state_shape)

    smoke = geopandas.read_file(smoke_shape_fn)

    states = states.to_crs(geos_proj)

    smoke = smoke.to_crs(geos_proj)
    #t_0 = int("{:%H%M}".format(scan_start))
    #t_f = int("{:%H%M}".format(scan_end))
    t_0 = pytz.utc.localize(scan_start)
    t_f = pytz.utc.localize(scan_end)
    yr_day = int("{:%Y%j}".format(scan_end))
    geos_x = scn['C02'].coords['x']
    geos_y = scn['C02'].coords['y']
    high_smoke, med_smoke, low_smoke = pick_out_shapes(smoke, t_0, t_f, geos_x, geos_y, lcc_proj)

    states = states.to_crs(lcc_proj)

    x = new_scn[composite].coords['x']
    y = new_scn[composite].coords['y']

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)

    ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    states.boundary.plot(ax=ax, edgecolor='white', linewidth=.25)
    high_smoke.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.5)
    med_smoke.plot(ax=ax, facecolor='none', edgecolor='orange', linewidth=2.5)
    low_smoke.plot(ax=ax, facecolor='none', edgecolor='yellow', linewidth=2.5)

    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig('conus.png')

    plt.show()

def main(input_dt):
    plot_data(input_dt)

if __name__ == '__main__':
    input_dt = sys.argv[1]
    main(input_dt)
