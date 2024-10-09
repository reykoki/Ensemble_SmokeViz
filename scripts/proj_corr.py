import pandas as pd
import matplotlib.gridspec as gridspec
from skimage.transform import resize
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
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def pick_out_shapes(smoke_shape, t_0, t_f, x, y, lcc_proj):
    print(smoke_shape)
    high_idx = []
    med_idx = []
    low_idx = []
    bounds = smoke_shape.bounds
    for idx, row in smoke_shape.iterrows():
        if row['Start'][:6] == row['End'][:6]:
            start = int(row['Start'][-4:])
            end = int(row['End'][-4:])
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

def plot_corr_data(fns):

    scn = Scene(reader='abi_l1b', filenames=fns)
    composite = 'cimss_true_color_sunz_rayleigh'
    scn.load([composite])
    geos_proj = scn['C02'].attrs['area'].to_cartopy_crs()
    new_scn = scn.resample(scn.coarsest_area(), resampler='native')
    corr_data = new_scn.save_dataset(composite, compute=False)
    RGB = get_rgb(corr_data)

    x = scn['C03'].coords['x']
    y = scn['C03'].coords['y']
    return RGB, x, y

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=geos_proj)
    ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.show()

def plot_proj_data(fns):

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
    corr_data = new_scn.save_dataset(composite, compute=False)
    RGB = get_rgb(corr_data)
    lcc_proj = new_scn[composite].attrs['area'].to_cartopy_crs()
    x = new_scn[composite].coords['x']
    y = new_scn[composite].coords['y']
    return RGB, x, y, lcc_proj
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)
    ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.show()

def plot_orig_data(input_dt):
    dt = get_dt(input_dt)
    fns = get_fns(dt)
    RGB_c, x_c, y_c = plot_corr_data(fns)
    RGB_p, x_p, y_p, lcc_proj = plot_proj_data(fns)

    scn = Scene(reader='abi_l1b', filenames=fns)
    scn.load(['C01', 'C02', 'C03'])
    R = scn['C02'].data
    G = scn['C03'].data
    B = scn['C01'].data
    R = get_norm(R)
    G = get_norm(G)
    B = get_norm(B)
    R = resize(R, G.shape)
    RGB = np.dstack([R, G, B])
    geos_proj = scn['C03'].attrs['area'].to_cartopy_crs()
    print(geos_proj)
    x = scn['C03'].coords['x']
    y = scn['C03'].coords['y']
    fig = plt.figure()
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[:, 0], projection=geos_proj)
    ax1.set_title('Raw')
    ax1.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))

    ax2 = fig.add_subplot(gs[:, 1], projection=geos_proj)
    ax2.imshow(RGB_c, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    ax2.set_title('Corrected')
    ax3 = fig.add_subplot(gs[:, 2], projection=lcc_proj)
    ax3.imshow(RGB_p, origin='upper', extent=(x_p.min(), x_p.max(), y_p.min(), y_p.max()))
    ax3.set_title('Corrected and Transformed')
    #plt.axis('off')
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = .02)
    plt.tight_layout(pad=.2)
    #plt.margins(0,0)
    plt.show()

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
    print(geos_proj)
    x = input('stop')

    corr_data = new_scn.save_dataset('cimss_true_color_sunz_rayleigh', compute=False)
    RGB = get_rgb(corr_data)

    scan_start = new_scn[composite].attrs['start_time']
    scan_end = new_scn[composite].attrs['end_time']
    print('start of satellite scan: ', scan_start)
    print('end of satellite scan: ', scan_end)

    smoke_shape_fn = data_dir + 'smoke/hms_smoke{:%Y%m%d}.shp'.format(scan_start)
    state_shape = '/home/rey/projects/smoke_seg/20201018/cb_2018_us_state_500k.shp'
    states = geopandas.read_file(state_shape)
    states = states.to_crs(geos_proj)

    t_0 = int("{:%H%M}".format(scan_start))
    t_f = int("{:%H%M}".format(scan_end))
    yr_day = int("{:%Y%j}".format(scan_end))
    geos_x = scn['C02'].coords['x']
    geos_y = scn['C02'].coords['y']

    states = states.to_crs(lcc_proj)

    x = new_scn[composite].coords['x']
    y = new_scn[composite].coords['y']

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)

    ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    #states.boundary.plot(ax=ax, edgecolor='white', linewidth=.25)

#plt.savefig('L1_proj.png')
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = .02)
    plt.tight_layout()
    plt.margins(0,0)
    plt.show()

def main(input_dt):
    plot_orig_data(input_dt)

if __name__ == '__main__':
    input_dt = sys.argv[1]
    main(input_dt)
