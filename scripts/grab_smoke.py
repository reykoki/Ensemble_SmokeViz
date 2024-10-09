import matplotlib
from matplotlib import pyplot as plt
import sys
from matplotlib.lines import Line2D
import geopandas
import pandas as pd
import os
import random
import glob
import time
from datetime import datetime
import pytz
import shutil
import wget
from suntime import Sun
from datetime import timedelta
from datetime import date
from helper_functions import *

data_dir = './data/'

def get_smoke_fn_url(dt):
    tt = dt.timetuple()
    month = str(tt.tm_mon).zfill(2)
    day = str(tt.tm_mday).zfill(2)
    yr = str(tt.tm_year)
    fn = 'hms_smoke{}{}{}.zip'.format(yr, month, day)
    url = 'https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/{}/{}/{}'.format(yr, month, fn)
    return fn, url

def get_idx(smoke_shape, dt):
    use_idx = []
    start_ends = []
    if smoke_shape is not None:
        fmt = '%Y%j %H%M'

        state_shape = data_dir + 'shape_files/contiguous_states.shp'
        states = geopandas.read_file(state_shape)
        states = states.to_crs(smoke_shape.crs)
        clip_smoke = smoke_shape.clip(states)
        indices_in_us = list(clip_smoke.index.values)
        smoke = smoke_shape.loc[indices_in_us]
        #print(smoke)
        for idx, row in smoke.iterrows():
            start = pytz.utc.localize(datetime.strptime(smoke.loc[idx]['Start'], fmt))
            end = pytz.utc.localize(datetime.strptime(smoke.loc[idx]['End'], fmt))
            start_ends.append((start, end))

            if dt-timedelta(minutes=10)<= end and start-timedelta(minutes=10) <= dt:
                use_idx.append(idx)
    print('\nthere are {} smoke annotations in the time period requested'.format(len(use_idx)))
    if use_idx:
        smoke = smoke_shape.loc[use_idx]
    else:
        print_start_ends(start_ends)
    return smoke, use_idx

def get_smoke(dt, entire_day=False):
    fn, url = get_smoke_fn_url(dt)
    print('DOWNLOADING SMOKE:')
    print(fn)
    out_dir = data_dir + 'smoke/'
    smoke_shape_fn = out_dir + fn
    print(smoke_shape_fn)
    use_idx = []
    if os.path.exists(out_dir+fn):
        print("{} already exists".format(fn))
    else:
        filename = wget.download(url, out=out_dir)
        shutil.unpack_archive(filename, out_dir)
    smoke = geopandas.read_file(smoke_shape_fn)
    if entire_day is False:
        smoke, use_idx = get_idx(smoke, dt)
    return smoke, use_idx

def get_sunrise_sunset(dt):
    west_lon = -124.8
    west_lat = 24.5
    east_lon = -71.1
    east_lat = 45.93
    east = Sun(east_lat, east_lon)
    west = Sun(west_lat, west_lon)
    sunset = east.get_sunset_time(dt)
    sunrise = west.get_sunrise_time(dt)
    if sunrise > sunset:
        sunset = west.get_sunset_time(dt + timedelta(days=1))
    return sunrise, sunset

def sun_out(dt):
    sunrise, sunset = get_sunrise_sunset(dt)
    print(sunrise.strftime("sunrise on the west coast is at %H:%M UTC for %Y/%m/%d"))
    print(sunset.strftime("sunset on the east coast is at %H:%M UTC for %Y/%m/%d"))
    if dt < sunrise:
        raise ValueError(sunrise.strftime("choose a time after sunrise \nsunrise on the west coast is at %H:%M UTC for %Y/%m/%d"))
    if dt > sunset:
        raise ValueError(sunset.strftime("choose a time before sunset \nsunset on the east coast is at %H:%M UTC for %Y/%m/%d"))
    return

def within_time_bounds(dt):
    t_min = pytz.utc.localize(datetime.strptime('2018', '%Y'))
    today = date.today()
    today_dt = pytz.utc.localize(datetime(today.year, today.month, today.day))
    if dt < t_min:
        Exception('please choose a date after January 1st, 2018')
    if dt > today_dt:
        Exception('please choose a date before today')
    return

def print_start_ends(start_ends):
    start_ends = set(start_ends)
    print('to show smoke annotations, choose a time within the following time bounds:')
    for s_e in start_ends:
        start_str = s_e[0].strftime("%Y/%m/%d %H:%M UTC")
        end_str = s_e[1].strftime("%Y/%m/%d %H:%M UTC")
        print('{} to {}'.format(start_str, end_str))

def plot_smoke_conus(smoke):
    state_shape = data_dir + 'shape_files/contiguous_states.shp'
    states = geopandas.read_file(state_shape)
    states = states.to_crs(smoke.crs)
    clip_smoke = smoke.clip(states)
    indices_in_us = list(clip_smoke.index.values)
    smoke = smoke.loc[indices_in_us]

    low_smoke = smoke.loc[smoke['Density'] == 'Light']
    med_smoke = smoke.loc[smoke['Density'] == 'Medium']
    high_smoke = smoke.loc[smoke['Density'] == 'Heavy']
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1)

    states.boundary.plot(ax=ax, edgecolor='black', linewidth=.25)
    if len(high_smoke)>0:
        high_smoke.plot(ax=ax, column='Density', categorical=True, legend=True, facecolor='none', edgecolor='red', linewidth=2.5)
    if len(med_smoke)>0:
        med_smoke.plot(ax=ax, facecolor='none', edgecolor='orange', linewidth=2.5)
    if len(low_smoke)>0:
        low_smoke.plot(ax=ax, facecolor='none', edgecolor='yellow', linewidth=2.5)
    custom_lines = [Line2D([0], [0], color='yellow', lw=3),
                                    Line2D([0], [0], color='orange', lw=3),
                                    Line2D([0], [0], color='red', lw=3)]
    ax.legend(custom_lines, ['Light', 'Medium', 'Heavy'], title='Smoke Density', loc=3)
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig('densities.png')
    plt.show()
    return smoke

# analysts can only label data that is taken during the daytime, we want to filter for geos data that was within the timeframe the analysts are looking at
def download_smoke(dt, entire_day=False):
    within_time_bounds(dt)
    if entire_day is False:
        sun_out(dt)
    smoke, use_idx = get_smoke(dt, entire_day)
    if len(use_idx) > 0 or entire_day is True:
        smoke = plot_smoke_conus(smoke)
    return smoke

def main(dt):
    smoke_shape = download_smoke(dt)
    return smoke_shape

if __name__ == '__main__':
    input_dt = sys.argv[1]
    dt = get_dt_from_str(input_dt)
    smoke = main(dt)
