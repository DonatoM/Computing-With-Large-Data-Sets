import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from datetime import datetime
from scipy.stats import f_oneway
from sklearn import cross_validation

W = pd.read_csv('data/weather.csv')
T = pd.read_csv('data/random_rides.csv')


# join all the data into one matrix
def fix_date(r):
    # because weather data dates look like "2012-1-1" instead of "2012-01-01"
    s = r['EST']
    return "%s-%s-%s" % tuple([e.zfill(2) for e in s.split('-')])

T['date'] = T.apply(lambda r: r['trip_data_pickup_datetime'].split()[0], axis=1)
W['date'] = W.apply(fix_date, axis=1)
X = pd.merge(T, W, how="inner", on=["date"])
X.rename(columns={
        'trip_data_medallion': 'medallion',
        'trip_data_pickup_datetime': 'start_time',
        'trip_data_dropoff_datetime': 'end_time',
        'trip_data_passenger_count': 'num_passengers',
        'trip_data_trip_time_in_secs': 'duration_in_secs',
        'trip_data_trip_distance': 'distance',
        'trip_data_pickup_longitude': 'start_lng',
        'trip_data_pickup_latitude': 'start_lat',
        'trip_data_dropoff_longitude': 'end_lng',
        'trip_data_dropoff_latitude': 'end_lat',
        'trip_fare_fare_amount': 'fare',
        'trip_fare_payment_type': 'payment_type',
        'trip_fare_surcharge': 'surcharge',
        'trip_fare_mta_tax': 'mta_tax',
        'trip_fare_tip_amount': 'tip',
        'trip_fare_tolls_amount': 'tolls',
        'trip_fare_total_amount': 'total_fare',
        'Max TemperatureF': 'max_temp',
        'Mean TemperatureF': 'mean_temp',
        'Min TemperatureF': 'min_temp',
        'Max Dew PointF': 'max_dew_point',
        'MeanDew PointF': 'mean_dew_point', 
        'Min DewpointF' : 'min_dew_point',
        'Max Humidity' : 'max_humidity',
        ' Mean Humidity' : 'mean_humidity',
        ' Min Humidity' : 'min_humidity',
        ' Max Sea Level PressureIn' : 'max_sea_level_pressure',
        ' Mean Sea Level PressureIn' : 'mean_sea_level_pressure',
        ' Min Sea Level PressureIn' : 'min_sea_level_pressure',
        ' Max VisibilityMiles' : 'max_visibility',
        ' Mean VisibilityMiles' : 'mean_visibility',
        ' Min VisibilityMiles' : 'min_visibility',
        ' Max Wind SpeedMPH' : 'max_wind_speed',
        ' Mean Wind SpeedMPH' : 'min_wind_speed',
        ' Max Gust SpeedMPH' : 'max_gust_speed',
        'PrecipitationIn' : 'precipitation',
        ' CloudCover' : 'cloud_cover',
        ' Events': 'events',
        ' WindDirDegrees' : 'wind_dir',
    },
    inplace=True
)

# save this data for google maps api
with open('start_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    for i,r in X[['start_lat','start_lng']].iterrows():
        if r['start_lat'] > 40 and r['start_lng'] < 73:
            lats.append(r['start_lat'])
            lngs.append(r['start_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['start_lat'], r['start_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))
# more spread out, no 'red' spots
# airport, manhattan mostly

with open('stop_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    for i,r in X[['end_lat','end_lng']].iterrows():
        if r['end_lat'] > 40 and r['end_lng'] < 73:
            lats.append(r['end_lat'])
            lngs.append(r['end_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['end_lat'], r['end_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))
# more stops in outer boroughs
# red at herald sq

with open('stop_no_tip_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    for i,r in X[X['tip'] == 0][['end_lat','end_lng']].iterrows():
        if r['end_lat'] > 40 and r['end_lng'] < 73:
            lats.append(r['end_lat'])
            lngs.append(r['end_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['end_lat'], r['end_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))

with open('stop_with_tip_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    for i,r in X[X['tip'] > 0][['end_lat','end_lng']].iterrows():
        if r['end_lat'] > 40 and r['end_lng'] < 73:
            lats.append(r['end_lat'])
            lngs.append(r['end_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['end_lat'], r['end_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))

with open('stop_short_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    # look for trips < 10 min
    for i,r in X[X['duration_in_secs'] < 10*60][['end_lat','end_lng']].iterrows():
        if r['end_lat'] > 40 and r['end_lng'] < 73:
            lats.append(r['end_lat'])
            lngs.append(r['end_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['end_lat'], r['end_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))
# mostly in manhattan

with open('stop_long_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    # look for trips > 20 min
    for i,r in X[X['duration_in_secs'] > 20*60][['end_lat','end_lng']].iterrows():
        if r['end_lat'] > 40 and r['end_lng'] < 73:
            lats.append(r['end_lat'])
            lngs.append(r['end_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['end_lat'], r['end_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))
# note as much data, mostly to airports

with open('stop_52_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    for i,r in X[X['fare'] == 52][['end_lat','end_lng']].iterrows():
        if r['end_lat'] > 40 and r['end_lng'] < 73:
            lats.append(r['end_lat'])
            lngs.append(r['end_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['end_lat'], r['end_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))
# where are the trips going that are exactly $52? JFK

with open('start_52_coordinates.txt', 'w') as f:
    lats = []
    lngs = []
    for i,r in X[X['fare'] == 52][['start_lat','start_lng']].iterrows():
        if r['start_lat'] > 40 and r['start_lng'] < 73:
            lats.append(r['start_lat'])
            lngs.append(r['start_lng'])
            # otherwise it's probably just 0.0, 0.0
            line = "  new google.maps.LatLng(%s, %s),\n" % (r['start_lat'], r['start_lng'])
            f.write(line)
    f.write("mean: (%s, %s)\n" % (np.mean(lats), np.mean(lngs)))
# still JFK


# check if all no-tip rides are cash paid
float(len(X[X.tip == 0][X.payment_type == 'CSH']['payment_type'])) / len(X[X.tip == 0]['payment_type'])
# 0.96

float(len(X[X.tip > 0][X.payment_type == 'CRD']['payment_type'])) / len(X[X.tip > 0]['payment_type'])
# 0.998