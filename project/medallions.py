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

#################### build some more features #####################
def get_hour(r):
    dt = datetime.strptime(r['start_time'], "%Y-%m-%d %H:%M:%S")
    return dt.hour


X['hour_of_day'] = X.apply(get_hour, axis=1)

# standardize lng/lat
start_lng_u = np.mean(X['start_lng'])
start_lng_s = np.std(X['start_lng'])
X['std_start_lng'] = X.apply(lambda r: (r['start_lng'] - start_lng_u) / start_lng_s, axis=1)

end_lng_u = np.mean(X['end_lng'])
end_lng_s = np.std(X['end_lng'])
X['std_end_lng'] = X.apply(lambda r: (r['end_lng'] - end_lng_u) / end_lng_s, axis=1)

start_lat_u = np.mean(X['start_lat'])
start_lat_s = np.std(X['start_lat'])
X['std_start_lat'] = X.apply(lambda r: (r['start_lat'] - start_lat_u) / start_lat_s, axis=1)

end_lat_u = np.mean(X['end_lat'])
end_lat_s = np.std(X['end_lat'])
X['std_end_lat'] = X.apply(lambda r: (r['end_lat'] - end_lat_u) / end_lat_s, axis=1)

# standardize weather vars
weather_vars = ('max_temp','mean_temp','min_temp','max_dew_point','mean_dew_point', 'min_dew_point','max_humidity',
            'mean_humidity','min_humidity','max_sea_level_pressure','mean_sea_level_pressure','min_sea_level_pressure',
            'max_visibility', 'mean_visibility','min_visibility','max_wind_speed','min_wind_speed',
            'max_gust_speed','cloud_cover')
for col in weather_vars:
    try:
        u = np.mean(X[col])
        s = np.std(X[col])
        X["std_" + col] = X.apply(lambda r: (r[col] - u) / s, axis=1)
    except:
        print "BAD COLUMN:", col

# is mostly numeric except for a weird T, set to numerical value
X['precipitation'] = X.apply(lambda r: -1 if r['precipitation'] == 'T' else r['precipitation'], axis=1)


# num passengers v bad weather

# tip v bad weather

# medallion total fare distrib
G = pd.read_csv('data/medallion_income.csv')
plt.hist(G['Total'], bins=100)
plt.xlabel('Amount Made ($)')
plt.ylabel('Frequency')
plt.title('How much money did each medallion make?')
plt.show()

# totals close to 0
plt.hist(G[G['Total'] < 10000]['Total'].as_matrix(), bins=100)
plt.show()

plt.hist(G[G['Total'] < 400000]['Total'].as_matrix(), bins=100)
plt.xlabel('Amount Made ($)')
plt.ylabel('Frequency')
plt.title('How much money did each medallion make?')
plt.show()