import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from datetime import datetime
from scipy.stats import f_oneway
from sklearn import cross_validation

W = pd.read_csv('data/weather.csv')
T = pd.read_csv('data/random_rides.csv')

X = T[['trip_fare_fare_amount', 'trip_fare_tip_amount']]
X.columns = ['fare', 'tip']

plt.scatter(X['fare'], X['tip'])
plt.xlabel('Fare')
plt.ylabel('Tip')
plt.title('Is there a Relationship between Fare & Tip?')
plt.show()

print "percent of people who don't tip:", len(X[X.tip == 0])/float(len(X))



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
        ' Events': 'events',
    },
    inplace=True
)

### avg speed
def avg_speed(r):
    if r['duration_in_secs'] > 0:
        return float(r['distance']) / (r['duration_in_secs']/3600.0)
    else:
        return np.nan

X['avg_speed'] = X.apply(avg_speed, axis=1)
plt.hist(X['avg_speed'].dropna(), bins=50) # problem!
plt.xlabel('Average Speed')
plt.ylabel('Frequency')
plt.title('Speed Frequencies With Outlier')
plt.show()


def dist(lat1, lng1, lat2, lng2):
    # Haversine formula
    phi1 = lat1*math.pi/180
    phi2 = lat2*math.pi/180
    dphi = phi2 - phi1
    dlam = (lng2 - lng1)*math.pi/180

    a = math.sin(dphi/2)*math.sin(dphi/2) + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)*math.sin(dlam/2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return c*6371 #3963.1676

def dist2(r):
    return dist(r.start_lat, r.start_lng, r.end_lat, r.end_lng)

X['distance2'] = X.apply(dist2, axis=1)

print("first")
plt.hist(X[X.avg_speed < 100]['avg_speed'].dropna(), bins=50)
med = np.median(X[X.avg_speed < 100]['avg_speed'].dropna()) ######## 2014-12-13
plt.axvline(med, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Average Speed')
plt.ylabel('Frequency')
plt.title('Speed Frequencies Without Outlier')
plt.show()


# speed does not vary with temperature
X_withspeed = X[X.avg_speed.notnull()]
X_withspeed = X_withspeed[X_withspeed.avg_speed < 100]
plt.scatter(X_withspeed['mean_temp'], X_withspeed['avg_speed'])
plt.xlabel('Temperature')
plt.ylabel('Speed')
plt.title('Does Temperature Affect Speed')
plt.show()

# speed vs. fog, rain, snow etc.
event_codes = [
    'Fog',
    'Fog-Rain',
    'Fog-Rain-Snow',
    'Fog-Snow',
    'Rain',
    'Rain-Snow',
    'Snow'
]
event_codes_dict = {
    'Fog' : 0,
    'Fog-Rain' : 1,
    'Fog-Rain-Snow' : 2,
    'Fog-Snow' : 3,
    'Rain' : 4,
    'Rain-Snow' : 5,
    'Snow' : 6
}
event_codes_lst = ['Normal','Fog','Fog-Rain','Fog-Rain-Snow','Fog-Snow','Rain','Rain-Snow','Snow']
X['event_code'] = X.apply(lambda r: event_codes_dict.get(r['events'], -1), axis=1)

Xf = X[X.avg_speed.notnull()]
Xf = Xf[Xf.avg_speed < 100]
plt.scatter(Xf['event_code'], Xf['avg_speed'])
plt.xticks([-1,0,1,2,3,4,5,6,7], event_codes_lst, rotation='vertical')
plt.xlabel('Events')
plt.ylabel('Speed (MPH)')
plt.title('Does Speed Change With Weather?')
plt.tight_layout()
plt.show()

vectors = [Xf[Xf.events.isnull()]['avg_speed']]
for event in event_codes:
    vectors.append(Xf[Xf.events == event]['avg_speed'])
plt.boxplot(vectors)
plt.xticks([1,2,3,4,5,6,7,8], event_codes_lst, rotation='vertical')
plt.xlabel('Events')
plt.ylabel('Speed (MPH)')
plt.title('Does Speed Change With Weather?')
plt.tight_layout()
plt.show()


# naive way of looking at tip vs num passengers
plt.scatter(X['num_passengers'], X['tip'])
plt.xlabel('Number of Passengers')
plt.ylabel('Tip')
plt.title('Tip vs Number of Passengers')
plt.show()
print("second")
# 2014-12-13
vectors = []
for n in set(X['num_passengers']):
    vectors.append(X[X['num_passengers'] == n]['tip'])
plt.boxplot(vectors)
plt.xlabel('Number of Passengers')
plt.ylabel('Tip')
plt.title('Tip vs Number of Passengers')
plt.show()

# without outliers
vectors = []
for n in set(X['num_passengers']):
    vectors.append(X[X['num_passengers'] == n][X['tip'] < 30]['tip'])
plt.boxplot(vectors)
plt.xlabel('Number of Passengers')
plt.ylabel('Tip')
plt.title('Tip vs Number of Passengers Without Outliers')
plt.show()

print("third")
# distribution of tip amounts
plt.hist(X['tip'])
med = np.median(X['tip']) ######## 2014-12-13
plt.axvline(med, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Amount in ($)')
plt.ylabel('Frequency')
plt.title('Tip Amount Distribution')
plt.show()

# tip percentage vs num passengers
plt.scatter(X['num_passengers'], X['tip']/X['fare'])
plt.xlabel('Number of Passengers')
plt.ylabel('Tip %')
plt.title('Tip % vs. Number of Passengers')
plt.show()

# do one-way anova between avg speed during
# different weather events
# fog-snow has less speed
for i,v in enumerate(vectors[1:]):
    fval, pval = f_oneway(vectors[0], v)
    print event_codes[i], pval

from sklearn.linear_model import LinearRegression, LogisticRegression

clf = LinearRegression()
def fix_precipitation(r):
    try:
        return float(r['PrecipitationIn'])
    except:
        return -1

X['precipitation'] = X.apply(fix_precipitation, axis=1)
Z = X[['duration_in_secs', 'tip', 'fare', 'num_passengers', 'distance', u'max_temp', u'mean_temp', u'min_temp', u'max_dew_point', u'mean_dew_point', u'Min DewpointF', u'Max Humidity', u' Mean Humidity', u' Min Humidity', u' Max Sea Level PressureIn', u' Mean Sea Level PressureIn', u' Min Sea Level PressureIn', u' Max VisibilityMiles', u' Mean VisibilityMiles', u' Min VisibilityMiles', u' Max Wind SpeedMPH', u' Mean Wind SpeedMPH', u' Max Gust SpeedMPH', 'precipitation', u' CloudCover']]
Z = Z[Z.max_temp.notnull()]
Z = Z[Z.mean_temp.notnull()]
Z = Z[Z.min_temp.notnull()]
Z = Z[Z.max_dew_point.notnull()]
Z = Z[Z.mean_dew_point.notnull()]
Z = Z[Z['Min DewpointF'].notnull()]
Z = Z[Z['Max Humidity'].notnull()]
Z = Z[Z[' Mean Humidity'].notnull()]
Z = Z[Z[' Min Humidity'].notnull()]
Z = Z[Z[' Max Sea Level PressureIn'].notnull()]
Z = Z[Z[' Mean Sea Level PressureIn'].notnull()]
Z = Z[Z[' Min Sea Level PressureIn'].notnull()]
Z = Z[Z[' Max VisibilityMiles'].notnull()]
Z = Z[Z[' Mean VisibilityMiles'].notnull()]
Z = Z[Z[' Min VisibilityMiles'].notnull()]
Z = Z[Z[' Max Wind SpeedMPH'].notnull()]
Z = Z[Z[' Mean Wind SpeedMPH'].notnull()]
Z = Z[Z[' Max Gust SpeedMPH'].notnull()]
Z = Z[Z[' CloudCover'].notnull()]
Z = Z[Z.fare > 0]
Z['ones'] = 1

z = Z[[u'max_temp', 'ones', 'num_passengers', u'mean_temp', u'min_temp', u'max_dew_point', u'mean_dew_point', u'Min DewpointF', u'Max Humidity', u' Mean Humidity', u' Min Humidity', u' Max Sea Level PressureIn', u' Mean Sea Level PressureIn', u' Min Sea Level PressureIn', u' Max VisibilityMiles', u' Mean VisibilityMiles', u' Min VisibilityMiles', u' Max Wind SpeedMPH', u' Mean Wind SpeedMPH', u' Max Gust SpeedMPH', 'precipitation', u' CloudCover']]
clf.fit(z, Z['duration_in_secs'])
clf.score(z, Z['duration_in_secs'])

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
clf = DecisionTreeRegressor()
clf.fit(z, Z['duration_in_secs'])
clf.score(z, Z['duration_in_secs'])


# reduce dimensionality and try to use the classifier
# score is still not good
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(z)
plt.plot(pca.explained_variance_ratio_)
plt.show()

zt = pca.fit_transform(z)
clf.fit(zt[:,:2],Z['duration_in_secs'])
clf.score(zt[:,:2],Z['duration_in_secs'])


# can we predict tip %?
clf.fit(z, Z['tip']/Z['fare'])
clf.score(z, Z['tip']/Z['fare'])

print("fourth")
# histogram of distances
plt.hist(Z['distance'])
med = np.median(Z['distance']) ######## 2014-12-13
plt.axvline(med, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Distance (KM)')
plt.ylabel('Frequency')
plt.title('Distance Traveled Frequency')
plt.show()

# can we predict distance of trip?
clf.fit(z, Z['distance'])
clf.score(z, Z['distance'])

# can we predict fare from distance? from duration?
plt.scatter(Z['distance'], Z['fare'])
plt.xlabel('Distance (KM)')
plt.ylabel('Fare')
plt.title('Distance vs. Fare')
plt.show()

Z['ones'] = 1
clf.fit(Z[['distance', 'ones']], Z['fare'])
print "distance vs. fare:", clf.score(Z[['distance', 'ones']], Z['fare'])

clf.fit(Z[['duration_in_secs', 'ones']], Z['fare'])
print "duration vs. fare:", clf.score(Z[['duration_in_secs', 'ones']], Z['fare'])


# try to predict whether or not a person will tip
clf = LogisticRegression()
Z['has_tip'] = Z.apply(lambda r: r['tip'] > 0, axis=1)
clf.fit(zt, Z['has_tip'])
print "transformed data vs. has tip using logistic reg:", clf.score(zt, Z['has_tip'])

clf = DecisionTreeClassifier()
clf.fit(zt, Z['has_tip'])
print "transformed data vs. has tip using d tree:", clf.score(zt, Z['has_tip'])
print "transformed data vs. has tip using d tree and cross-val:", cross_validation.cross_val_score(clf, zt, Z['has_tip'], cv=5)

# see if there is a pattern between number of passengers and whether or not they'll tip
print "num passengers vs. has tip using d tree and cross-val:", cross_validation.cross_val_score(clf, z[['num_passengers', 'ones']], Z['has_tip'], cv=5)


from sklearn.ensemble import RandomForestClassifier
# try random forest classifier to see if it performs better than decision tree
print "transformed data vs. has tip using random forest and cross-val:", cross_validation.cross_val_score(RandomForestClassifier(), zt, Z['has_tip'], cv=5)


# can we predict if a person will tip based on their pick up location or drop off location?
X['has_tip'] = X.apply(lambda r: r['tip'] > 0, axis=1)
print "start pos vs. has tip using random forest and cross-val:", cross_validation.cross_val_score(RandomForestClassifier(), X[['start_lat', 'start_lng']], X['has_tip'], cv=5)
print "end pos vs. has tip using random forest and cross-val:", cross_validation.cross_val_score(RandomForestClassifier(), X[['end_lat', 'end_lng']], X['has_tip'], cv=5)


# can we predict if a person will tip or not based on payment type?
X['ones'] = 1
payment_code_dict = {
    'CRD':0,
    'CSH':1,
    'DIS':2,
    'NOC':3,
    'UNK':4,
}
X['payment_code'] = X.apply(lambda r: payment_code_dict.get(r['payment_type'], -1), axis=1)
print "payment code vs. has tip using d tree and cross-val:", cross_validation.cross_val_score(DecisionTreeClassifier(), X[['payment_code','ones']], X['has_tip'], cv=5)

# weather event v tip%
plt.scatter(X['event_code'], X['tip']/X['fare'])
plt.xticks([-1,0,1,2,3,4,5,6,7,8], event_codes_lst, rotation='vertical')
plt.xlabel('Events')
plt.ylabel('Tip %')
plt.title('Tip % vs. Weather Event')
plt.tight_layout()
plt.show()
print("fifth")
######## 2014-12-13
from matplotlib import cm
H = pd.DataFrame(np.array([X['event_code'], X['tip']/X['fare']]).T, columns=['ev', 'tipp'])
H = H[H.tipp.notnull()]
plt.hexbin(H['ev'], H['tipp'], gridsize=30, cmap=cm.jet, bins=None)
plt.axis([H['ev'].min(), H['ev'].max(), H['tipp'].min(), H['tipp'].max()])
cb = plt.colorbar()
plt.xticks([-1,0,1,2,3,4,5,6], event_codes_lst, rotation='vertical')
plt.xlabel('Events')
plt.ylabel('Tip %')
plt.title('Tip % vs. Weather Event')
plt.tight_layout()
plt.show()  

X['tip_percent'] = X['tip']/X['fare']
vectors = [X[X.events.isnull()]['tip_percent']]
for event in event_codes:
    vectors.append(X[X.events == event]['tip_percent'])
plt.boxplot(vectors)
plt.xticks([1,2,3,4,5,6,7,8], event_codes_lst, rotation='vertical')
plt.xlabel('Events')
plt.ylabel('Tip %')
plt.title('Tip % vs. Weather Event')
plt.tight_layout()
plt.show()

# weather event v num passengers
plt.scatter(X['event_code'], X['num_passengers'])
plt.xticks([-1,0,1,2,3,4,5,6,7,8], event_codes_lst, rotation='vertical')
plt.xlabel('Events')
plt.ylabel('Number of Passengers')
plt.title('Does the number of passengers change with weather?')
plt.tight_layout()
plt.show()

vectors = [X[X.events.isnull()]['num_passengers']]
for event in event_codes:
    vectors.append(X[X.events == event]['num_passengers'])
plt.boxplot(vectors)
plt.show()

plt.hist(X[X['events'].isnull()]['num_passengers'].as_matrix())
plt.title("Number of  Passengers, event = None")
plt.xlabel('Number of Passengers')
plt.ylabel('Number of Rides')
plt.show()

for event in event_codes:
    plt.hist(X[X['events'] == event]['num_passengers'].as_matrix())
    plt.title("Number of Passengers, event = %s" % event)
    plt.xlabel('Number of Passengers')
    plt.ylabel('Number of Rides')
    plt.show()