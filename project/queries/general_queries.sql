-- Some queries taken from http://www.reddit.com/r/bigquery/comments/28ialf/173_million_2013_nyc_taxi_rides_shared_on_bigquery

-- Retrieve the amount of trips.
SELECT COUNT(*) trips FROM [833682135931:nyctaxi.trip_data]

-- Average trip distance, average trip time, and number of trips.
SELECT AVG(trip_distance) avg_distance, AVG(trip_time_in_secs) avg_time, COUNT(*) trips FROM [833682135931:nyctaxi.trip_data]

-- Average speed at each hour of the day.
SELECT
  HOUR(TIMESTAMP(pickup_datetime)) AS hour,
  ROUND(AVG(FLOAT(trip_distance)/FLOAT(trip_time_in_secs)*60*60)) AS speed
FROM
  [833682135931:nyctaxi.trip_data]
WHERE
  INTEGER(trip_time_in_secs) > 10
  AND FLOAT(trip_distance) < 90
GROUP BY
  hour
ORDER BY
  hour;

-- Trips with no tips.
SELECT count(*) trips_with_no_tip
FROM [833682135931:nyctaxi.trip_fare]
WHERE float(tip_amount) = 1.00 and float(fare_amount) > 0.00 ;

-- Trips with exactly $1 tip.
SELECT count(*) trips_with_one_dollar_tip
FROM [833682135931:nyctaxi.trip_fare]
WHERE float(tip_amount) = 1.00 and float(fare_amount) > 0.00 ;

-- Trips with exactly $2 tip.
SELECT count(*) trips_with_two_dollah_tip
FROM [833682135931:nyctaxi.trip_fare]
WHERE float(tip_amount=2.00) and float(fare_amount) > 0.00 ;

-- Trips with exactly $5 tip.
SELECT count(*) trips_with_fi_dollah_tip
FROM [833682135931:nyctaxi.trip_fare]
WHERE float(tip_amount=5.00) and float(fare_amount) > 0.00 ;

-- Average tip per month.
SELECT INTEGER(AVG(tip_amount)*100)/100 avg_tip,
  REGEXP_EXTRACT(pickup_datetime, "2013-([0-9]*)") month
FROM [833682135931:nyctaxi.trip_fare]
WHERE payment_type='CRD'
GROUP BY 2
ORDER BY 2

-- Number of trips based on amount of passengers.
SELECT
  INTEGER(passenger_count) AS passengers, count(*) trips
FROM
  [833682135931:nyctaxi.trip_data]
GROUP BY
  passengers
ORDER BY
  passengers;

-- Retrieve all rides on 'good weather days' for a specific medallion: 6D49E494913752B75B2685E0019FF3D5
SELECT
trip_data.medallion,trip_data.pickup_datetime,trip_data.dropoff_datetime,trip_data.passenger_count,trip_data.pickup_longitude,trip_data.pickup_latitude,trip_data.dropoff_longitude,trip_data.dropoff_latitude, trip_fare.fare_amount, trip_fare.payment_type, trip_fare.surcharge, trip_fare.mta_tax, trip_fare.tip_amount, trip_fare.tolls_amount, trip_fare.total_amount
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
WHERE
DATE(trip_data.pickup_datetime) IN ('2013-11-19' , '2013-11-11' , '2013-11-10' , '2013-11-13' , '2013-11-15' , '2013-11-14' , '2013-10-12' , '2013-10-6' , '2013-9-23' , '2013-9-26' , '2013-9-27' , '2013-9-24' , '2013-9-25' , '2013-9-28' , '2013-9-29' , '2013-8-30' , '2013-8-31' , '2013-4-28' , '2013-4-21' , '2013-4-23' , '2013-4-22' , '2013-4-25' , '2013-4-24' , '2013-4-27' , '2013-4-26' , '2013-10-23' , '2013-10-22' , '2013-10-21' , '2013-10-20' , '2013-10-25' , '2013-10-24' , '2013-3-15' , '2013-3-14' , '2013-10-29' , '2013-10-28' , '2013-3-11' , '2013-3-10' , '2013-3-13' , '2013-11-20' , '2013-11-30' , '2013-11-24' , '2013-11-25' , '2013-1-22' , '2013-1-23' , '2013-1-20' , '2013-1-21' , '2013-1-26' , '2013-1-27' , '2013-1-24' , '2013-4-2' , '2013-4-3' , '2013-4-4' , '2013-4-5' , '2013-4-6' , '2013-4-7' , '2013-4-8' , '2013-4-9' , '2013-10-16' , '2013-8-4' , '2013-8-5' , '2013-8-6' , '2013-8-7' , '2013-8-2' , '2013-8-18' , '2013-6-15' , '2013-6-12' , '2013-8-10' , '2013-8-11' , '2013-8-14' , '2013-8-15' , '2013-8-16' , '2013-8-17' , '2013-1-10' , '2013-1-17' , '2013-6-29' , '2013-6-28' , '2013-6-25' , '2013-6-24' , '2013-6-23' , '2013-6-22' , '2013-6-21' , '2013-6-20' , '2013-1-9' , '2013-12-11' , '2013-1-18' , '2013-2-1' , '2013-2-6' , '2013-10-30' , '2013-2-7' , '2013-1-19' , '2013-12-16' , '2013-4-15' , '2013-4-16' , '2013-4-11' , '2013-3-28' , '2013-3-29' , '2013-3-20' , '2013-3-22' , '2013-3-23' , '2013-3-24' , '2013-3-26' , '2013-3-27' , '2013-7-11' , '2013-11-21' , '2013-7-15' , '2013-7-17' , '2013-11-28' , '2013-11-29' , '2013-12-21' , '2013-12-20' , '2013-12-22' , '2013-12-25' , '2013-12-24' , '2013-12-27' , '2013-12-26' , '2013-12-28' , '2013-6-16' , '2013-8-19' , '2013-5-26' , '2013-5-27' , '2013-5-22' , '2013-5-21' , '2013-7-31' , '2013-7-30' , '2013-5-5' , '2013-1-2' , '2013-5-4' , '2013-5-7' , '2013-7-7' , '2013-7-5' , '2013-7-4' , '2013-10-19' , '2013-9-30' , '2013-1-4' , '2013-8-25' , '2013-8-24' , '2013-8-21' , '2013-8-20' , '2013-8-23' , '2013-12-30' , '2013-12-31' , '2013-7-20' , '2013-1-7' , '2013-1-6' , '2013-5-31' , '2013-5-30' , '2013-9-19' , '2013-9-18' , '2013-9-11' , '2013-9-10' , '2013-9-17' , '2013-9-15' , '2013-9-14' , '2013-1-8' , '2013-9-20' , '2013-10-5' , '2013-10-2' , '2013-10-3' , '2013-4-30' , '2013-10-1' , '2013-1-1' , '2013-12-19' , '2013-1-3' , '2013-10-15' , '2013-1-5' , '2013-10-13' , '2013-12-12' , '2013-12-13' , '2013-10-18' , '2013-2-12' , '2013-2-10' , '2013-2-17' , '2013-2-15' , '2013-2-18' , '2013-5-17' , '2013-5-16' , '2013-5-15' , '2013-5-14' , '2013-5-13' , '2013-10-8' , '2013-5-10' , '2013-10-9' , '2013-5-18' , '2013-7-24' , '2013-7-26' , '2013-7-27' , '2013-7-21' , '2013-3-30' , '2013-10-14' , '2013-7-16' , '2013-2-22' , '2013-2-21' , '2013-2-20' , '2013-2-25' , '2013-2-24' , '2013-11-3' , '2013-7-19' , '2013-11-6' , '2013-11-5' , '2013-11-4' , '2013-11-9' , '2013-11-8' , '2013-7-18' , '2013-10-27' , '2013-10-26' , '2013-11-2' , '2013-3-17' , '2013-5-6' , '2013-5-1' , '2013-7-6' , '2013-5-3' , '2013-5-2' , '2013-2-4' , '2013-3-9' , '2013-3-6' , '2013-3-5' , '2013-3-4' , '2013-3-1' , '2013-6-4' , '2013-6-5' , '2013-6-1' , '2013-12-18' , '2013-6-9' , '2013-9-9' , '2013-9-8' , '2013-9-1' , '2013-9-5' , '2013-9-4' , '2013-9-7' , '2013-9-6' , '2013-12-1' , '2013-12-2' , '2013-12-3' , '2013-12-4')
AND trip_data.medallion = '6D49E494913752B75B2685E0019FF3D5'
ORDER BY trip_data.pickup_datetime ASC


-- Retrieve all rides on 'good weather days' for 5 medallions.
SELECT
trip_data.medallion,trip_data.pickup_datetime,trip_data.dropoff_datetime,trip_data.passenger_count,trip_data.pickup_longitude,trip_data.pickup_latitude,trip_data.dropoff_longitude,trip_data.dropoff_latitude, trip_fare.fare_amount, trip_fare.payment_type, trip_fare.surcharge, trip_fare.mta_tax, trip_fare.tip_amount, trip_fare.tolls_amount, trip_fare.total_amount
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
WHERE
DATE(trip_data.pickup_datetime) IN ('2013-11-19' , '2013-11-11' , '2013-11-10' , '2013-11-13' , '2013-11-15' , '2013-11-14' , '2013-10-12' , '2013-10-6' , '2013-9-23' , '2013-9-26' , '2013-9-27' , '2013-9-24' , '2013-9-25' , '2013-9-28' , '2013-9-29' , '2013-8-30' , '2013-8-31' , '2013-4-28' , '2013-4-21' , '2013-4-23' , '2013-4-22' , '2013-4-25' , '2013-4-24' , '2013-4-27' , '2013-4-26' , '2013-10-23' , '2013-10-22' , '2013-10-21' , '2013-10-20' , '2013-10-25' , '2013-10-24' , '2013-3-15' , '2013-3-14' , '2013-10-29' , '2013-10-28' , '2013-3-11' , '2013-3-10' , '2013-3-13' , '2013-11-20' , '2013-11-30' , '2013-11-24' , '2013-11-25' , '2013-1-22' , '2013-1-23' , '2013-1-20' , '2013-1-21' , '2013-1-26' , '2013-1-27' , '2013-1-24' , '2013-4-2' , '2013-4-3' , '2013-4-4' , '2013-4-5' , '2013-4-6' , '2013-4-7' , '2013-4-8' , '2013-4-9' , '2013-10-16' , '2013-8-4' , '2013-8-5' , '2013-8-6' , '2013-8-7' , '2013-8-2' , '2013-8-18' , '2013-6-15' , '2013-6-12' , '2013-8-10' , '2013-8-11' , '2013-8-14' , '2013-8-15' , '2013-8-16' , '2013-8-17' , '2013-1-10' , '2013-1-17' , '2013-6-29' , '2013-6-28' , '2013-6-25' , '2013-6-24' , '2013-6-23' , '2013-6-22' , '2013-6-21' , '2013-6-20' , '2013-1-9' , '2013-12-11' , '2013-1-18' , '2013-2-1' , '2013-2-6' , '2013-10-30' , '2013-2-7' , '2013-1-19' , '2013-12-16' , '2013-4-15' , '2013-4-16' , '2013-4-11' , '2013-3-28' , '2013-3-29' , '2013-3-20' , '2013-3-22' , '2013-3-23' , '2013-3-24' , '2013-3-26' , '2013-3-27' , '2013-7-11' , '2013-11-21' , '2013-7-15' , '2013-7-17' , '2013-11-28' , '2013-11-29' , '2013-12-21' , '2013-12-20' , '2013-12-22' , '2013-12-25' , '2013-12-24' , '2013-12-27' , '2013-12-26' , '2013-12-28' , '2013-6-16' , '2013-8-19' , '2013-5-26' , '2013-5-27' , '2013-5-22' , '2013-5-21' , '2013-7-31' , '2013-7-30' , '2013-5-5' , '2013-1-2' , '2013-5-4' , '2013-5-7' , '2013-7-7' , '2013-7-5' , '2013-7-4' , '2013-10-19' , '2013-9-30' , '2013-1-4' , '2013-8-25' , '2013-8-24' , '2013-8-21' , '2013-8-20' , '2013-8-23' , '2013-12-30' , '2013-12-31' , '2013-7-20' , '2013-1-7' , '2013-1-6' , '2013-5-31' , '2013-5-30' , '2013-9-19' , '2013-9-18' , '2013-9-11' , '2013-9-10' , '2013-9-17' , '2013-9-15' , '2013-9-14' , '2013-1-8' , '2013-9-20' , '2013-10-5' , '2013-10-2' , '2013-10-3' , '2013-4-30' , '2013-10-1' , '2013-1-1' , '2013-12-19' , '2013-1-3' , '2013-10-15' , '2013-1-5' , '2013-10-13' , '2013-12-12' , '2013-12-13' , '2013-10-18' , '2013-2-12' , '2013-2-10' , '2013-2-17' , '2013-2-15' , '2013-2-18' , '2013-5-17' , '2013-5-16' , '2013-5-15' , '2013-5-14' , '2013-5-13' , '2013-10-8' , '2013-5-10' , '2013-10-9' , '2013-5-18' , '2013-7-24' , '2013-7-26' , '2013-7-27' , '2013-7-21' , '2013-3-30' , '2013-10-14' , '2013-7-16' , '2013-2-22' , '2013-2-21' , '2013-2-20' , '2013-2-25' , '2013-2-24' , '2013-11-3' , '2013-7-19' , '2013-11-6' , '2013-11-5' , '2013-11-4' , '2013-11-9' , '2013-11-8' , '2013-7-18' , '2013-10-27' , '2013-10-26' , '2013-11-2' , '2013-3-17' , '2013-5-6' , '2013-5-1' , '2013-7-6' , '2013-5-3' , '2013-5-2' , '2013-2-4' , '2013-3-9' , '2013-3-6' , '2013-3-5' , '2013-3-4' , '2013-3-1' , '2013-6-4' , '2013-6-5' , '2013-6-1' , '2013-12-18' , '2013-6-9' , '2013-9-9' , '2013-9-8' , '2013-9-1' , '2013-9-5' , '2013-9-4' , '2013-9-7' , '2013-9-6' , '2013-12-1' , '2013-12-2' , '2013-12-3' , '2013-12-4')
AND trip_data.medallion IN ('3418135604CD3F357DD9577AF978C5C0','6D3B2A7682C30DCF64F3F12976EF93B6','6D49E494913752B75B2685E0019FF3D5','4C4A0AFC432A1A87E97ED8F18403FF6E','1258CA1DF5E2A9E9A9F7848408A7AAEB')
ORDER BY trip_data.pickup_datetime ASC

-- Retrieve all rides on 'bad weather days' for a specific medallion: 6D49E494913752B75B2685E0019FF3D5
SELECT
trip_data.medallion,trip_data.pickup_datetime,trip_data.dropoff_datetime,trip_data.passenger_count,trip_data.pickup_longitude,trip_data.pickup_latitude,trip_data.dropoff_longitude,trip_data.dropoff_latitude, trip_fare.fare_amount, trip_fare.payment_type, trip_fare.surcharge, trip_fare.mta_tax, trip_fare.tip_amount, trip_fare.tolls_amount, trip_fare.total_amount
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
WHERE
DATE(trip_data.pickup_datetime) IN ('2013-8-22' , '2013-8-26' , '2013-8-27' , '2013-9-21' , '2013-9-22' , '2013-3-31' , '2013-11-1' , '2013-11-7' , '2013-8-13' , '2013-8-12' , '2013-2-3' , '2013-2-5' , '2013-2-9' , '2013-12-7' , '2013-12-6' , '2013-12-5' , '2013-12-9' , '2013-12-8' , '2013-5-29' , '2013-5-28' , '2013-4-29' , '2013-5-25' , '2013-5-24' , '2013-5-20' , '2013-4-20' , '2013-7-14' , '2013-7-12' , '2013-7-13' , '2013-7-10' , '2013-7-2' , '2013-7-3' , '2013-7-1' , '2013-7-8' , '2013-7-9' , '2013-8-1' , '2013-8-3' , '2013-8-9' , '2013-8-8' , '2013-7-29' , '2013-7-28' , '2013-7-23' , '2013-7-22' , '2013-7-25' , '2013-4-13' , '2013-6-14' , '2013-4-17' , '2013-6-10' , '2013-4-19' , '2013-4-18' , '2013-6-19' , '2013-6-18' , '2013-11-12' , '2013-11-16' , '2013-11-17' , '2013-11-18' , '2013-10-4' , '2013-10-7' , '2013-1-29' , '2013-1-28' , '2013-1-25' , '2013-12-15' , '2013-12-14' , '2013-12-17' , '2013-12-10' , '2013-2-2' , '2013-3-16' , '2013-3-18' , '2013-3-19' , '2013-5-19' , '2013-6-30' , '2013-2-28' , '2013-2-23' , '2013-2-26' , '2013-2-27' , '2013-6-8' , '2013-6-7' , '2013-6-6' , '2013-6-3' , '2013-6-2' , '2013-9-2' , '2013-9-3' , '2013-10-17' , '2013-10-11' , '2013-10-10' , '2013-8-28' , '2013-8-29' , '2013-1-11' , '2013-1-12' , '2013-1-13' , '2013-1-14' , '2013-1-15' , '2013-1-16' , '2013-5-11' , '2013-5-12' , '2013-2-8' , '2013-9-12' , '2013-9-13' , '2013-9-16' , '2013-2-11' , '2013-2-13' , '2013-2-14' , '2013-2-16' , '2013-2-19' , '2013-11-27' , '2013-11-26' , '2013-11-23' , '2013-11-22' , '2013-4-12' , '2013-5-8' , '2013-5-9' , '2013-4-10' , '2013-6-11' , '2013-6-13' , '2013-4-14' , '2013-3-7' , '2013-3-2' , '2013-5-23' , '2013-3-8' , '2013-4-1' , '2013-10-31' , '2013-3-12' , '2013-1-30' , '2013-1-31' , '2013-3-21' , '2013-12-29' , '2013-6-17' , '2013-12-23' , '2013-3-25' , '2013-3-3' , '2013-6-26' , '2013-6-27')
AND trip_data.medallion = '6D49E494913752B75B2685E0019FF3D5'
ORDER BY trip_data.pickup_datetime ASC

-- Retrieve all rides on 'bad weather days' for 5 medallions
SELECT
trip_data.medallion,trip_data.pickup_datetime,trip_data.dropoff_datetime,trip_data.passenger_count,trip_data.pickup_longitude,trip_data.pickup_latitude,trip_data.dropoff_longitude,trip_data.dropoff_latitude, trip_fare.fare_amount, trip_fare.payment_type, trip_fare.surcharge, trip_fare.mta_tax, trip_fare.tip_amount, trip_fare.tolls_amount, trip_fare.total_amount
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
WHERE
DATE(trip_data.pickup_datetime) IN ('2013-8-22' , '2013-8-26' , '2013-8-27' , '2013-9-21' , '2013-9-22' , '2013-3-31' , '2013-11-1' , '2013-11-7' , '2013-8-13' , '2013-8-12' , '2013-2-3' , '2013-2-5' , '2013-2-9' , '2013-12-7' , '2013-12-6' , '2013-12-5' , '2013-12-9' , '2013-12-8' , '2013-5-29' , '2013-5-28' , '2013-4-29' , '2013-5-25' , '2013-5-24' , '2013-5-20' , '2013-4-20' , '2013-7-14' , '2013-7-12' , '2013-7-13' , '2013-7-10' , '2013-7-2' , '2013-7-3' , '2013-7-1' , '2013-7-8' , '2013-7-9' , '2013-8-1' , '2013-8-3' , '2013-8-9' , '2013-8-8' , '2013-7-29' , '2013-7-28' , '2013-7-23' , '2013-7-22' , '2013-7-25' , '2013-4-13' , '2013-6-14' , '2013-4-17' , '2013-6-10' , '2013-4-19' , '2013-4-18' , '2013-6-19' , '2013-6-18' , '2013-11-12' , '2013-11-16' , '2013-11-17' , '2013-11-18' , '2013-10-4' , '2013-10-7' , '2013-1-29' , '2013-1-28' , '2013-1-25' , '2013-12-15' , '2013-12-14' , '2013-12-17' , '2013-12-10' , '2013-2-2' , '2013-3-16' , '2013-3-18' , '2013-3-19' , '2013-5-19' , '2013-6-30' , '2013-2-28' , '2013-2-23' , '2013-2-26' , '2013-2-27' , '2013-6-8' , '2013-6-7' , '2013-6-6' , '2013-6-3' , '2013-6-2' , '2013-9-2' , '2013-9-3' , '2013-10-17' , '2013-10-11' , '2013-10-10' , '2013-8-28' , '2013-8-29' , '2013-1-11' , '2013-1-12' , '2013-1-13' , '2013-1-14' , '2013-1-15' , '2013-1-16' , '2013-5-11' , '2013-5-12' , '2013-2-8' , '2013-9-12' , '2013-9-13' , '2013-9-16' , '2013-2-11' , '2013-2-13' , '2013-2-14' , '2013-2-16' , '2013-2-19' , '2013-11-27' , '2013-11-26' , '2013-11-23' , '2013-11-22' , '2013-4-12' , '2013-5-8' , '2013-5-9' , '2013-4-10' , '2013-6-11' , '2013-6-13' , '2013-4-14' , '2013-3-7' , '2013-3-2' , '2013-5-23' , '2013-3-8' , '2013-4-1' , '2013-10-31' , '2013-3-12' , '2013-1-30' , '2013-1-31' , '2013-3-21' , '2013-12-29' , '2013-6-17' , '2013-12-23' , '2013-3-25' , '2013-3-3' , '2013-6-26' , '2013-6-27')
AND trip_data.medallion IN ('3418135604CD3F357DD9577AF978C5C0','6D3B2A7682C30DCF64F3F12976EF93B6','6D49E494913752B75B2685E0019FF3D5','4C4A0AFC432A1A87E97ED8F18403FF6E','1258CA1DF5E2A9E9A9F7848408A7AAEB')
ORDER BY trip_data.pickup_datetime ASC

-- All rides for the medallion: 6D49E494913752B75B2685E0019FF3D5.
SELECT
trip_data.medallion,trip_data.pickup_datetime,trip_data.dropoff_datetime,trip_data.passenger_count,trip_data.trip_time_in_secs,trip_data.trip_distance,trip_data.pickup_longitude,trip_data.pickup_latitude,trip_data.dropoff_longitude,trip_data.dropoff_latitude, trip_fare.fare_amount, trip_fare.payment_type, trip_fare.surcharge, trip_fare.mta_tax, trip_fare.tip_amount, trip_fare.tolls_amount, trip_fare.total_amount
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
WHERE
trip_data.medallion = '6D49E494913752B75B2685E0019FF3D5'
ORDER BY trip_data.pickup_datetime ASC
LIMIT 25000

SELECT
trip_data.medallion,trip_data.pickup_datetime,trip_data.dropoff_datetime,trip_data.passenger_count,trip_data.trip_time_in_secs,trip_data.trip_distance,trip_data.pickup_longitude,trip_data.pickup_latitude,trip_data.dropoff_longitude,trip_data.dropoff_latitude, trip_fare.fare_amount, trip_fare.payment_type, trip_fare.surcharge, trip_fare.mta_tax, trip_fare.tip_amount, trip_fare.tolls_amount, trip_fare.total_amount
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
ORDER BY trip_data.pickup_datetime ASC
LIMIT 25000

-- Used to grab 16k of random rides. MAX was 16k because GoogleBiqQuery
-- Wouldn't let you save a csv with over 16k rows.
SELECT
trip_data.medallion,trip_data.pickup_datetime,trip_data.dropoff_datetime,trip_data.passenger_count,trip_data.trip_time_in_secs,trip_data.trip_distance,trip_data.pickup_longitude,trip_data.pickup_latitude,trip_data.dropoff_longitude,trip_data.dropoff_latitude, trip_fare.fare_amount, trip_fare.payment_type, trip_fare.surcharge, trip_fare.mta_tax, trip_fare.tip_amount, trip_fare.tolls_amount, trip_fare.total_amount
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
LIMIT 16000;

-- Find Medallion, # of rides, Total Amount Made By Medallion.
SELECT
trip_data.medallion as Medallion, COUNT(*) as Number_of_Rides,(SUM(ROUND(FLOAT(trip_fare.total_amount)))) as Total
FROM [833682135931:nyctaxi.trip_data] as trip_data
JOIN EACH [833682135931:nyctaxi.trip_fare] as trip_fare
ON trip_data.medallion = trip_fare.medallion
AND trip_data.pickup_datetime = trip_fare.pickup_datetime
GROUP BY Medallion
ORDER BY Total DESC;

