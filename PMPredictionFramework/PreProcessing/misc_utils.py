#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians

import pandas as pd
import scipy.fftpack
from sklearn.metrics import mean_squared_error, mean_absolute_error

from PreProcessing.constants import project_mapping
import sys
sys.path.append('.')
from PreProcessing.decrypt_file import *


def get_datastore_client():
    return datastore.Client('specknet-pyramid-test')


def datetime_to_timestamp(dt):
    return (dt - datetime(1970, 1, 1)).total_seconds()


def sqrt_nan_mean_squared(y_true, y_pred):
    # Remove None or Nan values
    mask = np.logical_and(np.logical_and(~np.equal(y_true, None), ~np.isnan(y_true)),
                          np.logical_and(~np.equal(y_pred, None), ~np.isnan(y_pred)))
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


def abs_nan_mean_squared(y_true, y_pred):
    # Remove None or Nan values
    mask = np.logical_and(np.logical_and(~np.equal(y_true, None), ~np.isnan(y_true)),
                          np.logical_and(~np.equal(y_pred, None), ~np.isnan(y_pred)))
    return mean_absolute_error(y_true[mask], y_pred[mask])


def distance_of_coords(lat1, lon1, lat2, lon2):
    # If one coordinate is 0, return 0
    if (lat1 == 0. and lon1 == 0.) or (lat2 == 0. and lon2 == 0.):
        return 0.

    # approximate radius of earth in km
    R = 6373.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = abs(lon2_rad - lon1_rad)
    dlat = abs(lat2_rad - lat1_rad)

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    if a < 0:
        return 0

    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


def add_speed_to_gps_data(data, lat_field_name, lon_field_name):
    lat_lngs = list(zip(data[lat_field_name], data[lon_field_name]))
    # In order to keep the length of the output the same, repeat last element
    lat_lngs.append(lat_lngs[-1])
    adj_coords = list(zip(lat_lngs[:-1], lat_lngs[1:]))
    distances = np.asarray(list(map(lambda c: distance_of_coords(c[0][0], c[0][1], c[1][0], c[1][1]), adj_coords)))

    times = data.index.values
    times = np.insert(times, -1, data.index[-1])
    times_diffs = np.diff(times, n=1) / np.timedelta64(1, 'h')

    data.loc[:, 'speed'] = distances / times_diffs
    data.loc[data['speed'] > 200, 'speed'] = 0.
    return data


def autocorr(x, mode='full'):
    result = np.correlate(x, x, mode=mode)
    return result[result.size / 2:]


def count_nonnans(signal):
    return np.count_nonzero(~np.isnan(signal))


def count_nans(signal):
    return np.count_nonzero(np.isnan(signal))


def get_activity_levels_for_accel(accel):
    return np.insert(np.linalg.norm(np.diff(accel, axis=0), axis=1), 0, 0)


def fourier_spectrum(signal, sampling_frequency=12.5):
    # Number of samplepoints
    n = len(signal)

    y = scipy.fftpack.fft(signal)
    x = np.linspace(0.0, sampling_frequency / 2.0, n / 2.0)

    # Only take positive frequencies
    y_pos = 2.0 / n * np.abs(y[:int(n / 2.0)])
    return x, y_pos


def difference(signal):
    extended_signal = np.insert(signal, 0, signal[0])
    return np.diff(extended_signal)


def drop_nans(arr):
    arr = np.asarray(arr)
    return arr[~np.isnan(arr)]


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def jitter(val, factor=0.5):
    if type(val) == np.ndarray:
        if val.ndim == 1:
            return val + (np.random.rand(len(val)) - 0.5) * factor
        else:
            return val + (np.random.rand(val.shape) - 0.5) * factor
    elif type(val) == list:
        return np.asarray(val) + (np.random.rand(len(val)) - 0.5) * factor
    else:
        return val + (np.random.rand() - 0.5) * factor


def gps_conversion_to_decimal(old):
    direction = {'N': 1, 'S': -1, 'E': 1, 'W': -1}
    new = old.replace(u'°', ' ').replace('\'', ' ').replace('"', ' ')
    new = new.split()
    new_dir = new.pop(0)
    new.extend([0, 0, 0])
    return (int(new[0]) + int(new[1]) / 60.0 + float(new[2]) / 3600.0) * direction[new_dir]


def running_mean(x, N):
    if x.ndim == 1:
        cumsum_x = np.nancumsum(np.insert(x, 0, [0]))
        return (cumsum_x[N:] - cumsum_x[:-N]) / float(N)
    else:
        cumsum_x = np.nancumsum(np.insert(x, 0, np.zeros((N, 1)), axis=1), axis=1)
        return (cumsum_x[:, N:] - cumsum_x[:, :-N]) / float(N)


def f_test_variance(x, y):
    # Test for equivalency of variances: https://stackoverflow.com/questions/21494141/how-do-i-do-a-f-test-in-python
    F = np.var(x) / np.var(y)
    df1 = len(x) - 1
    df2 = len(y) - 1
    return scipy.stats.f.sf(F, df1, df2)


def corr_without_nans(x, y):
    non_nan_mask = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(x[non_nan_mask], y[non_nan_mask])[0][1]


def load_data_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_data_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def percentage_of_true_values(ar):
    return np.round(np.count_nonzero(ar) / float(len(ar)) * 100, 2)


def get_from_entity_if_present(entity, key):
    if key in entity:
        return entity[key]
    else:
        return []


def correlation_with_nan(first, second):
    mask = ~np.logical_or(np.isnan(first), np.isnan(second))
    return np.corrcoef(first[mask], second[mask])[0][1]


# Remove samples around erroneous values, e.g. negative values due to heat. Surrounding samples will be affected as
# well. 10 was found to be a good cutoff point during 2019-05 co-location Daphne.
def set_before_after_nan(data, bad_value=-10, remove_before_after=10):
    copy_of_data = data.copy()
    if data.ndim > 1:
        for i in range(-remove_before_after, remove_before_after + 1):
            copy_of_data.loc[(data['pm2_5'] == bad_value).shift(i).fillna(False)] = np.nan
    else:
        for i in range(-remove_before_after, remove_before_after + 1):
            copy_of_data.loc[(data == bad_value).shift(i).fillna(False)] = np.nan
    return copy_of_data


def filter_out_outliers_gas(gas_data_org):
    if 'no2_we' not in gas_data_org.columns:
        return gas_data_org
    gas_data = pd.DataFrame(gas_data_org)
    
      # data = airspeck

    #for col in ['no2_we', 'no2_ae', 'ox_we', 'ox_ae']:
    quantile = 0.2
    
    for col in [ 'no2_we', 'ox_we']:
        gas_data_1 = gas_data.loc[(gas_data[col] > 10)]
        # If the data has a high standard deviation indicating the 'waterfall' distribution,
        # Filter out the lowest 20% of data 
        if gas_data_1[col].std() > 500:
            quant = gas_data[col].quantile(quantile)
            gas_data = gas_data.loc[gas_data[col] > quant]
        # If the data is clean, removing just the lowest values near 0 is better.
        else:
            gas_data = gas_data.loc[(gas_data[col] > 10)]
    '''
    gas_data = gas_data.loc[(gas_data['temperature'] > -20) & (gas_data['temperature'] <= 50)]
    gas_data = gas_data.loc[(gas_data['humidity'] > 0) & (gas_data['humidity'] <= 100)]
    
    #for col in ['no2_we', 'no2_ae', 'ox_we', 'ox_ae']:
    #    gas_data = gas_data.loc[(gas_data[col] > 10)]
    #for col in ['no2_ae', 'ox_ae']:
        gas_data = gas_data.loc[(gas_data[col] - gas_data[col].median()).abs() <= 4 * gas_data[col].std()]
    gas_data = gas_data.loc[(gas_data['ox_we'] - gas_data['ox_we'].median()).abs() <= 4 * gas_data['ox_ae'].std()]
    gas_data = gas_data.loc[(gas_data['no2_we'] - gas_data['no2_we'].median()).abs() <= 4 * gas_data['no2_ae'].std()]
    '''
   
    return gas_data

def get_project_for_subject(subject_id):
    if subject_id[:2] in project_mapping:
        return subject_id[:2]
    else:
        return subject_id[:3]


def get_home_id_for_subject(subj_id, participant_details):
    if np.count_nonzero(participant_details['Subject ID'] == subj_id) > 0:
        return participant_details.loc[participant_details['Subject ID'] == subj_id, 'Home static sensor ID'].values[0]
    else:
        print("Subject ID not in Excel sheet. Cannot load home sensor ID.")
        return "-"


def get_home_gps_for_subject(subject_id, participant_details):
    if np.count_nonzero(participant_details['Subject ID'] == subject_id) > 0:
        subj_details = participant_details.loc[participant_details['Subject ID'] == subject_id]
        if pd.isnull(subj_details['Home GPS Latitude'].iloc[0]): # If no coordinates were given, return nan
            return {'gpsLatitude': np.nan, 'gpsLongitude': np.nan}
        else:
            #if type(subj_details['Home GPS Latitude'].values[0]) == float: #if the number read in is a float, return it
            return {'gpsLatitude':subj_details['Home GPS Latitude'].values[0],
                        'gpsLongitude':subj_details['Home GPS Longitude'].values[0]}
            #else: #if the number read in is in old gps format
            #    return {'gpsLatitude': gps_conversion_to_decimal(subj_details['Home GPS Latitude'].values[0]),
            #        'gpsLongitude': gps_conversion_to_decimal(subj_details['Home GPS Longitude'].values[0])}
    else:
        print("Subject ID not in Excel sheet of future subjects")
        return "-"


def get_work_id_for_subject(subject_id, participant_details):
    if np.count_nonzero(participant_details['Subject ID'] == subject_id) > 0:
        return participant_details.loc[participant_details['Subject ID'] == subject_id,
                                       'Work static sensor ID'].values[0]
    else:
        print("Subject ID not in Excel sheet of future subjects")
        return "-"

def get_pm2_5_recording_time(airspeck, start_recording, end_recording):
    if airspeck is None or len(airspeck) < 1:
        return 0.0
    recording = airspeck[start_recording:end_recording]
    resampled = recording.resample('1min').mean()
    non_null = resampled.loc[~resampled['pm2_5'].isnull()]
    recording_time = len(non_null) / 60. 
    return recording_time

def get_pm2_5_recording_time_static(airspeck, start_recording, end_recording):
    if airspeck is None or len(airspeck) < 1:
        return 0.0
    recording = airspeck[start_recording:end_recording]
    resampled = recording.resample('5min').mean()
    non_null = resampled.loc[~resampled['pm2_5'].isnull()]
    recording_time = len(non_null) / 12. #reading every 5 minutes = 12 readings per hour
    return recording_time

def get_respeck_recording_time(respeck, start_recording, end_recording):
    if respeck is None or len(respeck) < 1:
        return 0.0
    recording = respeck[start_recording:end_recording]
    resampled = recording.resample('1min').mean()
    non_null = resampled.loc[~resampled['activity_level'].isnull()]
    recording_time = len(non_null) / 60. 
    return recording_time

def get_breathing_recording_time(respeck, start_recording, end_recording):
    if respeck is None or len(respeck) < 1:
        return 0.0
    recording = respeck[start_recording:end_recording]
    resampled = recording.resample('1min').mean()
    non_null = resampled.loc[~resampled['breathing_rate'].isnull()]
    recording_time = len(non_null) / 60. 
    return recording_time