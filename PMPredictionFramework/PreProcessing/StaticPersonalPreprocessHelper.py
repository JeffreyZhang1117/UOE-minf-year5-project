import pandas as pd
import sys
sys.path.append('.')
# from PMPredictionFramework.PreProcessing.load_files import *
# from PMPredictionFramework.PreProcessing.constants import *
from PreProcessing.load_files import *
from PreProcessing.constants import *

from math import radians, cos, sin, asin, sqrt
import numpy as np
from scipy import stats
import swifter
from shapely import geometry
from shapely import ops
import numpy as np
import geojson
import utm
import datetime

def preprocess(dataset):
    if(dataset == 'PEEPS'):
        static_df = loading_peeps_static()
    elif(dataset == 'DAPHNE'):
        static_df = loading_daphne_static()
    elif (dataset == 'INHALE'):
        # static_df = loading_AURN_static()
        static_df = loading_LA_static()
    static_df_processed = preprocessing_static(static_df, dataset)
    fileName = dataset + '_stationary.csv'
    static_df_processed.to_csv(fileName)
    # if (dataset == 'INHALE'):
    #     personal_df, subjectID = loading_inhale_personal()
    # else:
    #     personal_df = loading_personal(dataset)
    subject_list = ['001(1)','001(2)','002(1)','002(2)','005(1)','005(2)','006(1)','006(2)','007(1)','008(1)','011(1)',
                    '014(1)','100(1)','100(2)','101(1)','101(2)','102(1)','102(2)','103(1)','103(2)','104(1)','106(1)',
                    '106(2)','107(1)','107(2)','108(1)','108(2)','109(1)','109(2)','110(2)','111(1)','112(1)','112(2)',
                    '113(2)','114(1)']
    for id in subject_list:
        personal_df, subjectID = loading_inhale_personal(id)
        personal_df_preprocessed = preprocessing_personal(personal_df, static_df_processed, dataset)
        # fileName = dataset + '_personal_' + subjectID + '.csv'
        # personal_df_preprocessed.to_csv(fileName)
        updateRoadType(dataset, personal_df_preprocessed, subjectID)

def updateRoadType(dataset, personal_df, subjectID):
    road_list = ['residential', 'living_street', 'tertiary', 'trunk', 'secondary', 'primary', 'pedestrian',
                 'tertiary_link', 'trunk_link', 'primary_link', 'secondary_link']
    road_dict = {k: v for v, k in enumerate(road_list)}
    if(dataset == 'PEEPS'):
        roads = load_map_delhi('PreProcessing\\osmfiles\\delhi.geojson')
        roads = roads.iloc[19:]
        roads = roads[~roads.id.str.startswith('node')]
        roads = roads.set_index('id')
    elif (dataset == 'DAPHNE'):
        roads = load_map('PreProcessing\\osmfiles\\mapmodified_lines.geojson')
        roads = roads.reset_index()
        roads = roads[~roads.osm_id.str.startswith('node')]
        roads = roads.set_index('osm_id')
    elif (dataset == 'INHALE'):
        roads = load_map('PreProcessing\\osmfiles\\mapINHALEfullPersonal_lines.geojson')
        roads = roads.reset_index()
        roads = roads[~roads.osm_id.str.startswith('node')]
        roads = roads.set_index('osm_id')
    roads['utmLocation'] = roads.apply(lambda row: utm_convert(row), axis=1)
    roads['locationLineString'] = roads.apply(lambda row: linestring(row), axis=1)
    roads = roads.filter(['name', 'highway', 'location', 'utmLocation', 'locationLineString'])
    personal_df['roadType'] = personal_df.swifter.apply(lambda row: get_roadtype(row, roads), axis=1)
    fileName = dataset + '_personal_RT_' + subjectID + '.csv'
    personal_df.to_csv(fileName)

def load_map_delhi(path):
    with open(path, encoding="utf-8") as f:
        lines = geojson.load(f)
    Roads = []
    for allFeatures in lines.features:
        if 'highway' in allFeatures['properties']:
            roadinfo = allFeatures['properties']

            roadinfo['location'] = allFeatures['geometry']['coordinates']

            Roads.append(roadinfo)
    Roads = pd.DataFrame.from_dict(Roads)
    Roads = Roads[Roads.highway.isin(
        ['residential', 'living_street', 'tertiary', 'trunk', 'secondary', 'primary', 'pedestrian', 'tertiary_link',
         'trunk_link', 'primary_link', 'secondary_link'])]
    # Roads = Roads.set_index('id')
    return Roads

def load_map(path):
    with open(path, encoding="utf-8") as f:
        lines = geojson.load(f)
    Roads = []
    for allFeatures in lines.features:
        if 'highway' in allFeatures['properties']:
            roadinfo = allFeatures['properties']

            roadinfo['location'] = allFeatures['geometry']['coordinates']

            Roads.append(roadinfo)
    Roads = pd.DataFrame.from_dict(Roads)
    Roads = Roads[Roads.highway.isin(
            ['residential', 'living_street', 'tertiary', 'trunk', 'secondary', 'primary', 'pedestrian', 'tertiary_link',
             'trunk_link', 'primary_link', 'secondary_link'])]
    Roads = Roads.set_index('osm_id')
    print('Roads::', Roads.shape)
    return Roads

def utm_convert(row):
    ls = []
    for y in row['location']:
        ls.append(list(utm.from_latlon(y[1], y[0])[0:2]))
    return ls


def linestring(row):
    return geometry.LineString(row.utmLocation)

def get_roadtype(row, roadmap):
    airspeckCoords = [(row['gpsLatitude'], row['gpsLongitude'])]

    closestRoadstoCoords = ''
    airspeckCoordsUtm = [list(utm.from_latlon(*x))[0:2] for x in airspeckCoords]
    airspeckShapelyPoints = [geometry.Point(x) for x in airspeckCoordsUtm]

    for points in airspeckShapelyPoints:
        roadDistancesToPoint = [points.distance(x) for x in roadmap.locationLineString]
        minIdx = np.argmin(roadDistancesToPoint)
        minRoadType = roadmap.highway[minIdx]
        closestRoadstoCoords = minRoadType
        minRoadName = roadmap.name[minIdx]

    return closestRoadstoCoords

def loading_personal(dataset):  # Function loads airspeck-p files and removes rows with erroneous values
    if (dataset == 'PEEPS'):
        data_logs = load_peeps_participant_details()
    elif(dataset=='DAPHNE'):
        data_logs = load_daphne_subject_details()
    pers_airspeck = pd.DataFrame()
    for idx, row in data_logs.iterrows():
        subj_id = row.name
        if (dataset == 'PEEPS'):
            frame = load_personal_airspeck_file(subj_id, 'peeps', upload_type='manual', subject_visit_number=None,
                                            is_minute_averaged=False)
        elif(dataset=='DAPHNE'):
            frame = load_personal_airspeck_file(subj_id, 'daphne', upload_type='manual',
                                                subject_visit_number=int(row.get(key='Visit number')),
                                                is_minute_averaged=True)

        if frame is not None:
            frame['walk'] = int(subj_id[3:])
        pers_airspeck = pers_airspeck.append(frame)

    pers_airspeck['timestamp'] = pers_airspeck.timestamp.dt.round('S', 'NaT')
    pers_airspeck = pers_airspeck.set_index('timestamp')
    pers_airspeck = pers_airspeck.resample('min').mean()
    pers_airspeck = pers_airspeck[pers_airspeck['temperature'].notna()]
    pers_airspeck = pers_airspeck[pers_airspeck['gpsLongitude'] != 0]
    pers_airspeck = pers_airspeck[pers_airspeck['gpsLatitude'] != 0]
    pers_airspeck.to_csv('verify.csv')
    return pers_airspeck

def loading_inhale_personal(filename):  # Function loads airspeck-p files and removes rows with erroneous values
    #let's use one subject first
    fileName = filename
    inhale_personal = pd.read_csv('D:\\Desktop\\Project\\AirspeckP\\INH{}_airspeck_personal_manual_raw.csv'.format(fileName))
    #inhale_personal = inhale_personal.iloc[0:8000,:]
    inhale_personal['timestamp'] = pd.to_datetime([str(s).split('+')[0].split('.')[0] for s in inhale_personal['timestamp']])
    print('Original DF size::',inhale_personal.shape[0])
    inhale_personal = inhale_personal[~(inhale_personal['pm2_5'] <= 1)]
    print('After removing pm < 1 DF size::',inhale_personal.shape[0])
    inhale_personal.index = inhale_personal['timestamp']
    inhale_personal = inhale_personal[inhale_personal['temperature'].notna()]
    inhale_personal = inhale_personal[inhale_personal['gpsLongitude'] != 0]
    inhale_personal = inhale_personal[inhale_personal['gpsLatitude'] != 0]
    inhale_personal.rename(columns={"subject_id": "walk"}, inplace=True)
    return inhale_personal, fileName



def preprocessing_static(unprocessed_dataset, dataset):
    if(dataset != 'INHALE'):
        static_df = unprocessed_dataset.filter(
            items=['timestamp', 'pm2_5', 'temperature', 'humidity', 'gpsLongitude', 'gpsLatitude', 'UUID'])
    else:
        static_df = unprocessed_dataset
    static_df = static_df.reset_index()
    static_dataframe = parse_dow_hod(static_df)

    static_dataframe = static_dataframe.set_index("timestamp")

    return static_dataframe


def parse_dow_hod(dataset):
    # filDataset = dataset.filter(items = ['timestamp'])
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df['day_of_week'] = dataset['timestamp'].dt.day_name()
    df2['hour_of_day'] = dataset['timestamp'].dt.hour
    dataset = pd.concat([df2, df, dataset], axis=1)
    dataset = dataset.filter(
        items=['timestamp', 'pm2_5', 'hour_of_day', 'day_of_week', 'temperature', 'humidity', 'gpsLongitude',
               'gpsLatitude', 'UUID','walk'])

    dataset.loc[(dataset.day_of_week == 'Monday'), 'day_of_week'] = int(0)
    dataset.loc[(dataset.day_of_week == 'Tuesday'), 'day_of_week'] = int(1)
    dataset.loc[(dataset.day_of_week == 'Wednesday'), 'day_of_week'] = int(2)
    dataset.loc[(dataset.day_of_week == 'Thursday'), 'day_of_week'] = int(3)
    dataset.loc[(dataset.day_of_week == 'Friday'), 'day_of_week'] = int(4)
    dataset.loc[(dataset.day_of_week == 'Saturday'), 'day_of_week'] = int(5)
    dataset.loc[(dataset.day_of_week == 'Sunday'), 'day_of_week'] = int(6)

    return dataset

def load_peeps_participant_details(phase=1):
    details = pd.read_excel(peeps_participant_details_filepath, engine='openpyxl')
    if (phase == 1):
        details = details.iloc[1:75]
    if (phase == 2):
        details = details.iloc[77:]
    details = details.set_index('Subject ID')
    details['Subject ID'] = details.index
    #if (phase == 1):
    #    details = details.drop('PEV015')
    return details

def load_daphne_subject_details():
    logs_ap = pd.read_excel(daphne_logs_filepath, sheet_name="Asthma Cohort WP1.1", engine = 'openpyxl')
    logs_mc = pd.read_excel(daphne_logs_filepath, sheet_name="MC Cohort", engine = 'openpyxl')
    logs = pd.concat([logs_ap, logs_mc], sort=True)
    # Set Subject ID as index but keep column as well
    logs = logs.set_index('Subject ID')
    logs['Subject ID'] = logs.index

    # Only keep non-empty rows
    logs = logs[~pd.isnull(logs.index)]
    return logs

def loading_peeps_static():
    peeps_logs = load_peeps_participant_details()
    pps_static_airspeck = pd.DataFrame()
    i = 0
    for idx, row in peeps_logs.iterrows():
        subj_id = row.name

        frame = load_static_airspeck_file(subj_id, 'peeps', suffix_filename="_home", upload_type='automatic',
                                          subject_visit_number=None)
        frame1 = load_static_airspeck_file(subj_id, 'peeps', suffix_filename="_work", upload_type='automatic',
                                           subject_visit_number=None)

        if frame is not None:
            frame['UUID'] = row[6]
            frame['type'] = 'home'

        if frame1 is not None:
            frame1['UUID'] = row[5]
            frame1['type'] = 'work'

        pps_static_airspeck = pps_static_airspeck.append(frame)
        pps_static_airspeck = pps_static_airspeck.append(frame1)

    pps_static_airspeck['timestamp'] = pps_static_airspeck.timestamp.dt.round('S', 'NaT')
    pps_static_airspeck = pps_static_airspeck.set_index('timestamp')

    return pps_static_airspeck

def loading_daphne_static():
    daphne_logs = load_daphne_subject_details()
    daphne_static_airspeck = pd.DataFrame()
    i = 0
    for idx, row in daphne_logs.iterrows():
        subj_id = row.name
        frame = load_static_airspeck_file(subj_id, 'daphne', suffix_filename="_home", upload_type='automatic',
                                          subject_visit_number=int(row.get(key='Visit number')))
        frame1 = load_static_airspeck_file(subj_id, 'daphne', suffix_filename="_school", upload_type='automatic',
                                           subject_visit_number=int(row.get(key='Visit number')))
        frame2 = load_static_airspeck_file(subj_id, 'daphne', suffix_filename="_community", upload_type='automatic',
                                           subject_visit_number=int(row.get(key='Visit number')))
        if frame is not None:
            frame['UUID'] = row.get(key='Static sensor inside home ID')
            frame['type'] = 'home'

        if frame1 is not None:
            frame1['UUID'] = row.get(key='School static sensor ID')
            frame1['type'] = 'school'

        if frame2 is not None:
            frame2['UUID'] = row.get(key='Nearest community static sensor ID')
            frame2['type'] = 'community'

        daphne_static_airspeck = daphne_static_airspeck.append(frame)
        daphne_static_airspeck = daphne_static_airspeck.append(frame1)
        daphne_static_airspeck = daphne_static_airspeck.append(frame2)

    daphne_static_airspeck['timestamp'] = daphne_static_airspeck.timestamp.dt.round('S', 'NaT')
    daphne_static_airspeck = daphne_static_airspeck.set_index('timestamp')

    return daphne_static_airspeck

def loading_AURN_static():
    originalAURNFile = pd.read_csv('D:\\Desktop\\Project\\DataFiles\\INHALE_AURN_StaticFiles_SPSmodel\\AirQualityData_GreaterLondon_FebToJuly_modified_1.csv',
        parse_dates=[['Date', 'Time']],  dayfirst=True)
    df = originalAURNFile.melt(id_vars=["Date_Time"],var_name="UUID",value_name="pm2_5")
    df = df.rename(columns={"Date_Time": "timestamp"})
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df = df.set_index('timestamp')
    df['timestamp'] = df.index
    df = df.replace('No data', np.nan)
    df['pm2_5']= pd.to_numeric(df['pm2_5'])
    df = df[(df['pm2_5'])>1]
    df['timestamp'] = df.timestamp.dt.round('S', 'NaT')
    df = df.set_index('timestamp')
    return df

def loading_LA_static():
    df = pd.read_csv('D:\\Desktop\\Project\\PMPredictionFramework\\PreProcessing\\LaqnData.csv')
    df = df[['DateTime', 'Site', 'Value']]
    df = df.rename(columns={"DateTime": "timestamp", "Site": "UUID", "Value": "pm2_5"})
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, utc=True).dt.tz_convert('Europe/London')
    df = df.set_index('timestamp')
    df['timestamp'] = df.index
    df['pm2_5'] = pd.to_numeric(df['pm2_5'])
    df = df[(df['pm2_5']) > 1]
    df['timestamp'] = df.timestamp.dt.round('S', 'NaT')
    df = df.set_index('timestamp')
    return df

def pmt(timestamp):
    # timestamp = row['timestamp']
    unix = timestamp.timestamp()
    minus = unix - 120
    plus = unix + 120
    c_minus = pd.Timestamp(minus, unit='s', tz='Asia/Kolkata')
    c_plus = pd.Timestamp(plus, unit='s', tz='Asia/Kolkata')
    first = str(c_minus).split()[1].split('-')[0].split('+')[0]
    last = str(c_plus).split()[1].split('-')[0].split('+')[0]

    return first, last


def drop_numerical_outliers(df, z_thresh=1.5):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    df = df.loc[[not np.isnan(x) for x in df['pm2_5']]]
    test = df[['pm2_5', 'hour_of_day', 'day_of_week', 'humidity', 'gpsLongitude', 'gpsLatitude', 'dist_to_closest_pm']]
    constrains = test.select_dtypes(include=[np.number, np.float]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)
    return df


def havesine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_sensor_uuids(row, all_stations, dataset):
    lon1 = row['gpsLongitude']
    lat1 = row['gpsLatitude']

    # maybe choose 4 first based on avg lat and lon? could reduce time

    haves = {}
    for idx, rows in all_stations.iterrows():
        haves[idx] = havesine(lon1, lat1, rows[0], rows[1])

    x = sorted(((v, k) for k, v in haves.items()))
    # uuid =  [k_closest][1]
    # dist = x[k_closest][0]

    if(dataset == 'INHALE'):
        return pd.Series([x[0][1], x[0][0], x[1][1], x[1][0], x[2][1], x[2][0]])
    else:
        return pd.Series(
        [x[0][1], x[0][0], x[1][1], x[1][0], x[2][1], x[2][0], x[3][1], x[3][0], x[4][1], x[4][0], x[5][1], x[5][0],
         x[6][1], x[6][0], x[7][1], x[7][0], x[8][1], x[8][0], x[9][1], x[9][0], x[10][1], x[10][0], x[11][1], x[11][0],
         x[12][1], x[12][0]])


def get_sensors(dataset, datasetName):
    inter = dataset
    if(datasetName=='PEEPS'):
        all_stations = pd.read_csv("Preprocessing\\peeps_sensors.csv")
    elif(datasetName=='DAPHNE'):
        all_stations = pd.read_csv("Preprocessing\\daphne_sensors.csv")
    elif (datasetName == 'INHALE'):
        # all_stations = pd.read_csv("Preprocessing\\AURN_sensors.csv") # AURN
        all_stations = pd.read_csv("Preprocessing\\LA_sensors.csv") # London Air
    all_stations = all_stations.set_index("UUID")
    if (datasetName == 'INHALE'):
        inter[['closest_UUID', 'dist_closest_s', '2_closest_UUID', '2_dist_closest_s', '3_closest_UUID',
               '3_dist_closest_s']] = inter.swifter.apply(lambda row: get_sensor_uuids(row, all_stations, datasetName), axis=1)
        #For AURN sufficient to check for 3 sensors as data available most of the time
    else:
        inter[['closest_UUID', 'dist_closest_s', '2_closest_UUID', '2_dist_closest_s', '3_closest_UUID', '3_dist_closest_s',
           '4_closest_UUID', '4_dist_closest_s', '5_closest_UUID', '5_dist_closest_s', '6_closest_UUID',
           '6_dist_closest_s', '7_closest_UUID', '7_dist_closest_s', '8_closest_UUID', '8_dist_closest_s',
           '9_closest_UUID', '9_dist_closest_s', '10_closest_UUID', '10_dist_closest_s', '11_closest_UUID',
           '11_dist_closest_s', '12_closest_UUID', '12_dist_closest_s', '13_closest_UUID',
           '13_dist_closest_s']] = inter.swifter.apply(lambda row: get_sensor_uuids(row, all_stations, datasetName), axis=1)

    inter = inter.reset_index()

    return inter


def returnRow(row, index):
    return get_sensor_uuids(row, index)


def closest_pm_value(row, dataframe):
    uuidArr = np.empty(13, dtype=object)
    distArr = np.empty(13, dtype=object)
    uuidArr[0] = row['closest_UUID']  # .item()
    distArr[0] = row['dist_closest_s']
    uuidArr[1] = row['2_closest_UUID']
    distArr[1] = row['2_dist_closest_s']
    uuidArr[2] = row['3_closest_UUID']
    distArr[2] = row['3_dist_closest_s']
    uuidArr[3] = row['4_closest_UUID']
    distArr[3] = row['4_dist_closest_s']
    uuidArr[4] = row['5_closest_UUID']
    distArr[4] = row['5_dist_closest_s']
    uuidArr[5] = row['6_closest_UUID']
    distArr[5] = row['6_dist_closest_s']
    uuidArr[6] = row['7_closest_UUID']
    distArr[6] = row['7_dist_closest_s']
    uuidArr[7] = row['8_closest_UUID']
    distArr[7] = row['8_dist_closest_s']
    uuidArr[8] = row['9_closest_UUID']
    distArr[8] = row['9_dist_closest_s']
    uuidArr[9] = row['10_closest_UUID']
    distArr[9] = row['10_dist_closest_s']
    uuidArr[10] = row['11_closest_UUID']
    distArr[10] = row['11_dist_closest_s']
    uuidArr[11] = row['12_closest_UUID']
    distArr[11] = row['12_dist_closest_s']
    uuidArr[12] = row['13_closest_UUID']
    distArr[12] = row['13_dist_closest_s']

    date = row['date']
    first, last = np.vectorize(pmt)(row['timestamp'])

    for i in range(13):
        try:
            temp = dataframe[dataframe.UUID == uuidArr[i]].loc[date]
            temp = temp.between_time(first, last)
            pm, humidity = temp[['pm2_5', 'humidity']].iloc[0, :]

            # pm, humidity = dataframe[dataframe.UUID == uuidArr[i]].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
            return pd.Series([pm, humidity, distArr[i], uuidArr[i]])
        except:
            continue
    pm = 0
    humidity = 0
    return pd.Series([pm, humidity, 'no sensor', 'no sensor'])


def parse_date_time(dataframe):
    def parse_date(row):
        date = str(row['timestamp']).split()[0]
        return date

    def parse_time(row):
        timestamp = str(row['timestamp']).split()[1]
        time = timestamp.split('-')[0]
        return time

    split_ind = dataframe.copy()
    split_ind['date'] = split_ind.apply(lambda row: parse_date(row), axis=1)
    split_ind['time'] = split_ind.apply(lambda row: parse_time(row), axis=1)
    # split_ind = split_ind.filter(items=['timestamp','date','time','pm2_5','hour_of_day','day_of_week' ,'temperature', 'humidity','gpsLongitude', 'gpsLatitude', 'UUID_closest_s', 'dist_closest_S'])
    return split_ind


def clean_df(df):
    df = df.filter(items=['timestamp', 'pm2_5', 'hour_of_day', 'day_of_week', 'humidity', 'gpsLongitude', 'gpsLatitude',
                          'closest_pm', 'dist_to_closest_pm', 'closest_pm_id', 'walk'])
    return df


def outlier_removal(dataset, datasetName):
    dataset = dataset.set_index('timestamp')
    if(datasetName == 'INHALE'):
        dataset = dataset[dataset['gpsLatitude'] > 51]
    else:
        dataset = dataset[dataset['gpsLongitude'] > 76]
    # dataset = dataset[dataset.temperature.gt(0)]
    dataset = dataset[dataset.humidity.gt(0)]
    cleaned_dataset = drop_numerical_outliers(dataset, 3)
    return cleaned_dataset


def preprocessing_personal(unprocessed_dataset, static_data, dataset):
    #test = unprocessed_dataset.iloc[199000:199200,:]
    #unprocessed_dataset = unprocessed_dataset.iloc[30000:30500, :]
    if(dataset == 'INHALE'):
        pers_df = unprocessed_dataset.filter(items=['pm2_5','humidity', 'gpsLongitude', 'gpsLatitude',
                        'walk']) #Use humidity from personal sensors as AURN sensors did not have humidity data
    else:
        pers_df = unprocessed_dataset.filter(items=['timestamp','pm2_5','gpsLongitude', 'gpsLatitude', 'walk'])
    pers_df = pers_df.sort_index()
    pers_df = pers_df.reset_index()

    personal_df = parse_dow_hod(pers_df)

    personal_df = personal_df.set_index('timestamp')

    inter = personal_df

    inter = get_sensors(inter, dataset)

    dtp_inter = parse_date_time(inter)
    if (dataset == 'INHALE'):
        dtp_inter[['closest_pm', 'dist_to_closest_pm', 'closest_pm_id']] = dtp_inter.swifter.apply(
            lambda row: closest_pm_value_inhale(row, static_data), axis=1)
    else:
        dtp_inter[['closest_pm','humidity', 'dist_to_closest_pm', 'closest_pm_id']] = dtp_inter.swifter.apply(lambda row: closest_pm_value(row, static_data), axis=1)
    #dtp_inter[['closest_pm','humidity', 'dist_to_closest_pm', 'closest_pm_id']] = dd.from_pandas(dtp_inter, npartitions=6*multiprocessing.cpu_count()).map_partitions(lambda df: df.apply((lambda row: closest_pm_value(row, static_data)), axis=1)).compute(scheduler='processes')
    dtp_inter.to_csv('dtp_inter.csv')
    personal_dataframe = clean_df(dtp_inter)
    personal_dataframe = personal_dataframe[personal_dataframe.dist_to_closest_pm != 'no sensor']

    personal_dataframe = outlier_removal(personal_dataframe, dataset)
    print('After outlier removal::',personal_dataframe.shape)
    return personal_dataframe

def closest_pm_value_inhale(row, dataframe):
    uuidArr = np.empty(3, dtype=object)
    distArr = np.empty(3, dtype=object)
    uuidArr[0] = row['closest_UUID']  # .item()
    distArr[0] = row['dist_closest_s']
    uuidArr[1] = row['2_closest_UUID']
    distArr[1] = row['2_dist_closest_s']
    uuidArr[2] = row['3_closest_UUID']
    distArr[2] = row['3_dist_closest_s']

    date = row['date']

    for i in range(3):
        try:
            temp = dataframe[dataframe.UUID == uuidArr[i]].loc[date]
            temp = temp[temp['hour_of_day'] == row['hour_of_day']]
            pm = temp[['pm2_5']].iloc[0, 0]
            return pd.Series([pm, distArr[i], uuidArr[i]])
        except:
            continue
    pm = 0
    return pd.Series([pm, 'no sensor', 'no sensor'])