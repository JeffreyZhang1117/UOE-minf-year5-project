import pandas as pd
import sys
sys.path.append('.')
# from PMPredictionFramework.PreProcessing.load_files import *
# from PMPredictionFramework.PreProcessing.formatdata import *
from PreProcessing.load_files import *
from PreProcessing.formatdata import *
from os import listdir
from os.path import join

def preprocessSS(dataset):
    if dataset == 'DAPHNE':
        preprocessDaphne()
    elif dataset == 'INHALE':
        preprocessInhale()
    elif dataset == 'London':
        preprocessLondon()

def preprocessDaphne():
    datapath = "D:\AHALYA\DataScience\Dissertation\Daphne\Daphne\Daphne device logging Main Study_29012021.xlsx"
    daphne_static_airspeck = pd.DataFrame()
    for sheetName in ['Asthma Cohort WP1.1','MC Cohort']:
        logs = pd.read_excel(datapath, sheet_name=sheetName, engine='openpyxl')
        # Set Subject ID as index but keep column as well
        logs = logs.set_index('Subject ID')
        logs['Subject ID'] = logs.index
        dfsampleperiod = "60T"
        for idx, row in logs.iterrows():
            subj_id = row.name

            frame = load_static_airspeck_file(subj_id, 'daphne', suffix_filename="_home", upload_type='automatic',
                                              subject_visit_number=int(row.get(key='Visit number')))
            frame1 = load_static_airspeck_file(subj_id, 'daphne', suffix_filename="_school", upload_type='automatic',
                                               subject_visit_number=int(row.get(key='Visit number')))
            frame2 = load_static_airspeck_file(subj_id, 'daphne', suffix_filename="_community", upload_type='automatic',
                                               subject_visit_number=int(row.get(key='Visit number')))

            if frame is not None:
                frame = frame[['timestamp', 'pm2_5']]
                frame = frame.resample(dfsampleperiod, on='timestamp').mean()
                frame['timestamp'] = frame.index
                frame['SubjectID'] = subj_id
                frame['UUID'] = row.get('Static sensor inside home ID')
                frame['latitude'] = row.get('Home/Community GPS Latitude')
                frame['longitude'] = row.get('Home/Community GPS Longitude')
            if frame1 is not None:
                frame1 = frame1[['timestamp', 'pm2_5']]
                frame1 = frame1.resample(dfsampleperiod, on='timestamp').mean()
                frame1['timestamp'] = frame1.index
                frame1['UUID'] = row.get('School static sensor ID')
                frame1['SubjectID'] = subj_id
                frame1['latitude'] = row.get('School GPS Latitude')
                frame1['longitude'] = row.get('School GPS Longitude')
            if frame2 is not None:
                frame2 = frame2[['timestamp', 'pm2_5']]
                frame2 = frame2.resample(dfsampleperiod, on='timestamp').mean()
                frame2['timestamp'] = frame2.index
                frame2['UUID'] = row.get('Nearest community static sensor ID')
                frame2['SubjectID'] = subj_id
                frame2['latitude'] = row.get('Home/Community GPS Latitude')
                frame2['longitude'] = row.get('Home/Community GPS Longitude')
            daphne_static_airspeck = daphne_static_airspeck.append(frame)
            daphne_static_airspeck = daphne_static_airspeck.append(frame1)
            daphne_static_airspeck = daphne_static_airspeck.append(frame2)
    daphne_static_airspeck.dropna(inplace=True)
    daphne_static_airspeck.sort_index(inplace=True)
    daphne_static_airspeck['weekday'] = daphne_static_airspeck['timestamp'].dt.dayofweek
    daphne_static_airspeck['hour'] = daphne_static_airspeck['timestamp'].dt.hour
    print(daphne_static_airspeck.shape)
    daphne_static_airspeck.to_csv("checkdata.csv")
    osmRoadData = readOSMGeoJson('PreProcessing\\osmfiles\\map_daphne_model2_lines.geojson')
    osmPointData = readOSMGeoJson('PreProcessing\\osmfiles\\map_daphne_model2.geojson')

    osmRoads = formatRoads(osmRoadData)
    osmRoads['road'] = osmRoads.apply(lambda row: geometry.LineString(row['location']), axis=1)

    sensorsList = daphne_static_airspeck[['UUID', 'latitude', 'longitude']]
    #sensorsList.drop(['timestamp'], axis=1, inplace=True)
    sensorsList.drop_duplicates(inplace=True)

    minimumDist = []
    roadMultiline = createMultilineRoads(osmRoads)
    minimumDistRest = []
    osmRestuarants = formatRestuarants(osmPointData)
    restuarantPoints = geometry.MultiPoint(osmRestuarants.location.to_list())

    for index, row in sensorsList.iterrows():
        locationUsed = (utm.from_latlon(row['latitude'], row['longitude']))[0:2]
        minimumDist.append(findDistToRoadImproved(locationUsed, roadMultiline))
        p1 = geometry.Point(locationUsed)
        dista = np.log(restuarantPoints.distance(p1) + 10)
        minimumDistRest.append(dista)
    sensorsList['minDistToRoad'] = minimumDist
    sensorsList['minDistToRest'] = minimumDistRest
    roadDict = buildRoadDictionary()
    sensorsList['closestRoadLoc'] = sensorsList.apply(lambda row: closest_roadLoc(row, roadDict), axis=1)
    sensorsList.to_csv("sensorsList.csv")
    traffic_data_file = 'PreProcessing\\trafficfiles\\traffic_data.csv'
    columns = ['timestamp', 'Location code', 'Confidence', 'FF', 'Jam Factor', 'Speed Restricted', 'Speed Unrestricted',
               'ty']
    ts = pd.read_csv(traffic_data_file, names=columns)
    ts['timestamp'] = pd.to_datetime(ts['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    ts['weekday'] = ts['timestamp'].dt.dayofweek
    ts['hour'] = ts['timestamp'].dt.hour
    ts = ts.groupby(['Location code','weekday','hour']).mean()
    ts = ts.reset_index()
    ts = ts[['Jam Factor', 'Location code','weekday','hour']]

    #sensorsList = pd.merge(sensorsList, tsJamFactor, left_on='closestRoadLoc', right_on='LocationCode')
    #sensorsList.drop('LocationCode', axis=1, inplace=True)

    sensorsList.to_csv("sensorsList.csv")
    daphne_merged = pd.merge(daphne_static_airspeck, sensorsList, on=['UUID','latitude','longitude'])
    daphne_traffic_merged = pd.merge(daphne_merged, ts, left_on=['closestRoadLoc', 'weekday', 'hour'],
                                     right_on=['Location code', 'weekday', 'hour'])

    daphne_traffic_merged['pm2_5_mean'] = 0
    meanArr = np.zeros(len(daphne_traffic_merged))
    for i in range(len(daphne_traffic_merged) - 1):
        meanArr[i] = (daphne_traffic_merged.iloc[i]['pm2_5'] + daphne_traffic_merged.iloc[i + 1]['pm2_5']) / 2
    meanArr[len(daphne_traffic_merged) - 1] = daphne_traffic_merged.iloc[len(daphne_traffic_merged) - 1]['pm2_5']
    daphne_traffic_merged['pm2_5_mean'] = meanArr
    daphne_traffic_merged.to_csv("merged_traffic.csv")

def buildRoadDictionary():
    roadDict = {}
    coOrdinatesList = []
    shape_dir = 'PreProcessing\\trafficfiles\\shape_files\\csv'
    files = listdir(shape_dir)
    for file in files:
        df = pd.read_csv(join(shape_dir, file), names=['Location code', 'x', 'lat', 'long'])
        coOrdinates = [[lat, lng] for lat, lng in zip(df['lat'], df['long'])]
        shapeId = (str(file)[4:-4])
        road = geometry.LineString(coOrdinates)
        roadDict[shapeId] = road
    return roadDict

def closest_roadLoc(row, roadDict):
    latitude = row['latitude']
    longitude = row['longitude']
    point = geometry.Point(latitude, longitude)
    minDist = point.distance(roadDict[list(roadDict.keys())[0]])
    closestRoadLoc = list(roadDict.keys())[0]
    for roadLoc in roadDict:
        distance = point.distance(roadDict[roadLoc])
        if distance < minDist:
            minDist = distance
            closestRoadLoc = roadLoc
    return closestRoadLoc

def calibratePM(row, refSensorId, calibrationFactor, useIntercept = False):
    pm2_5 = row['pm2_5']
    if(row['UUID'] != refSensorId):
        if (useIntercept):
            pm2_5 = (row['pm2_5'] * calibrationFactor[0]) + calibrationFactor[1]
        else:
            pm2_5 = row['pm2_5'] * calibrationFactor[0]
    return pm2_5

def preprocessInhale():
    resolutiontime = 60
    gridXCount = 670
    gridYCount = 260
    resolutionMeters = 15

    startPoint = (51.469158, -0.339978)
    newstartPoint = list(utm.from_latlon(startPoint[0], startPoint[1]))[0:2]
    endPoint = (
    (newstartPoint[0] + (resolutionMeters * gridXCount)), (newstartPoint[1] + (resolutionMeters * gridYCount)), 30, 'U')
    endPointLatLong = utm.to_latlon(*endPoint)

    print(startPoint, endPointLatLong)
    dataPath = 'D:\\Desktop\\Project\\INHALE_modelling\\'
    inhaleDf, monitorlocs, monitorlocsNames = readPollutionDataInhale(dataPath, resolutiontime)
    osmRoadData = readOSMGeoJson('PreProcessing\\osmfiles\\mapINHALE_lines.geojson ')
    osmPointData = readOSMGeoJson('PreProcessing\\osmfiles\\mapINHALE.geojson')
    osmRoads = formatRoads(osmRoadData)
    osmRoads['road'] = osmRoads.apply(lambda row: geometry.LineString(row['location']), axis=1)
    calibrationFactor = [0.9071589753052491, 0.3995782108732513]
    refSensorId = 'C2AC7B090C556FF0'
    inhaleDf['pm2_5'] = inhaleDf.apply(lambda row: calibratePM(row, refSensorId, calibrationFactor), axis=1)
    yAmount = (endPointLatLong[0] - startPoint[0]) / gridYCount
    xAmount = (endPointLatLong[1] - startPoint[1]) / gridXCount

    utmGrid = np.zeros((gridYCount, gridXCount, 2))
    latlongGrid = np.zeros((gridYCount, gridXCount, 2))
    endPoint = (
    (newstartPoint[0] + (resolutionMeters * gridXCount)), (newstartPoint[1] + (resolutionMeters * gridYCount)), 30, 'U')
    for i in range(gridXCount):
        for j in range(gridYCount):
            pointlatlong = (startPoint[0] + (yAmount * j), startPoint[1] + (xAmount * i))
            utmGrid[j, i, :] = utm.from_latlon(*pointlatlong)[0:2]
            latlongGrid[j, i, :] = pointlatlong
    trafficFolder = 'D:\\Desktop\\Project\\INHALE_traffic_subset'
    minimumDate = dt.datetime(2021, 2, 2, 0, 0, 0)
    maximumDate = dt.datetime(2021, 7, 6, 0, 0, 0)
    locsDictFull, trafficDictFull, trafficHourly = loadTrafficData(trafficFolder, minimumDate, maximumDate,
                                                                   resolutiontime)
    trafficHourlyReduced = trafficHourly[trafficHourly['traffic'] > 0]
    trafficPointsAll, trafficPointsAllSmall, trafficPointsOverlapAll, trafficPointsOverlapAllSmall, roadPointsAll = aligntrafficwithgrids(
        osmRoads, newstartPoint, gridXCount, gridYCount, resolutionMeters, locsDictFull, utmGrid)
    weatherINHALE = pd.read_csv('PreProcessing\\weather\\weather_inhale.csv')
    weatherINHALE['Time'] = pd.to_datetime(weatherINHALE['Time']).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    inhaleDf['Time'] = inhaleDf.index
    merged = pd.merge(inhaleDf, weatherINHALE, on='Time')
    merged.index = merged['Time']
    trafficHourlyReduced.index = pd.to_datetime(trafficHourlyReduced.index).tz_localize('UTC').tz_convert(
        'Europe/London')
    weatherPmTraffic = pd.merge(merged, trafficHourlyReduced, left_index=True, right_index=True)

    roadGrid = formatRoadwithGrids(gridXCount, gridYCount, resolutionMeters, newstartPoint, roadPointsAll, osmRoads,
                                   utmGrid)
    distGrid = createRoadDistanceGrid(osmRoads, gridYCount, gridXCount, utmGrid)
    avgPmAtEachSensor = inhaleDf.groupby(['UUID', 'deploymentId'])['pm2_5'].mean()

    minimumDist = []
    roadMultiline = createMultilineRoads(osmRoads)
    for a in monitorlocs:
        locationUsed = (utm.from_latlon(*a))[0:2]
        minimumDist.append(findDistToRoadImproved(locationUsed, roadMultiline))
    minimumDist = np.array(minimumDist)

    trafficgrid, trafficgridsmall, trafficAvgGrid = formattrafficgridwithtime(gridXCount,
                                                                              gridYCount,
                                                                              resolutionMeters,
                                                                              newstartPoint,
                                                                              trafficDictFull,
                                                                              trafficPointsAll,
                                                                              trafficPointsAllSmall,
                                                                              trafficPointsOverlapAll,
                                                                              trafficPointsOverlapAllSmall,
                                                                              utmGrid)
    weatherPm2 = addTrafficToWeatherPmModified(weatherPmTraffic, monitorlocs, monitorlocsNames, resolutionMeters,
                                               newstartPoint, trafficgrid)
    restGrid = createRestGrid(osmPointData, gridYCount, gridXCount, utmGrid)

    osmRestuarants = formatRestuarants(osmPointData)
    restuarantPoints = geometry.MultiPoint(osmRestuarants.location.to_list())
    minimumDistRest = []
    for a in monitorlocs:
        locationUsed = (utm.from_latlon(*a))[0:2]
        p1 = geometry.Point(locationUsed)
        dista = np.log(restuarantPoints.distance(p1) + 10)
        minimumDistRest.append(dista)
    minimumDistRest = np.array(minimumDistRest)

    weatherTrafficRoadData = compileVariablesInhale(monitorlocs, monitorlocsNames, newstartPoint, roadGrid,
                                                      trafficAvgGrid, trafficgrid, weatherPm2, resolutionMeters, 60,
                                                      minimumDist, minimumDistRest)
    weatherTrafficRoadData.to_csv('weatherTrafficRoadData_INHALE.csv')
    weatherPm2.to_csv('weatherPm2_INHALE.csv')

def preprocessLondon():

    minimumDate = dt.datetime(2020, 6, 6, 0, 0, 0)
    maximumDate = dt.datetime(2020, 8, 10, 0, 0, 0)
    maximumDate2 = dt.datetime(2020, 11, 10, 0, 0, 0)
    resolutiontime = 60
    gridXCount = 100
    gridYCount = 160
    resolutionMeters = 15

    startPoint = (51.483536, -0.181310)
    newstartPoint = list(utm.from_latlon(startPoint[0], startPoint[1]))[0:2]
    endPoint = (
    (newstartPoint[0] + (resolutionMeters * gridXCount)), (newstartPoint[1] + (resolutionMeters * gridYCount)), 30, 'U')
    endPointLatLong = utm.to_latlon(*endPoint)

    pmvalues15, monitorlocs, monitorNames = readPollutionData("D:\\AHALYA\\DataScience\\Dissertation\\Code\\SamuelHonorsProjectWithReadme\\SamuelHonorsProject\\calibratedairreadings2", minimumDate,
                                                              maximumDate, resolutiontime)
    pmvaluesAll, monitorlocs, monitorNames = readPollutionData("D:\\AHALYA\\DataScience\\Dissertation\\Code\\SamuelHonorsProjectWithReadme\\SamuelHonorsProject\\calibratedairreadings2", minimumDate,
                                                               maximumDate2, resolutiontime)

    dfsampleperiod = str(resolutiontime) + "T"
    pmvalues = pmvalues15.resample(dfsampleperiod).mean()
    pmValuesall = pmvaluesAll.resample("15T").mean()
    osmRoadData = readOSMGeoJson('PreProcessing\\osmfiles\\kensingtonmaplines2.geojson')
    osmPointData = readOSMGeoJson('PreProcessing\\osmfiles\\kensingtonmappoints2.geojson')

    osmRoads = formatRoads(osmRoadData)
    osmRoads['road'] = osmRoads.apply(lambda row: geometry.LineString(row['location']), axis=1)

    calibrationvalues = {'905801CA0E1F1D11 pm2_5': [1, 0], '90E275086B4D99A3 pm2_5': [1.053187368001334, 0.0], 'D849BF7848210A4A pm2_5': [1.137764827889944, 0.0], 'E7C0CD8112BA98D7 pm2_5': [0.9402721589804153, 0.0]}
    calibrationvalues2 = {'905801CA0E1F1D11 pm2_5': [1, 0], '90E275086B4D99A3 pm2_5': [1.0405128836449076, 0.5606541606910298], 'D849BF7848210A4A pm2_5': [1.1567735516718558, -0.7785780151824042],'E7C0CD8112BA98D7 pm2_5': [0.9694651737918651, -1.4304941378410376]}
    pmValuesCalibrated = pmvalues.copy()
    pmValuesCalibratedWithI = pmvalues.copy()
    for a in calibrationvalues:
        pmValuesCalibrated[a] = (pmvalues[a] * calibrationvalues[a][0]) + calibrationvalues[a][1]
        pmValuesCalibratedWithI[a] = (pmvalues[a] * calibrationvalues2[a][0]) + calibrationvalues2[a][1]
    pmValuesCalibrated['pm2_5average'] = pmValuesCalibrated.mean(axis=1)

    yAmount = (endPointLatLong[0] - startPoint[0]) / gridYCount
    xAmount = (endPointLatLong[1] - startPoint[1]) / gridXCount

    utmGrid = np.zeros((gridYCount, gridXCount, 2))
    latlongGrid = np.zeros((gridYCount, gridXCount, 2))
    endPoint = (
    (newstartPoint[0] + (resolutionMeters * gridXCount)), (newstartPoint[1] + (resolutionMeters * gridYCount)), 30, 'U')
    for i in range(gridXCount):
        for j in range(gridYCount):
            pointlatlong = (startPoint[0] + (yAmount * j), startPoint[1] + (xAmount * i))
            utmGrid[j, i, :] = utm.from_latlon(*pointlatlong)[0:2]
            latlongGrid[j, i, :] = pointlatlong
    trafficFolder = 'PreProcessing\\trafficfiles\\traffic2'
    locsDictFull, trafficDictFull, trafficHourly = loadTrafficData(trafficFolder, minimumDate, maximumDate,
                                                                   resolutiontime)

    weatherPath = 'PreProcessing\\weather\\newScraper'

    weatherPmTraffic = loadWeatherDataAddTraffic(weatherPath, trafficHourly, pmValuesCalibrated)
    condition = pd.get_dummies(weatherPmTraffic.Condition, prefix="Condition")
    weatherPmTraffic[condition.columns.to_list()] = condition

    trafficPointsAll, trafficPointsAllSmall, trafficPointsOverlapAll, trafficPointsOverlapAllSmall, roadPointsAll = aligntrafficwithgrids(
        osmRoads, newstartPoint, gridXCount, gridYCount, resolutionMeters, locsDictFull, utmGrid)
    roadGrid = formatRoadwithGrids(gridXCount, gridYCount, resolutionMeters, newstartPoint, roadPointsAll, osmRoads,
                                   utmGrid)
    distGrid = createRoadDistanceGrid(osmRoads, gridYCount, gridXCount, utmGrid)

    minimumDist = []
    roadMultiline = createMultilineRoads(osmRoads)
    for a in monitorlocs:
        locationUsed = (utm.from_latlon(*a))[0:2]
        minimumDist.append(findDistToRoadImproved(locationUsed, roadMultiline))
    minimumDist = np.array(minimumDist)

    trafficgrid, trafficgridsmall, trafficAvgGrid = formattrafficgridwithtime(gridXCount,
                                                                              gridYCount,
                                                                              resolutionMeters,
                                                                              newstartPoint,
                                                                              trafficDictFull,
                                                                              trafficPointsAll,
                                                                              trafficPointsAllSmall,
                                                                              trafficPointsOverlapAll,
                                                                              trafficPointsOverlapAllSmall,
                                                                              utmGrid)
    weatherPm2 = addTrafficToWeatherPm(weatherPmTraffic, monitorlocs, monitorNames, resolutionMeters, newstartPoint,
                                       trafficgrid)
    weatherPm2.to_csv("weatherPm2.csv")
    weatherPm3 = addTrafficToWeatherPm(weatherPmTraffic, monitorlocs, monitorNames, resolutionMeters, newstartPoint,
                                       trafficgridsmall)
    restGrid = createRestGrid(osmPointData, gridYCount, gridXCount, utmGrid)

    osmRestuarants = formatRestuarants(osmPointData)
    restuarantPoints = geometry.MultiPoint(osmRestuarants.location.to_list())
    minimumDistRest = []
    for a in monitorlocs:
        locationUsed = (utm.from_latlon(*a))[0:2]
        p1 = geometry.Point(locationUsed)
        dista = np.log(restuarantPoints.distance(p1) + 10)
        minimumDistRest.append(dista)
    minimumDistRest = np.array(minimumDistRest)
    weatherPmVariables = []
    weatherTrafficRoadData = compileVariables(monitorlocs, newstartPoint, roadGrid, trafficAvgGrid, trafficgrid,
                                              weatherPmTraffic, resolutionMeters, weatherPmVariables, 60, minimumDist,
                                              minimumDistRest)
    pd.DataFrame(weatherTrafficRoadData).to_csv("weatherTrafficRoadData.csv")
    pmValuesCalibrated.to_csv("pmValuesCalibrated.csv")