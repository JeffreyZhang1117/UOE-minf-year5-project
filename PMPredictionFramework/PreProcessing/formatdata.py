import os

from shapely import geometry
import numpy as np
import pandas as pd
import glob
import re
#import mpu
from scipy.spatial import distance
from scipy.stats import norm, zscore
from pathlib import Path
import datetime as dt
import folium
import geojson
import pyproj
import utm
import math
from shapely import geometry,affinity
import xml.etree.ElementTree as ET
from sklearn import preprocessing,metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
def readPollutionData(datapath,minimumDate,maximumDate,resolutiontime):
    all_paths = list(Path(datapath).rglob('*.csv'))
    allPm15 = []
    monitorlocs = []
    monitorlocsNames = []
    averagePm = []
    for path in all_paths:
        df = pd.read_csv(path)
        df['timestamp'] = pd.DatetimeIndex(df['timestamp']).tz_convert(None)
        dfsampleperiod = str(resolutiontime) +"T"
        averagePm.append(df.resample(dfsampleperiod,on='timestamp').mean().loc[minimumDate:maximumDate][0:-1]['pm2_5'])
        name = (str(path)[-20:-4])

        #columnnames.append(name + " pm2_5")


        #monitorlocs[name] = ((df[df['gpsLatitude'] > 50]['gpsLatitude'].mean(),df[df['gpsLongitude'] < -0.14]['gpsLongitude'].mean()))
        monitorlocsNames.append(str(path)[-20:-4])
        monitorlocs.append((df[df['gpsLatitude'] > 50]['gpsLatitude'].median(),df[df['gpsLongitude'] < -0.14]['gpsLongitude'].median()))

        allPm15.append(df.resample("15T",on='timestamp').mean().loc[minimumDate:maximumDate][0:-1]['pm2_5'].rename(name + " pm2_5"))
    print(monitorlocs)
    print(monitorlocsNames)
    allPm15 = (pd.concat([a for a in allPm15],axis=1))
    allPm15['pm2_5average'] = allPm15.mean(axis=1)
    for names in monitorlocsNames:
        allPm15[(names + " pm2_5")] = allPm15[(names + " pm2_5")].fillna(allPm15.pm2_5average)

    return (allPm15.ffill(), monitorlocs,monitorlocsNames)

def readPollutionDataAndTemp(datapath,minimumDate,maximumDate,resolutiontime):
    all_paths = list(Path(datapath).rglob('*.csv'))
    allPm15 = []
    monitorlocs = []
    monitorlocsNames = []
    averagePm = []
    for path in all_paths:
        df = pd.read_csv(path)
        df['timestamp'] = pd.DatetimeIndex(df['timestamp']).tz_convert(None)
        dfsampleperiod = str(resolutiontime) +"T"
        averagePm.append(df.resample(dfsampleperiod,on='timestamp').mean().loc[minimumDate:maximumDate][0:-1]['pm2_5'])
        name = (str(path)[-20:-4])

        #columnnames.append(name + " pm2_5")


        #monitorlocs[name] = ((df[df['gpsLatitude'] > 50]['gpsLatitude'].mean(),df[df['gpsLongitude'] < -0.14]['gpsLongitude'].mean()))
        monitorlocsNames.append(str(path)[-20:-4])
        monitorlocs.append((df[df['gpsLatitude'] > 50]['gpsLatitude'].median(),df[df['gpsLongitude'] < -0.14]['gpsLongitude'].median()))

        allPm15.append(df.resample("15T",on='timestamp').mean().loc[minimumDate:maximumDate][0:-1]['pm2_5'].rename(name + " pm2_5"))
        allPm15.append(df.resample("15T",on='timestamp').mean().loc[minimumDate:maximumDate][0:-1]['temperature'].rename(name + " temperature"))
    print(monitorlocs)
    print(monitorlocsNames)
    allPm15 = (pd.concat([a for a in allPm15],axis=1))
    allPm15['pm2_5average'] = allPm15.mean(axis=1)
    for names in monitorlocsNames:
        allPm15[(names + " pm2_5")] = allPm15[(names + " pm2_5")].fillna(allPm15.pm2_5average)
        
    return (allPm15.ffill(), monitorlocs,monitorlocsNames)
    
  

def readOSMGeoJson(path):
    with open(path, encoding="utf-8") as f:
        osmlines = geojson.load(f)
        return osmlines

def loadTrafficData(folder, minimumDate,maximumDate,resolutiontime):
    currentPaths = []
    currentPathsDates = []

    all_paths = list(Path(folder).rglob('traffic*.xml'))
    all_paths.sort()

    for paths in all_paths:
        date = (str(paths)[-24:-4])
        date_time = dt.datetime.strptime(date,'%d-%m-%Y-%Hh%Mm%Ss')
        #get only the data in the right range
        if(date_time < maximumDate):
            if(date_time >= minimumDate):
                currentPaths.append(paths)
                currentPathsDates.append(date_time)
    #group them by the time resolution
    dfsampleperiod = str(resolutiontime) +"T"
    groupedList= pd.DataFrame(currentPaths,currentPathsDates).resample(dfsampleperiod).apply(list)[0]
    trafficDictFull = []
    hourlyTrafficFull = []
    trafficTimeStamps = []
    locsDictFull = {}
    trafficDict = {}
    #go throught
    for daysGrouped in groupedList:

        for tr in trafficDict:
            trafficDict[tr] = np.array([0,0,0])
        for paths in daysGrouped:
            root = ET.parse(paths).getroot()[0]
            date = (str(paths)[-24:-4])
            date_time = dt.datetime.strptime(date,'%d-%m-%Y-%Hh%Mm%Ss')
            trafficTimeStamps.append(date_time)
            currentTraffic = 0
            for rootobj in root:
                for fi in rootobj[0]:
                    locs = [sh.text for sh in fi[1:-1]]
                    inputsTraffic = np.array([float(fi[-1].attrib['FF']),float(fi[-1].attrib['JF']),1])
                    key = hash(str(locs))
                    if key in trafficDict:
                        trafficDict[key][2] += 1
                        trafficDict[key][0] += inputsTraffic[0]
                        trafficDict[key][1] += inputsTraffic[1]
                        currentTraffic += inputsTraffic[1]
                    else:
                        trafficDict[key] = inputsTraffic
                        #locsDict[key] = [list(utm.from_latlon(*list(map(float, a.split(',')))))[0:2] for a in (locs[0].split(' ')[0],locs[-1].split(' ')[0])]
                        locsDictFull[key] = [list(utm.from_latlon(*list(map(float, a.split(',')))))[0:2] for b in locs for a in (b[0:-1].split(' '))]
            hourlyTrafficFull.append(currentTraffic)
        trafficDictAvg = {}
        for key in trafficDict:
            if(trafficDict[key][2] != 0):
                trafficDictAvg[key] = np.array([(trafficDict[key][0] / trafficDict[key][2]),(trafficDict[key][1] / trafficDict[key][2])])
            else:
                trafficDictAvg[key] = np.zeros(2)
        trafficDictFull.append(trafficDictAvg)
        trafficHourly = pd.DataFrame(list(zip(trafficTimeStamps, hourlyTrafficFull)), columns =['Time', 'traffic']).resample(dfsampleperiod,on='Time').mean()
    return (locsDictFull,trafficDictFull,trafficHourly)

def loadWeatherDataAddTraffic(path,trafficHourly,allPm):
    all_paths = list(Path(path).rglob('*.csv'))
    averagePm = []


    li = []
    for path in all_paths:
        df = pd.read_csv(path)
        del df['Column 2']
        del df['Column 9']
        df.columns = ['Time','Temperature','Condition','WindSpeed','WindDirection','Humidity','Pressure' ]
        df.WindDirection = df.WindDirection.str.split('from ').str[1].str.split('°').str[0]
        df.Temperature = (((df.Temperature.str.split('°').str[0].astype(float)) - 32) *(5/9)).round(decimals=0).astype(int)
        (month, day) = df.Time[0].split(', ')[1].split(' ')
        testVar = day + month + '2020' + df.Time.str.split(' ').str[0].str.split(':').str[0].astype(int).map("{0:0=2d}".format).astype(str)+  df.Time.str.split(' ').str[0].str.split(':').str[1] + df.Time.str.split(' ').str[1].str[0:2]
        df['Time'] = pd.to_datetime(testVar, format =  '%d%b%Y%I%M%p')
        df['WindSpeed'] = df['WindSpeed'].str.split(' ').str[0].replace('No', '0').fillna(0).astype(int)
        df.set_index('Time',inplace=True)
        df.Humidity = df.Humidity.str.split('%').str[0].astype(int)
        df.Pressure = df.Pressure.str.split(' \"').str[0].astype(float)
        df.WindDirection = df.WindDirection.astype(int)
        li.append(df)
    weatherDataOriginal = pd.concat(li)
    weatherDataOriginal.sort_index(inplace=True)
    weatherRain = weatherDataOriginal.Condition.str.split(pat= '.').str[-3].fillna('None').astype("category")
    weatherDataOriginal['Rain'] = (weatherRain != 'None')
    weatherData = weatherDataOriginal.resample("60T").mean().join(weatherDataOriginal.Condition.resample("60T").pad().bfill().astype("category"))
   

    weatherPmTraffic = allPm.merge(weatherData,left_index=True,right_index=True).merge(trafficHourly,left_index=True,right_index=True)
    weatherPmTraffic.to_csv('weatherPmTraffic.csv')

    #weatherPmTraffic['pm2_5average'] = weatherPmTraffic['pm2_5'].mean(axis=1)


    weatherPmTraffic['Condition'] = weatherPmTraffic['Condition'].str.split(pat= '.').str[-2].str.strip()
    weatherPmTraffic = weatherPmTraffic.fillna(method='ffill')

    return weatherPmTraffic

def formatRoads(osmlines):
    osmRoads = []
    for allFeatures in osmlines.features:
        if 'highway' in allFeatures['properties']:
            roadinfo = allFeatures['properties']
            roadinfo['location'] = [list(utm.from_latlon(x[1],x[0]))[0:2] for x in allFeatures['geometry']['coordinates']]
            osmRoads.append(roadinfo)
    osmRoads = pd.DataFrame.from_dict(osmRoads)
    osmRoads = osmRoads[osmRoads.highway.isin(['residential','living_street','tertiary','trunk','secondary','primary','pedestrian','tertiary_link','trunk_link','primary_link','secondary_link'])]


    osmRoads = osmRoads.set_index('osm_id')
    return osmRoads
def formatRestuarants(osmPointData):
    allEater = []
    for allFeatures in osmPointData.features:
        isEatery = False
        if 'name' in (allFeatures['properties']) and 'other_tags' in (allFeatures['properties']):
            
            currentPlace = allFeatures['properties']['other_tags']
            thetype = ""
            if(currentPlace.find("\"amenity\"=>\"restaurant\"") != -1):
                isEatery = True
                thetype = "restaurant"
            elif(currentPlace.find("\"amenity\"=>\"cafe\"") != -1):
                isEatery = True
                thetype = "cafe"
            elif(currentPlace.find("\"amenity\"=>\"fast_food\"") != -1):
                isEatery = True
                thetype = "fastfood"
            elif(currentPlace.find("\"amenity\"=>\"bar\"") != -1):
                isEatery = True
                thetype = "bar"
            elif(currentPlace.find("\"amenity\"=>\"pub\"") != -1):
                isEatery = True
                thetype = "pub"
            if(isEatery):
                currentProperty = {}
                coords = allFeatures['geometry']['coordinates']
                utmcoords = list(utm.from_latlon(coords[1],coords[0]))[0:2]
                osmid = (allFeatures['properties']["osm_id"])
                currentProperty['id'] = osmid
                name = allFeatures['properties']['name']
                currentProperty['name'] = name
                currentProperty['type'] = thetype
                currentProperty['location'] = utmcoords
                p1 = geometry.Point(utmcoords)
                currentProperty['locationShape'] = p1
                allEater.append(currentProperty)
    osmFood = pd.DataFrame.from_dict(allEater)           
    osmFood = osmFood.set_index('id')
    return osmFood

def createRestGrid(osmPointData,gridYCount,gridXCount,utmGrid):
    osmRestuarants = formatRestuarants(osmPointData)
    restuarantPoints = geometry.MultiPoint(osmRestuarants.location.to_list())
    restGrid = np.zeros((gridYCount,gridXCount))
    for i in range(gridXCount):
     
        for j in range(gridYCount):
            #clocation = (newstartPoint[0] + resolutionMeters*i),(newstartPoint[1] + resolutionMeters*j)
            clocation = utmGrid[j,i,:]
            p1 = geometry.Point(clocation)
            restGrid[j,i] = np.log((restuarantPoints.distance(p1)) + 10)
    return restGrid


def dist(x,y):
  
    return math.sqrt(math.pow((y[0] - x[0]),2) + math.pow((y[1] - x[1]),2))

def aligntrafficwithgrids(osmRoads,newstartPoint,gridXCount,gridYCount,resolutionMeters,locsDictfull,utmGrid):
    threshold = 150
    smallerthreshold = 80

    osmids = list(osmRoads.index)
    roadPointsAll = []
    
    boxSize = resolutionMeters
    locsDictRoads = {}
    trafficPointsAll = []
    trafficPointsAllSmall = []
    trafficPointsOverlapAll = []
    trafficPointsOverlapAllSmall = []


    for trafficRoads in locsDictfull:
        road = geometry.LineString(locsDictfull[trafficRoads])
        locsDictRoads[trafficRoads] = road
    for i in range(gridXCount):
        print(i,end=" ")
        for j in range(gridYCount):
            roadPoints = []
            location = utmGrid[j,i,:]
            #location = [(newstartPoint[0] + boxSize*i),(newstartPoint[1] + boxSize*j)]

            minDistance = 10000
            p2 = geometry.Point(location)
            trafficpoints = []
            trafficpoints80 = []
            
            for trafficRoads in locsDictRoads:
                
                dist2traffic = p2.distance(locsDictRoads[trafficRoads])
                if(dist2traffic < threshold):

                    trafficpoints.append(trafficRoads)
                    if(dist2traffic < smallerthreshold):
                        trafficpoints80.append(trafficRoads)
            trafficPointsAll.append(trafficpoints)
            trafficPointsAllSmall.append(trafficpoints80)
            
            
            p2scaled = geometry.Point(location).buffer(threshold)
            p2scaled80 = geometry.Point(location).buffer(smallerthreshold)
            trafficpointsOverlap = np.zeros(len(trafficpoints))
            trafficpointsOverlapSmall = np.zeros(len(trafficpoints80))
            
            for k in range(len(trafficpoints)):
                trafficpointsOverlap[k] = p2scaled.intersection(locsDictRoads[trafficpoints[k]]).length
            for k in range(len(trafficpoints80)):
                trafficpointsOverlapSmall[k] = p2scaled80.intersection(locsDictRoads[trafficpoints80[k]]).length
            trafficPointsOverlapAll.append(trafficpointsOverlap / 1000)
            trafficPointsOverlapAllSmall.append(trafficpointsOverlapSmall / 1000)
            for a, roadLocs in enumerate(osmRoads.location):
                dist1 = dist(roadLocs[0],location)
                dist2 = dist(roadLocs[-1],location)
                distBetweenline = dist(roadLocs[-1],roadLocs[0])
                preDist = min(dist1,dist2)
                if(preDist < threshold):
                    roadPoints.append(osmids[a])
                elif(preDist < (distBetweenline/2) + threshold*2):
                    
                    AB = geometry.LineString(roadLocs)
                    if( (p2.distance(AB)) < 250):
                        roadPoints.append(osmids[a])
            roadPointsAll.append(roadPoints)
    return trafficPointsAll,trafficPointsAllSmall, trafficPointsOverlapAll, trafficPointsOverlapAllSmall, roadPointsAll

def formatRoadwithGrids(gridXCount, gridYCount,boxSize,newstartPoint,roadPointsAll,osmRoads,utmGrid):
    
    currentThreshold = 60
    foodDistance = 300
    roadVariants = {'trunk': 0,
                    'trunk_link': 0,
                    'primary': 1,
                    'primary_link': 1,
                    'secondary': 2,
                    'secondary_link': 2,
                    'tertiary':3,
                    'tertiary_link':3,
                    'residential': 4,
                    'pedestrian':5, 
                    }

    grid = np.zeros((gridYCount,gridXCount,6))

    winddistort=1
    print(len(roadVariants))
    for i in range(gridXCount):
        
        print(i,end=" ")
        for j in range(gridYCount):
            
            #clocation = (newstartPoint[0] + boxSize*i),(newstartPoint[1] + boxSize*j)
            clocation = utmGrid[j,i,:]
            goodTrafficPoint = [0,0]
            allRoads = roadPointsAll[(i*gridYCount)+j]
            if(len(allRoads) > 0):
                p1 = geometry.Point(clocation).buffer(60)
                #p1Scaled = affinity.scale(p1,1,winddistort,origin=(clocation[0],clocation[1]-30))
                p1Scaled = p1
                p2 = geometry.Point(clocation)
                roads = osmRoads.loc[allRoads].location
                roadType = osmRoads.loc[allRoads].highway
                for k,road in enumerate(roads):
                    AB = geometry.LineString(road)
                    #print(roadVariants[roadType[k]])
                    if(roadType[k] in roadVariants):
                        grid[j,i,roadVariants[roadType[k]]] += p1Scaled.intersection(AB).length
                        #currentRoadAm
    return grid
def formattrafficgridwithtime(gridXCount, gridYCount,boxSize,newstartPoint,trafficDictFull,trafficPointsAll,trafficPointsAllSmall, trafficPointsOverlapAll, trafficPointsOverlapAllSmall,utmGrid):
    allgrids15 = []
    allgrids15Small = []
    timer = 0
    for trafficDictOne in trafficDictFull:
        print(timer,end=" ")
        timer += 1
        trafficgrid = np.zeros((gridYCount,gridXCount))
        trafficgridSmall = np.zeros((gridYCount,gridXCount))
        for i in range(gridXCount):
            for j in range(gridYCount):
                goodTrafficPoint = [0,0]
                trafficPointKeys = trafficPointsAll[(i*gridYCount)+j]
                trafficPointoverlap = trafficPointsOverlapAll[(i*gridYCount)+j]
                trafficPointKeysSmall = trafficPointsAllSmall[(i*gridYCount)+j]
                trafficPointoverlapSmall = trafficPointsOverlapAllSmall[(i*gridYCount)+j]
                
                #clocation = (newstartPoint[0] + boxSize*i),(newstartPoint[1] + boxSize*j)
                clocation = utmGrid[j,i,:]
                totalTraffic = 0
                avgSpeed = 0
                maxSpeed = 0
                trafficRoadType = 0
                for k,tpKey in enumerate(trafficPointKeys):
                    if(tpKey in trafficDictOne):
                        totalTraffic += ((trafficPointoverlap[k]) * ((trafficDictOne[tpKey][1]) + (trafficPointoverlap[k])))
                        #trafficRoadType += ((trafficPointoverlap[k]) * trafficDictOne[tpKey][0])
                trafficgrid[j,i] = totalTraffic#2*np.log(totalTraffic + 1)
                #grid[i,j,2] = np.log(minDistance+0.1)
                for k,tpKey in enumerate(trafficPointKeysSmall):
                    if(tpKey in trafficDictOne):
                        totalTraffic += ((trafficPointoverlapSmall[k]) * trafficDictOne[tpKey][1])
                        #trafficRoadType += ((trafficPointoverlap[k]) * trafficDictOne[tpKey][0])
                trafficgridSmall[j,i] = totalTraffic#2*np.log(totalTraffic + 1)
                #grid[i,j,2] = np.log(minDistance+0.1)
        allgrids15.append(trafficgrid)
        allgrids15Small.append(trafficgridSmall)
    allgrids15 = np.array(allgrids15)
    allgrids15Small = np.array(allgrids15Small)
    trafficAvgGrid = np.mean(allgrids15Small,axis=0)
    #trafficAvgGrid = np.mean(allgrids15, axis=0)
    return allgrids15, allgrids15Small, trafficAvgGrid

def compileVariables(monitorlocs,newstartPoint,roadgrid2,trafficAvgGrid,allgrids,weatherPmTraffic,boxSize,weatherPmVariables,timeResolution,distFromRoad,distFromRest,liveTraffic=False):
    staticVariables = []
    resize = int(timeResolution / 60)
    
    roadgrid = np.log(roadgrid2.copy() + 4)
    #dfsampleperiod = str(timeResolution) +"T"
    #weatherPmTraffic = weatherPmTraffic.resample(dfsampleperiod).mean()

    for stations in monitorlocs:
        
        (xAmount, yAmount,r1,r2) = (utm.from_latlon(stations[0],stations[1]))
        dueEast = int((xAmount-newstartPoint[0]) / boxSize)
        dueNorth = int((yAmount - newstartPoint[1]) / boxSize)
        staticVariables.append(np.append(roadgrid[dueNorth,dueEast],trafficAvgGrid[dueNorth,dueEast]))
    staticVariables = np.array(staticVariables).transpose()
    
    print(staticVariables[0].shape)
    varFull = []
    #print(len(allgridsResize))
    for trafficgrids in allgrids:
        dynamicVariables = []
        for stations in monitorlocs:
            (xAmount, yAmount,r1,r2) = (utm.from_latlon(stations[0],stations[1]))
            dueEast = int((xAmount-newstartPoint[0]) / boxSize)
            dueNorth = int((yAmount - newstartPoint[1]) / boxSize)
            dynamicVariables.append(trafficgrids[dueNorth,dueEast])

        varFull.append(dynamicVariables)
    varFull = np.array(varFull)
    #print(varFull)
    #newstat = np.stack((distFromRoad,staticVariables[0] + staticVariables[1],staticVariables[2] + staticVariables[3]+staticVariables[4],staticVariables[6]))
    newstat = np.stack((distFromRoad,distFromRest,staticVariables[6]))




    ShapedStatic = np.tile(newstat,len(varFull)).transpose()

    weatherData = np.repeat(weatherPmTraffic[weatherPmVariables].to_numpy(),4,axis=0)




    ShapedStaticDynamic = np.hstack((ShapedStatic,varFull.reshape(1,-1).transpose(),weatherData))


    return ShapedStaticDynamic

#returns the log of the distance to road. It only finds the distance to roads of type "roadType"
def createMultilineRoads(osmRoads,roadType=['primary','primary_link','secondary','secondary_link','trunk']):
    return (geometry.MultiLineString(osmRoads[osmRoads.highway.isin(roadType)].road.to_list()))
def findDistToRoadImproved(location,roadMultiline):
    p1 = geometry.Point(location)
    return np.log(p1.distance(roadMultiline) + 1)
def createRoadDistanceGrid(osmRoads,gridYCount,gridXCount,utmGrid):
    distGrid = np.zeros((gridYCount,gridXCount))
    roadMultiline = createMultilineRoads(osmRoads)
    for i in range(gridXCount):
        for j in range(gridYCount):
            clocation = utmGrid[j,i,:]
            distGrid[j,i] = findDistToRoadImproved(clocation,roadMultiline)
    return distGrid
def findDistToRoad(osmRoads,newstartPoint,locationUsed,resolutionMeters,roadPointsAll,gridYCount,gridXCount,roadType=['primary','primary_link','secondary','secondary_link','trunk']):

    i = min(max(int((locationUsed[0]-newstartPoint[0]) / resolutionMeters),0),gridXCount-1)
    j = min(max(int((locationUsed[1] - newstartPoint[1]) /resolutionMeters),0),gridYCount-1)
    roadsNearest = osmRoads[['road','highway']].loc[roadPointsAll[(i*gridYCount)+j]]
    
    roadsNearest = roadsNearest[roadsNearest['highway'].isin(roadType)]['road']

    p1 = geometry.Point(locationUsed)
    if(len(roadsNearest) == 0):
        minimumDistance = 300
    else:
        #print(roadsNearest.apply(lambda row: p1.distance(row)).min())
        minimumDistance = (roadsNearest.apply(lambda row: p1.distance(row)).min())
    
    return np.log(minimumDistance + 1)


def compileVariablesTestTrain(monitorlocs,newstartPoint,roadgrid2,trafficAvgGrid,allgrids,weatherPmTraffic,boxSize,weatherPmVariables,timeResolution,testchunks,trainchunks,distFromRoad,distFromRest):

    dfsampleperiod = str(timeResolution) +"T"
    weatherPmTraffic = weatherPmTraffic.resample(dfsampleperiod).mean()
    resize = int(timeResolution / 60)
    allgrids = (allgrids[0:len(allgrids)-(len(allgrids) % resize)])
    allgridsResize = np.zeros((int(len(allgrids)/resize),allgrids.shape[1],allgrids.shape[2]))
    for i,c in enumerate(range(0,len(allgrids),resize)):
        allgridsResize[i] = np.sum(allgrids[c:c+resize-1],axis=0)

    weatherPmTest = weatherPmTraffic.loc[weatherPmTraffic.index.dayofyear.isin(testchunks)]
    trafficGridTest = allgridsResize[weatherPmTraffic.index.isin(weatherPmTest.index)]
    weatherPmTrain = weatherPmTraffic.loc[weatherPmTraffic.index.dayofyear.isin(trainchunks)]
    trafficGridTrain = allgridsResize[weatherPmTraffic.index.isin(weatherPmTrain.index)]

    xTrain = compileVariables(monitorlocs,newstartPoint,roadgrid2,trafficAvgGrid,trafficGridTrain,weatherPmTrain,boxSize,weatherPmVariables,timeResolution,distFromRoad,distFromRest)
    xTest = compileVariables(monitorlocs,newstartPoint,roadgrid2,trafficAvgGrid,trafficGridTest,weatherPmTest,boxSize,weatherPmVariables,timeResolution,distFromRoad,distFromRest)
    return xTest,xTrain



def linearRegressionSpatioTemporalModel(xTrain,yTrain,xTest,yTest,weatherPmTrain,weatherPmTest):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    daytimetrain = np.repeat(weatherPmTrain.index,4)
    daytimetest = np.repeat(weatherPmTest.index,4)

    splitbyhourXtrain = (np.array(pd.DataFrame(xTrain).groupby(daytimetrain.hour)))
    splitbyhourYtrain = np.array(pd.DataFrame(yTrain).groupby(daytimetrain.hour))

    splitbyhourXtest = (np.array(pd.DataFrame(xTest).groupby(daytimetest.hour)))
    splitbyhourYtest = np.array(pd.DataFrame(yTest).groupby(daytimetest.hour))

    a = 0
    b=0
    allReg = []
    for i in range(24):
        xVals = np.array(splitbyhourXtrain[i,1])
        yVals = np.array(splitbyhourYtrain[i,1])
        reg = LinearRegression().fit(xVals, yVals)
        #print("Accuracy ")
        #print(reg.score(xVals, yVals))
        xValsTest = np.array(splitbyhourXtest[i,1])
        yValsTest = np.array(splitbyhourYtest[i,1])


        #a+=reg.score(xValsTest, yValsTest)

        yPred = reg.predict(xValsTest)
        #print("Mean Squared Error ")
        #print(mean_absolute_percentage_error(yVals, yPred))
        a+= metrics.mean_absolute_error(yValsTest,yPred)
        b += mean_absolute_percentage_error(yValsTest, yPred)
        allReg.append(reg)
        

    print("MAPE: ",end="")
    print(b/24)
    print("MAE: ",end="")
    print(a/24)

    return(allReg)

def addTrafficToWeatherPm(weatherPmTraffic,monitorlocs,monitorlocsNames,resolutionMeters,newstartPoint,trafficgrid):
    weatherPmTrafficLive = weatherPmTraffic.copy()
    for i,stations in enumerate(monitorlocs):
        dynamicVariables = []
        for tgri in trafficgrid:
            (xAmount, yAmount,r1,r2) = (utm.from_latlon(stations[0],stations[1]))
            dueEast = int((xAmount-newstartPoint[0]) / resolutionMeters)
            dueNorth = int((yAmount - newstartPoint[1]) / resolutionMeters)
            dynamicVariables.append(tgri[dueNorth,dueEast])
        weatherPmTrafficLive[str(monitorlocsNames[i]) + " traffic"] = (dynamicVariables)
    return weatherPmTrafficLive
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def rollingWindow(a,n):
    testVar = np.arange(a.shape[0]-n+1)[:,None]
    size = np.arange(a.shape[0]-n+1)[:,None] + np.arange(n)
    return a[np.arange(a.shape[0]-n+1)[:,None] + np.arange(n)]

def splitintoWindow(xData,yData,lookback,seed) :
    xVar = np.hstack((xData,yData))
    xlen = xData.shape[1]
    print(xlen)
    split = 0.1
    testsize = int((split*len(xVar)))
    chunks = []
    lookback = 8
    for i in range(0,len(xVar),testsize):

        if((i+testsize) >= len(xVar)):
            ending = len(xVar) - 1
        else:
            ending = i+ testsize - 1
        if((ending - i) > lookback):
            chunks.append((i,ending))

    random.Random(seed).shuffle(chunks)
    testchunks = chunks[0:3]
    trainchunks = chunks[3:]
    print(trainchunks)
    testsize = sum([(b - a - lookback + 1) for a,b in testchunks])
    testdata = np.zeros((testsize,lookback,xVar.shape[1]))
    mark = 0
    for a,b in testchunks:
        (testdata[mark:mark+(b-a - lookback + 1),:,:]) = (rollingWindow(xVar[a:b,:],lookback))
        mark = mark+(b-a - lookback + 1)

    trainsize = sum([(b - a - lookback + 1) for a,b in trainchunks])
    traindata = np.zeros((trainsize,lookback,xVar.shape[1]))

    mark = 0
    for a,b in trainchunks:
        (traindata[mark:mark+(b-a - lookback + 1),:,:]) = (rollingWindow(xVar[a:b,:],lookback))
        mark = mark+(b-a - lookback + 1)
    return traindata[:,:,:xlen], testdata[:,:,:xlen], traindata[:,-1,xlen:], testdata[:,-1,xlen:]

def splitintoWindowTestTrain(xData,yData,trainchunks2,testchunks2,lookback,seed):

    trainc2 = [(a[0],a[-1]) for a in trainchunks2.reshape(-1,5)]
    testc2 = [(a[0],a[-1]) for a in testchunks2.reshape(-1,5)]

    trainchunks = []
    testchunks = []
    for a in trainc2:
        firstp = xData.index.get_loc(xData.loc[xData.index.dayofyear == a[0]].index[0])
        secondp = xData.index.get_loc(xData.loc[xData.index.dayofyear == a[-1]].index[-1])
        trainchunks.append((firstp,secondp))

    for a in testc2:
        firstp = xData.index.get_loc(xData.loc[xData.index.dayofyear == a[0]].index[0])
        secondp = xData.index.get_loc(xData.loc[xData.index.dayofyear == a[-1]].index[-1])
        testchunks.append((firstp,secondp))

    xVar = np.hstack((xData.to_numpy(),yData.to_numpy()))
    xlen = xData.shape[1]
    print(xlen)
    testsize = sum([(b - a - lookback + 1) for a,b in testchunks])
    print(testsize)
    testdata = np.zeros((testsize,lookback,xVar.shape[1]))

    mark = 0
    for a,b in testchunks:
        (testdata[mark:mark+(b-a - lookback + 1),:,:]) = (rollingWindow(xVar[a:b,:],lookback))
        mark = mark+(b-a - lookback + 1)

    trainsize = sum([(b - a - lookback + 1) for a,b in trainchunks])
    traindata = np.zeros((trainsize,lookback,xVar.shape[1]))

    mark = 0
    for a,b in trainchunks:
        (traindata[mark:mark+(b-a - lookback + 1),:,:]) = (rollingWindow(xVar[a:b,:],lookback))
        mark = mark+(b-a - lookback + 1)
    return traindata[:,:,:xlen], testdata[:,:,:xlen], traindata[:,-1,xlen:], testdata[:,-1,xlen:]


def testtraindata(pmValuesCalibrated,seed,chunklength):
 
    totaldaysofyear = pmValuesCalibrated.index.dayofyear.unique()
    daychunks = []
    for a in range((min(totaldaysofyear)),(max(totaldaysofyear)),chunklength):
        daychunks.append(np.array(range(a,a+chunklength)))
    random.Random(seed).shuffle(daychunks)
    testchunks = np.array(daychunks[0:(int(len(daychunks) / 3))]).flatten()
    trainchunks = np.array(daychunks[(int(len(daychunks) / 3)):]).flatten()
    return trainchunks, testchunks

def compileVariablesTestTrainWithWindow(monitorlocs,newstartPoint,roadgrid2,trafficAvgGrid,allgrids,weatherPmTraffic,boxSize,weatherPmVariables,timeResolution,testchunks,trainchunks,distFromRoad,distFromRest,liveTraffic=True,windows = 8,positioninWindow=8):
    dfsampleperiod = str(timeResolution) +"T"
    weatherPmTraffic = weatherPmTraffic.resample(dfsampleperiod).mean()
    resize = int(timeResolution / 60)
    allgrids = (allgrids[0:len(allgrids)-(len(allgrids) % resize)])
    allgridsResize = np.zeros((int(len(allgrids)/resize),allgrids.shape[1],allgrids.shape[2]))

    for i,c in enumerate(range(0,len(allgrids),resize)):
        allgridsResize[i] = np.sum(allgrids[c:c+resize],axis=0)


    trainc2 = [(a[0],a[-1]) for a in trainchunks.reshape(-1,5)]
    testc2 = [(a[0],a[-1]) for a in testchunks.reshape(-1,5)]
    trainchunks2 = []
    testchunks2 = []
    for a in trainc2:
        firstp = weatherPmTraffic.index.get_loc(weatherPmTraffic.loc[weatherPmTraffic.index.dayofyear == a[0]].index[0])
        secondp = weatherPmTraffic.index.get_loc(weatherPmTraffic.loc[weatherPmTraffic.index.dayofyear == a[-1]].index[-1])
        trainchunks2.append((firstp,secondp))


    for a in testc2:
        firstp = weatherPmTraffic.index.get_loc(weatherPmTraffic.loc[weatherPmTraffic.index.dayofyear == a[0]].index[0])
        secondp = weatherPmTraffic.index.get_loc(weatherPmTraffic.loc[weatherPmTraffic.index.dayofyear == a[-1]].index[-1])
        testchunks2.append((firstp,secondp))
    trainchunks2 = (np.array([list(range(a[0]+windows-1-(windows-positioninWindow),a[1]-(windows-positioninWindow))) for a in trainchunks2]).flatten())
    testchunks2 = (np.array([list(range(a[0]+windows-1-(windows-positioninWindow),a[1]-(windows-positioninWindow))) for a in testchunks2]).flatten())
    weatherPmTest = weatherPmTraffic.iloc[testchunks2]
    trafficGridTest = allgridsResize[weatherPmTraffic.index.isin(weatherPmTest.index)]
    weatherPmTrain = weatherPmTraffic.iloc[trainchunks2]
    trafficGridTrain = allgridsResize[weatherPmTraffic.index.isin(weatherPmTrain.index)]
    yTest = weatherPmTest.iloc[:,0:4].to_numpy().reshape(-1,1)
    yTrain = weatherPmTrain.iloc[:,0:4].to_numpy().reshape(-1,1)
    testIndex = weatherPmTest.index
    trainIndex = weatherPmTrain.index
    xTrain = compileVariables(monitorlocs,newstartPoint,roadgrid2,trafficAvgGrid,trafficGridTrain,weatherPmTrain,boxSize,weatherPmVariables,timeResolution,distFromRoad,distFromRest,liveTraffic)
    xTest = compileVariables(monitorlocs,newstartPoint,roadgrid2,trafficAvgGrid,trafficGridTest,weatherPmTest,boxSize,weatherPmVariables,timeResolution,distFromRoad,distFromRest,liveTraffic)

    return xTest,xTrain,yTest,yTrain,testIndex,trainIndex



def getLURWeights(pmvalues,dataSources):
    airPollutionData = pmvalues.to_numpy().reshape(-1,1)
    airPMean = pmvalues.to_numpy().mean(axis=1).repeat(4)
    airMeaned = ((airPollutionData.flatten() - airPMean) / airPMean).reshape(-1,1)
    reg = LinearRegression(fit_intercept=False).fit(dataSources,airMeaned)
    return reg,(reg.coef_).flatten()

def getLURWeightsDaphneStaticModel(dataSources):
    airPollutionData = dataSources['pm2_5']
    airPMean = dataSources['pm2_5_mean']
    airMeaned = ((airPollutionData - airPMean) / airPMean)
    reg = LinearRegression(fit_intercept=False).fit(
        dataSources[['minDistToRoad', 'minDistToRest', 'Jam Factor']].to_numpy(),airMeaned)
    return reg,(reg.coef_).flatten()

def getLURWeightsDaphneStaticModelModified(dataSources):
    airPollutionData = dataSources['pm2_5']
    airPMean = dataSources['pm2_5_mean']
    airMeaned = ((airPollutionData - airPMean) / airPMean)
    reg = LinearRegression(fit_intercept=False).fit(
        dataSources[['minDistToRoad', 'minDistToRest', 'Jam Factor']].to_numpy(),airPollutionData)
    return reg,(reg.coef_).flatten()

def getLURWeightsINHALE(weatherTrafficRoadData, Xtrain):
    airPollutionData = weatherTrafficRoadData.to_numpy()[:,1]
    airPMean = weatherTrafficRoadData.to_numpy()[:,-1]
    airMeaned = (airPollutionData - airPMean) / airPMean
    reg = LinearRegression(fit_intercept=False).fit(Xtrain,airMeaned)
    return reg,(reg.coef_).flatten()

def createSpatialGridNoLiveTraffic(lurWeights,baselineValue,distGrid,restGrid,trafficAvgGrid):
    gridValues = baselineValue + baselineValue *(((lurWeights[0] * distGrid) + (lurWeights[1] * restGrid) +(lurWeights[2] * trafficAvgGrid)))
    return gridValues
def createSpatialGridWithLiveTraffic(lurWeights,baselineValue,distGrid,restGrid,trafficAvgGrid,trafficgrid):
    gridValues = baselineValue + baselineValue *((lurWeights[0] * distGrid) + (lurWeights[1] * restGrid) +(lurWeights[2] * trafficAvgGrid) +lurWeights[3] * trafficgrid)
    return gridValues


def readPollutionDataInhale(datapath,resolutiontime):
    all_paths = list(Path(datapath).rglob('*.csv'))
    monitorlocs = []
    monitorlocsNames = []
    allPm = []
    inhaleDf = pd.DataFrame()
    for path in all_paths:
        df = pd.read_csv(path)
        df = df[~(df['gpsLatitude'] == 0)]
        df = df[~(df['gpsLongitude'] == 0)]
        print('Removed 0 GPS values::', df.shape)
        df = df[['timestamp','pm2_5','gpsLatitude','gpsLongitude']]
        #df['timestamp'] = pd.DatetimeIndex(df['timestamp']).tz_convert(None)
        #df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(project_mapping['inhale'][1])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
        dfsampleperiod = str(resolutiontime) + "T"
        df = df.resample(dfsampleperiod, on='timestamp').mean()
        df['timestamp'] = df.index
        fileName = str(path).split(os.path.sep)[-1]
        name = (str(fileName)[0:16] + '_' + str(fileName)[-10:-4])
        df['UUID'] = (str(fileName)[0:16])
        df['deploymentId'] = str(fileName)[-10:-4]
        monitorlocsNames.append(name)
        monitorlocs.append((df[df['gpsLatitude'] > 50]['gpsLatitude'].median(),df[df['gpsLongitude'] < -0.14]['gpsLongitude'].median()))
        inhaleDf = inhaleDf.append(df)
    print(monitorlocs)
    print(monitorlocsNames)

    return (inhaleDf, monitorlocs,monitorlocsNames)

def addTrafficToWeatherPmModified(weatherPmTraffic,monitorlocs,monitorlocsNames,resolutionMeters,newstartPoint,trafficgrid):
    ##Move to preprocessing
    print('before nan removal', weatherPmTraffic.shape)
    weatherPmTraffic = weatherPmTraffic.dropna()
    print('after nan removal', weatherPmTraffic.shape)
    z = np.abs(zscore(weatherPmTraffic[['gpsLatitude', 'gpsLongitude']]))
    threshold = 3
    weatherPmTraffic = weatherPmTraffic[(z < 3).all(axis=1)]
    print('after normalization', weatherPmTraffic.shape)
    indicesToRetain = []
    ctr = 0
    times = pd.date_range('2021-02-02', periods=3696, freq='60min')
    times = pd.to_datetime(times).tz_localize('UTC').tz_convert('Europe/London')
    for i in times:
        if(i in weatherPmTraffic['timestamp']):
            indicesToRetain.append(ctr)
        ctr += 1
    trafficgridModified = trafficgrid[indicesToRetain,:,:]
    weatherPmTrafficLive = weatherPmTraffic.copy()
    dynamicVariables = []
    for i, tgri in enumerate(trafficgridModified):
        (xAmount, yAmount,r1,r2) = (utm.from_latlon(weatherPmTraffic.iloc[i]['gpsLatitude'],weatherPmTraffic.iloc[i]['gpsLongitude']))
        dueEast = int((xAmount-newstartPoint[0]) / resolutionMeters)
        dueNorth = int((yAmount - newstartPoint[1]) / resolutionMeters)
        try:
            dynamicVariables.append(tgri[dueNorth,dueEast])
        except IndexError:
            indexLoc = monitorlocsNames.index(str(weatherPmTraffic.iloc[i]['UUID']+'_'+weatherPmTraffic.iloc[i]['deploymentId']))
            (xAmount, yAmount,r1,r2) = (utm.from_latlon(monitorlocs[indexLoc][0],monitorlocs[indexLoc][1]))
            dueEast = int((xAmount-newstartPoint[0]) / resolutionMeters)
            dueNorth = int((yAmount - newstartPoint[1]) / resolutionMeters)

            if dueNorth > tgri.shape[0] and dueEast > tgri.shape[1]:
                dynamicVariables.append(tgri[-1,-1])
            elif dueNorth > tgri.shape[0]:
                dynamicVariables.append(tgri[-1, dueEast])
            elif dueEast > tgri.shape[1]:
                dynamicVariables.append(tgri[dueNorth, -1])
            else:
                dynamicVariables.append(tgri[dueNorth, dueEast])

    weatherPmTrafficLive["liveTraffic"] = pd.Series(dynamicVariables) # insert NaNs for empty rows
    weatherPmTrafficLive = weatherPmTrafficLive.dropna()
    print(weatherPmTrafficLive)
    return weatherPmTrafficLive

def compileVariablesInhale(monitorlocs,monitorlocsNames,newstartPoint,roadgrid2,trafficAvgGrid,allgrids,weatherPm2,boxSize,timeResolution,distFromRoad,distFromRest,liveTraffic=False):
    staticVariables = []
    for stations in monitorlocs:
        (xAmount, yAmount,r1,r2) = (utm.from_latlon(stations[0],stations[1]))
        dueEast = int((xAmount-newstartPoint[0]) / boxSize)
        dueNorth = int((yAmount - newstartPoint[1]) / boxSize)
        staticVariables.append(trafficAvgGrid[dueNorth,dueEast])
    print(staticVariables)
    print(distFromRoad)
    print(distFromRest)
    df = weatherPm2.copy()
    df[['minDistRoad','minDistRest','trafficAvg']] = df.apply(lambda row: updateVariables(row, monitorlocs,monitorlocsNames,distFromRoad,distFromRest, staticVariables), axis = 1)
    #print(df)
    return df

def updateVariables(row, monitorlocs,monitorlocsNames,minimumDist,minimumDistRest,trafficAvg):
    monitorName = str(row['UUID'] + '_' + row['deploymentId'])
    indexLoc = monitorlocsNames.index(monitorName)
    return pd.Series([minimumDist[indexLoc], minimumDistRest[indexLoc], trafficAvg[indexLoc]])








