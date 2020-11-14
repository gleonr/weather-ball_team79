import pandas as pd 
import numpy as np
import os, datetime
from sqlalchemy import create_engine

#Get data from IOWA State University - Mesonet, generating a link with the variables and airports info to download

def getdata(stationslist,df,year=1,adjUTC=-5,Last24h=False,link=True):
    
    #stationslist: Request of airports to download (Dictionary)
    #year:Amount of historical years to extract weather data
    #df: Airports data frame
    #adjUTC: Integer to adjust to Universal Time (UTC) to Local Time
    #Last24h: Download only last 24h
    
    url="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    
    #Add request of airports to download
    for i in stationslist:
        url=url+"station="+str(df.at[i,'gps_code'])+"&"
    
    #Add Weather variables to download
    #Temperauture (°C)//Wind Speed (Knots)//Visibility (miles)
    #Cloud Coverage//Cloud Height Level (ft)//Present Wx Codes
    
    url=url+"data=tmpc&data=sknt&data=vsby&data=skyl1&data=wxcodes"
    
    #Define the period of time to download
   
    if Last24h==True:
        #Dowload the last 24h
        
        date1=datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
        date2=datetime.datetime.utcnow().date() + datetime.timedelta(days=1)

        y1=str(date1.year)
        m1=str(date1.month)
        d1=str(date1.day)

        y2=str(date2.year)
        m2=str(date2.month)
        d2=str(date2.day)
    
        #Date from-to
        url=url+"&year1="+y1+"&month1="+m1+"&day1="+d1+"&year2="+y2+"&month2="+m2+"&day2="+d2
    
    else:
        #Download historical data according the amount of years
        date2=datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
        y2=str(date2.year)
        m2=str(date2.month)
        d2=str(date2.day)
        y1=str(date2.year-year)

        #Date from-to
        url=url+"&year1="+y1+"&month1=1&day1=1&year2="+y2+"&month2="+m2+"&day2="+d2
    
    #Missing values as null, download directly as txt file    
    url=url+"&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=null&trace=null&direct=yes&report_type=1&report_type=2"
    
    #Link to download data
    if link==True:
        print("Link:\n\n"+url)
    
    data=pd.read_csv(url)
    
    #Define time variables, adjust variables to local time (Colombia UTC-5)
    data["valid"]=pd.to_datetime(data["valid"])
    data["Year"]= pd.DatetimeIndex(data["valid"]).year
    data["Month"]= pd.DatetimeIndex(data["valid"]).month
    data["Day"]= pd.DatetimeIndex(data["valid"]+datetime.timedelta(hours=adjUTC)).day
    data["Hour"]= pd.DatetimeIndex(data["valid"]+datetime.timedelta(hours=adjUTC)).hour
    data["valid"]=pd.DatetimeIndex(data["valid"]+datetime.timedelta(hours=adjUTC))
    
    #Generate a Boolean 100 if a meteorological phenomenon occurs 0 Otherwise
    data["Rain"]=data['wxcodes'].str.contains('RA').fillna(False).astype(int)*100
    data["Fog-Brume"]=(data['wxcodes'].str.contains('FG')|data['wxcodes'].str.contains('BR')|(data['wxcodes'].str.contains('HZ'))).fillna(False).astype(int)*100
    data["Thunder"]=data['wxcodes'].str.contains('TS').fillna(False).astype(int)*100
    
    #Generate a Boolean 100 if a Fog-Brume or Thunder occur 0 Otherwise
    #Fog-Brume and Thunders restrict the operation
    data["Op constraint"]=((data["Fog-Brume"]==100) | (data["Thunder"]==100)).astype(int)*100
    
    #Organize columns of data
    data.rename(columns={"station":"OACI","tmpc": "Temperature","vsby":"H Vsby","skyl1":"V Vsby","sknt": "Wind"},inplace=True)   
    df2=df['gps_code'].to_frame().reset_index()
    data=pd.merge(left=data, right=df2, how='left', left_on='OACI', right_on='gps_code')
    data.rename(columns={"iata_code":"IATA","valid":"Date"},inplace=True)    
    data=data[['IATA','OACI',"Date",'Year','Month','Day', 'Hour', 'H Vsby', 'V Vsby','Temperature', 
               'Wind', 'Rain', 'Fog-Brume', 'Thunder','Op constraint']]
    data.sort_values(by=['IATA','Year','Month','Day', 'Hour'],inplace=True)
    data.reset_index(drop=True,inplace=True)
    
    #Data Imputation / If a null value exists replace with historical mean value of variable in the airport
    data['H Vsby'] = data[['IATA','H Vsby']].groupby("IATA").transform(lambda x: x.fillna(x.mean()))
    data['V Vsby'] = data[['IATA','V Vsby']].groupby("IATA").transform(lambda x: x.fillna(x.mean()))
    data['Temperature'] = data[['IATA','Temperature']].groupby("IATA").transform(lambda x: x.fillna(x.mean()))
    data['Wind'] = data[['IATA','Wind']].groupby("IATA").transform(lambda x: x.fillna(x.mean()))
    
    #Drop outliers according to references values
    data=data[(data['H Vsby']<=6.21) & (data['Wind']<=30)&(data['Temperature']>=-5)&(data['Temperature']<=50)&(data['V Vsby']<5000)]
    
    #Check missing data
    missing_data = data.isnull()

    if len(data.columns[data.isnull().any()].tolist())==0:
        print("\nData without null values")
    else:
        print("\nData with null values")
        for column in data.columns[data.isnull().any()].tolist():
            print (missing_data[column].value_counts())
    
    return data


strConnection = 'postgresql://weather_user:159753@159.65.233.116:5432/weather_tool'
engine = create_engine(strConnection, pool_pre_ping=True)
engine.execute('DELETE FROM tempmesonet;')

query_CitiesFind = 'SELECT * FROM airports'
dfCities = pd.read_sql(query_CitiesFind, engine)
lstCities = dfCities['iata_code'].to_list()
dfCities.set_index('iata_code', inplace=True)
Years=5

data=getdata(lstCities,dfCities,year=Years,adjUTC=-5,Last24h=True,link=True)

if len(data) > 0 :
    data['Rain'] = data.Rain.astype(bool)
    data['Fog-Brume'] = data['Fog-Brume'].astype(bool)
    data['Thunder'] = data['Thunder'].astype(bool)
    data['Op constraint'] = data['Op constraint'].astype(bool)
    print(data.shape)

    data.to_sql('tempmesonet', engine, if_exists='append', chunksize=1000) 