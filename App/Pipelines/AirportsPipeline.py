import pandas as pd 
from sqlalchemy import create_engine

strConnection = 'postgresql://weather_user:159753@159.65.233.116:5432/weather_tool'
engine = create_engine(strConnection, pool_pre_ping=True)

query_CitiesFind = 'SELECT iata_code FROM cities c WHERE NOT EXISTS (SELECT * FROM airports a2 WHERE a2.iata_code = c.iata_code)'
dfCities = pd.read_sql(query_CitiesFind, engine)
lstCities = dfCities['iata_code'].to_list()

if len(lstCities) > 0 :
    airports=pd.read_csv('/home/gleo/weather_tool/data')
    airports=airports[(airports["scheduled_service"]=='yes') & (airports["type"].str.contains("medium") | airports["type"].str.contains("large"))]
    airports.drop(["id", "ident", "iso_region","continent","type","scheduled_service","local_code","home_link","wikipedia_link","keywords"],axis=1,inplace=True)
    airports=airports[airports['iata_code'].isin(lstCities)].reset_index().drop(labels='index', axis=1)

    airports.to_sql('airports', engine, if_exists='append')
