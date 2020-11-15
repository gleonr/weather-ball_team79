
# OS libraries
import pathlib, datetime as dt
from configparser import ConfigParser

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# Data manipulation libraries
import pandas as pd
import numpy as np
import json
import re
from itertools import chain

# Drawing Libraries
import plotly.express as px
import plotly.figure_factory as ff

# Access
config = ConfigParser()
config.read('config.ini')
mapbox_token = config['mapbox']['access_token']
px.set_mapbox_access_token(mapbox_token)

# Dash Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import visdcc
from dash.dependencies import Input, Output, State, ClientsideFunction

# import the lib for querying SQL data
from sqlalchemy import create_engine
strConnection = 'postgresql://weather_user:159753@159.65.233.116:5432/weather_tool'
engine = create_engine(strConnection, pool_pre_ping=True)

# import libs for running the model
import os, logging
import tensorflow as tf
import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

######################################################
#   Functions for loading and preparing data         #
######################################################

# Load Country airports
def get_airports_info():  
    """"get the airports info from the database"""
    sqlQuery = "SELECT * FROM airports;"
    airports = pd.read_sql(sqlQuery, engine)
    return airports


# Get and prepare data from IOWA State University - Mesonet

def get_weather_data():
    sqlQuery = "SELECT * FROM mesonet;"
    data = pd.read_sql(sqlQuery, engine)   
    return data

# Set current airport
def set_current_airport(weather_data, airport='BOG'):
    # Format months and hours
    months = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    hours =  {n : f"{n % 12 or 12} {['AM', 'PM'][n > 11]}" for n in range(24)}
    time_frame_cats = ['Overnight', 'Morning', 'Afternoon', 'Evening']
    
    airport_weather_data = weather_data[weather_data['iata'] == airport].copy()

    airport_weather_data['month'] = pd.Categorical(airport_weather_data['month']).rename_categories(months)
    airport_weather_data['hour'] = pd.Categorical(airport_weather_data['hour']).rename_categories(hours)

    # Create a new field to evaluate hour ranges
    airport_weather_data['Time Frame'] =  airport_weather_data['hour'].apply(lambda x:  time_frame(x))
    airport_weather_data['Time Frame'] = pd.Categorical(airport_weather_data['Time Frame'], categories=time_frame_cats, ordered=True)

    return airport_weather_data

# Dividing the data into categories for a better explanatory user analysis
def time_frame(Hour):
    # Divide the day into segments for a better approach
    # Based on classification in http://www.angelfire.com/pa/pawx/time.html
    hours =  {f"{n % 12 or 12} {['AM', 'PM'][n > 11]}" : n  for n in range(24)}
    Hour = hours[Hour]

    day_t=''
    if (Hour >= 0) & (Hour < 6): 
        day_t='Overnight'
    elif (Hour >= 6) & (Hour < 12):
        day_t='Morning'
    elif (Hour >= 12) & (Hour < 18):
        day_t='Afternoon'
    else: day_t= 'Evening'
    return day_t


#############################################
#   Functions for computing figures         #
#############################################

def adjust_period(start, end, airport_weather_data):
    # Defines the period for analysis of data default current year - five years
    # The number 5 is used due to constraints in the range slider component that doesn not 
    # allows to use the year directly as a parameter
    
    current_year = dt.datetime.utcnow().year
    start = current_year - (5 - start)
    end = current_year - (5 - end)
    airport_weather_data_period = airport_weather_data[(airport_weather_data['year'] >= start) & \
                                     (airport_weather_data['year'] <= end)].copy()
    return airport_weather_data_period

def compute_meteorological_heatmap(airport_weather_data, variable = 'temperature'):
    # Variables for met heatmaps 
    # Horizontal Visibility (miles)
    # Vertical Visibility (ft)
    # wind (knots)
    # temperature (°C)

    heatmap_met_data = pd.pivot_table(airport_weather_data[['iata', 'month', 'hour', variable]],\
               index='month', columns='hour' , values=variable, aggfunc=np.mean).round(2)
    return heatmap_met_data

def compute_metereological_boxplot(airport_weather_data, variable = 'temperature', descriptor = 'month',\
    descriptor_value = 'all', period = [dt.datetime.utcnow().year - 5, dt.datetime.utcnow().year], categories='Time Frame'):
    # Compute the data for a box | violin plot basen on:
    # airport_weather_data : contains information about the current airport
    # variable : the meteorological variable under analysis - variables in Met_vars
    # descriptor : time division for analysis months or Hours
    # descriptor value: the period for evaluation 'all' for all (months or Hours) 0-23 for hours , 1-12 for months
    # period: the time frame used for analysis default is five years until now (a list with start and end periods)
    # categories: classification of data default 'Time Frame'  

    vars_in_plot = [categories, variable , descriptor]

    time_frame = ((airport_weather_data['year'] >= period[0]) & (airport_weather_data['year'] <= period[1]))
    
    if (descriptor_value != 'all'):
        descriptor_length =  (airport_weather_data[descriptor] == descriptor_value)
        box_met_data = airport_weather_data[time_frame & descriptor_length][vars_in_plot].copy()
    else:
        box_met_data = airport_weather_data[time_frame][vars_in_plot].copy()
    box_met_data.sort_values(by=[descriptor, categories], inplace=True)
    return box_met_data

def compute_probabilistic_heatmap(airport_weather_data, variable = 'fog_brume'):
    # Variables for prob heatmaps 
    # rain
    # thunder
    # Fog Brume
    
    numerator = pd.pivot_table(airport_weather_data[(airport_weather_data[variable] == 1)], index='month', columns='hour', values='iata', fill_value=0, aggfunc=['count'])
    denominator = pd.pivot_table(airport_weather_data, index='month', columns='hour', values='iata', aggfunc=['count'])
    heatmap_prob_data = (100 * numerator/denominator).round(2)
    heatmap_prob_data.columns = heatmap_prob_data.columns.droplevel(0)


    return heatmap_prob_data



##############################################################################
#   Functions for computing model prediction and variable inspection         #
##############################################################################

#################################################
#                 Model charging                #
#################################################

###################################Generates the class WindowGenerator########################################

def model_predict(iata='BOG', engine= engine):
    """Function to charge the model for the airport city and return the prediction"""

    query = "select h_vsby, v_vsby, temperature, wind, rain, fog_brume, thunder \
            from mesonet m where iata = '" + iata + "' and m.\"Date\" between now() - interval '48 HOURS' and now()  order by m.\"Date\" "
    df = pd.read_sql(query, engine)

    class WindowGenerator():
    #The first function within the class is to instantiate and define the variables entered as input
        def __init__(self, input_width, label_width, shift,
                df=df, label_columns=None):
            # Stores the raw data from the dataset
            self.df = df


        # Work out the label and the column indices, depending on the input parameters
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {name: i for i, name in
                                                enumerate(label_columns)}
                self.column_indices = {name: i for i, name in
                                    enumerate(df.columns)}

            # Work out the window parameters.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        #This functions returns the Total Window Size, the Input and Label Indices and the Label column name
        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])

    ###############################Defines a function to split the data sets #####################################
    def split_window(self, features):
        #The input variables will be those that will not be predicted
        inputs = features[:, self.input_slice, :]
        #The label variables will be those that will be predicted
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
             #In case there is more than one label variable, this for stacks all of them
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        #Defines the shapes for the input and the label variables
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        #Returns the input and the labels  
        return inputs, labels

    WindowGenerator.split_window = split_window

    mean_df = pd.read_csv('./forecast/means.csv')
    std_df = pd.read_csv('./forecast/std.csv')

    hist_means = mean_df.iloc[0]
    hist_std = std_df.iloc[0]

    df_normalized=(df-hist_means)/hist_std

    result_df = pd.DataFrame()
    varlst = ['h_vsby','v_vsby']

    for Variable in varlst:
        wide_window= WindowGenerator(
            input_width=24, label_width=24, shift=1,
            label_columns=[Variable])
            
        predict_window = tf.stack([np.array(df_normalized[len(df_normalized)-wide_window.total_window_size:])])

        predict_inputs, predict_labels = wide_window.split_window(predict_window)

        modelPath = './data/Models/{}/{}.h5'.format(iata,Variable)
        model = keras.models.load_model(modelPath)

        predicted = model.predict(predict_inputs)

        predicted_denorm_H=predicted*hist_std[Variable]+hist_means[Variable]
        predicted_denorm_H=predicted_denorm_H[0]
        #As the previous data is standarized, here we calcule the real value for each prediction
        prediction_labels_denorm_H=predict_labels*hist_std[Variable]+hist_means[Variable]
        prediction_labels_denorm_H=prediction_labels_denorm_H[0]

        predlst = np.reshape((predicted*hist_std[Variable]+hist_means[Variable])[0],(1,24) ).tolist()[0]
        labellst = np.reshape((predict_labels*hist_std[Variable]+hist_means[Variable])[0],(1,24) ).tolist()[0]

        if Variable == 'h_vsby':
            tittle = 'H_Vsby'
        
        if Variable == 'v_vsby':
            tittle = 'V_Vsby'

        result_df['{}_predict'.format(tittle)] = pd.Series(predlst)
        result_df['{}_labels'.format(tittle)] = pd.Series(labellst)

    return result_df
    


#################################################
#               Variable inspection             #
#################################################

def operation_conditions(iata='BOG'):
    # Return the minimal operation conditions for selectd airport
    # op_conditions[0] = h_vsby (miles)
    # op_conditions[1] = v_vsby (ft)

    op_cond_df = pd.read_csv("./assets/req_min_airports.csv", sep=';')
    op_conditions = list(op_cond_df[op_cond_df['IATA'] == iata].values[0][1:3])
    return op_conditions



def frequency_table(current_airport, op_conditions):  
    # This function explore the presence of non-operating conditions
    # for the last week, month and year 
    #                   |  YTD  |  MTD  |   Wk  |
    # Total acc         |       |       |       |
    # Current hour      |       |       |       |
    # Next hour         |       |       |       |
    #
    # op_conditions represent a list with H_vis and V_vis that limit airport operations
    # item 0 refers to H_vis and item 1 to V_vis
    # current airport contains the weather info for the airport currently selected

    hours =  {n : f"{n % 12 or 12} {['AM', 'PM'][n > 11]}" for n in range(24)}

    date = dt.datetime.now() #current date
    year = date.year
    month = date.month # current month
    week = pd.to_datetime(date) - dt.timedelta(days=7) # One week behind from now
    cur_hour = hours[date.hour]
    if date.hour == 23:
        next_hour = hours[0]
    else:
        next_hour = hours[date.hour + 1]

    airport_copy = current_airport[current_airport['year'] == year].copy()

    # New categories for setting non-operative conditions for airports
    airport_copy['non_op_conditions'] = airport_copy.apply(lambda x: 1 if (x['h_vsby'] < op_conditions[0]) or (x['v_vsby'] < op_conditions[1]) else 0, axis=1)

    # For the total number of events last running year
    YTD = airport_copy[airport_copy['non_op_conditions'] == 1]['non_op_conditions'].sum()
    MTD = airport_copy[(airport_copy['month'] == month) & (airport_copy['non_op_conditions'] == 1)]['non_op_conditions'].sum()
    Wk = airport_copy[(airport_copy['month'] == month) & ((airport_copy['Date'] >= week)) & (airport_copy['non_op_conditions'] == 1)]['non_op_conditions'].sum()


    # For the total number of events at current hour
    airport_tmp = airport_copy[airport_copy['hour'] == cur_hour].copy()
    YTD_cur = airport_tmp[airport_tmp['non_op_conditions'] == 1]['non_op_conditions'].sum()
    MTD_cur = airport_tmp[(airport_tmp['month'] == month) & (airport_tmp['non_op_conditions'] == 1)]['non_op_conditions'].sum()
    Wk_cur = airport_tmp[(airport_tmp['month'] == month) & ((airport_tmp['Date'] >= week)) & (airport_tmp['non_op_conditions'] == 1)]['non_op_conditions'].sum()

    # For the total number of events at next hour
    airport_tmp = airport_copy[airport_copy['hour'] == next_hour].copy()
    YTD_next = airport_tmp[airport_tmp['non_op_conditions'] == 1]['non_op_conditions'].sum()
    MTD_next = airport_tmp[(airport_tmp['month'] == month) & (airport_tmp['non_op_conditions'] == 1)]['non_op_conditions'].sum()
    Wk_next = airport_tmp[(airport_tmp['month'] == month) & ((airport_tmp['Date'] >= pd.to_datetime(week))) & (airport_tmp['non_op_conditions'] == 1)]['non_op_conditions'].sum()

    # Data frame for non operating frequencies
    non_op_freq = pd.DataFrame({'YTD': [YTD, YTD_cur, YTD_next], 'MTD': [MTD, MTD_cur, MTD_next], 'Week':[ Wk, Wk_cur, Wk_next]}, index=['Total @', "@ " + cur_hour, "@ " + next_hour])

    return non_op_freq

''' End Functions for computing figures''' 


#############################################
#   Functions for creating figures          #
#############################################

def create_airports_map(airports, iata_code = 'BOG'):
    airports_map = px.scatter_mapbox(airports, lat='latitude_deg', lon='longitude_deg',  \
                        size=np.log(airports['elevation_ft']), color='elevation_ft', color_continuous_scale=px.colors.sequential.Cividis, zoom=12, \
                        mapbox_style='open-street-map',
                        hover_name="name",
                        hover_data=["municipality", "latitude_deg", "longitude_deg", "elevation_ft", "iata_code"],
                        labels={"municipality" : "Capital City ", 
                                "latitude_deg" : "Latitude ",
                                "longitude_deg" : "Longitude ",
                                "elevation_ft": "Elevation (ft) ",
                                "iata_code" : "iata Code "
                                },
                        center={'lat': airports[airports['iata_code'] == iata_code]['latitude_deg'].array[0],
                                'lon': airports[airports['iata_code'] == iata_code]['longitude_deg'].array[0]},
                        
                        )

    airports_map['layout']['margin'] = {'l':0, 'r':0, 't':0, 'b':0}
    return airports_map

def create_meteorological_heatmap(heatmap_met_data, variable='temperature', units='°C'):
    labels = dict(y="month", x="Time of Day", color=(units))
    heatmap_met = px.imshow(heatmap_met_data, labels=labels, color_continuous_scale=px.colors.cyclical.IceFire)
    heatmap_met.layout.update(transition_duration=500, autosize=True, margin={'t':0})
        
    return heatmap_met

def create_metereological_boxplot(box_met_data, variable = 'temperature', descriptor='month', category='Time Frame'):
    box_met = px.violin(data_frame=box_met_data, y=variable, x=descriptor, color=category)
    box_met.update_layout(legend=dict(orientation="h", yanchor="top", y=1.12,
                                      xanchor="left", x=0.01))
    box_met['data'][1]['visible'] ='legendonly'
    box_met['data'][2]['visible'] ='legendonly'
    box_met['data'][3]['visible'] ='legendonly'
    
    return box_met

def create_probabilistic_heatmap(heatmap_prob_data, variable='fog_brume'):
    labels = dict(y="month", x="Time of Day", color="%")
    heatmap_prob = px.imshow(heatmap_prob_data, labels=labels, color_continuous_scale='gray')
    heatmap_prob.layout.update(transition_duration=500, autosize=True, legend=dict(orientation="h"), margin={'t':0})
    return heatmap_prob


def create_frequency_map(non_op_freq):
    z = non_op_freq.values
    y = list(non_op_freq.index)
    x = list(non_op_freq.columns)

    colorscale=[[0.0, 'rgb(230, 230, 230)'], [1.0, 'rgb(57, 149, 172)']]

    frequency_map = ff.create_annotated_heatmap(z,y=y, x=x, colorscale=colorscale)
    frequency_map.layout.update(autosize=True, margin= {'l':80, 'r':20, 't':0, 'b':20}, hovermode=False)
    return frequency_map


def create_predict_plots(model_results, op_conditions):
    # Setting the start time (24 hours earlier) based on current time 
    # and the time for the forecast
    # op_conditions[0] = h_vsby (miles)
    # op_conditions[1] = v_vsby (ft)

    time = dt.datetime.utcnow() - dt.timedelta(hours=5)
    time_start = time - dt.timedelta(hours=23)
    time_next = (time + dt.timedelta(hours=1)).strftime('%d@%H')

    # Plot for Horizontal visibility
 
    H_vis_min = op_conditions[0]    
    H_vis_plot = px.line(model_results, y=['H_Vsby_labels', 'H_Vsby_predict'], x=list(pd.date_range(start=time_start,end=time, freq='h').strftime('%d@%H')), \
              category_orders={"x": [pd.date_range(start=time_start,end=time, freq='h').strftime('%d@%H')]})
    
    '''
    # Adding the threshold line
    H_vis_plot.add_hline(y=H_vis_min, line_dash="dashdot", annotation_text="Visibility Threshold: " + str(np.round(H_vis_min,2)), annotation_position="bottom right", line_color='red')
    '''
    H_vis_threshold = px.line(x=[time_start.strftime('%d@%H'), time.strftime('%d@%H')], y=[H_vis_min, H_vis_min],color_discrete_sequence=["red"] ,line_dash_sequence=['dashdot'])
    H_vis_threshold.update_traces (name='Visibility Threshold:' + str(H_vis_min), showlegend=True, )

    # Adding forecast point
    H_vis_forecast = px.scatter(x=[time_next], y=[model_results['H_Vsby_predict'].iat[-1]], text=[np.round([model_results['H_Vsby_predict'].iat[-1]],2)])
    H_vis_forecast.update_traces(name='Forecast', showlegend=True, marker_symbol='square-dot',marker_color='green', textposition='top center')
    

    H_vis_plot.add_trace(H_vis_forecast.data[0])
    H_vis_plot.add_trace(H_vis_threshold.data[0])

    # Formatting the plot
    H_vis_plot.update_layout(
    showlegend=True,
    plot_bgcolor="white",
    )
    H_vis_plot.update_xaxes(title="Daytime (hours)", type='category')
    H_vis_plot.update_yaxes(title="Visibility (miles)")

    # Plot for Vertical visibility
 
    V_vis_min = op_conditions[1]
    V_vis_plot = px.line(model_results, y=['V_Vsby_labels', 'V_Vsby_predict'], x=list(pd.date_range(start=time_start,end=time, freq='h').strftime('%d@%H')), \
              category_orders={"x": [pd.date_range(start=time_start,end=time, freq='h').strftime('%d@%H')]})

    # Adding forecast point
    V_vis_forecast = px.scatter(x=[time_next], y=[model_results['V_Vsby_predict'].iat[-1]], text=[np.round([model_results['V_Vsby_predict'].iat[-1]],2)])
    V_vis_forecast.update_traces(name='Forecast', showlegend=True, marker_symbol='square-dot',marker_color='green', textposition='top center')
    '''
    # Adding the threshold line
    V_vis_plot.add_hline(y=V_vis_min, line_dash="dashdot", annotation_text="Visibility Threshold: " + str(np.round(V_vis_min,2)), annotation_position="bottom right", line_color='red')
    '''
    V_vis_threshold = px.line(x=[time_start.strftime('%d@%H'), time.strftime('%d@%H')], y=[V_vis_min, V_vis_min],color_discrete_sequence=["red"] ,line_dash_sequence=['dashdot'])
    V_vis_threshold.update_traces (name='Visibility Threshold:' + str(V_vis_min), showlegend=True, )

    V_vis_plot.add_trace(V_vis_threshold.data[0])
    V_vis_plot.add_trace(V_vis_forecast.data[0])

    # Formatting the plot
    V_vis_plot.update_layout(
    showlegend=True,
    plot_bgcolor="white",
    )
    V_vis_plot.update_xaxes(title="Daytime (hours)", type='category')
    V_vis_plot.update_yaxes(title="Visibility (ft)")

    return H_vis_plot, V_vis_plot

''' End Functions for creating figures'''


################################################
#              Initial parameters              #
################################################

# Meteorological variables
Met_vars = ['temperature', 'wind', 'h_vsby', 'v_vsby']
Met_vars_units = ['°C', 'Knots', 'Miles', 'ft']

# Probabilistic variables
Prob_vars = ['fog_brume', 'rain', 'thunder']

# Load airports data for default country 'CO'
airports = get_airports_info()
airports_map = create_airports_map(airports)  # create the map

# Load data set from Mesonet for default country
weather_data = get_weather_data() 

# Filter data for default airport 'BOG' in default country 
airport_weather_data = set_current_airport(weather_data)

# Compute data for default meteorological variable 'temperature'
heatmap_met_data = compute_meteorological_heatmap(airport_weather_data)
heatmap_met = create_meteorological_heatmap(heatmap_met_data) # create the map

# Compute data for default meteorological box plot 'month'
box_met_data = compute_metereological_boxplot(airport_weather_data)
box_met = create_metereological_boxplot(box_met_data)

# Compute data for default probabilistic variable 'fog_brume'
heatmap_prob_data = compute_probabilistic_heatmap(airport_weather_data)
heatmap_prob = create_probabilistic_heatmap(heatmap_prob_data)

# Compute model and prediction data for default airport data set 
op_conditions = operation_conditions() # Operation conditions for BOG
non_op_freq = frequency_table(airport_weather_data, op_conditions)
frequency_map = create_frequency_map(non_op_freq)

model_results = model_predict(engine= engine)
H_vis_plot, V_vis_plot = create_predict_plots(model_results,op_conditions)


########################################################################################################
#                                             App layout                                               #
########################################################################################################

##############################################################################
#                             HTML Components                                #
##############################################################################

page_title = "Weather-ball WIS"

head = """
        <head>
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{page_title}</title>
            <link rel="stylesheet" href="https://meyerweb.com/eric/tools/css/reset/reset.css">
            <link rel="stylesheet" href="/assets/aeris-wxblox.css">
            <link rel="stylesheet" href="/assets/bootstrap.min.css">
            <link rel="stylesheet" href="/assets/weather-icons.css">
            <link rel="stylesheet" href="/assets/weather-icons-wind.css">
            <link rel="stylesheet" href="/assets/fixed.css">
            <link rel="stylesheet" href="/assets/style.css">
            <link rel="stylesheet" href="/assets/fontawesome.min.css">
            <link rel="stylesheet" href="/assets/brands.min.css">
            <link rel="shortcut icon" href="/assets/favicon.ico">
            <script src="https://cdn.aerisapi.com/wxblox/latest/aeris-wxblox.js"></script>



        </head> """.format(page_title = page_title)


#################################
#              Menu             #
#################################

navigation = '''
        <!--- Navigation -->
        <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top">
            <a class="navbar-brand" href="index.html"><img src="assets/wb_logo.png"></a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarResponsive">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#home">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#explore">Explore</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#analyze">Analyze</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#inspect">Inspect</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#about">About</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#team">Our Team</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#sponsors">Sponsors</a>
                    </li>			
                </ul>
            </div>
        </nav>
        <!--- End navigation -->
'''


#################################
#             Carousel          #
#################################
image_slider = '''
        <!--- Start image slider -->
        <div id="carouselExampleCaptions" class="carousel slide carousel-fade" data-ride="carousel" data-interval="8000">
            <!-- Start Carousel inner -->
            <div class="carousel-inner" role="listbox">
            <!-- Slide 1 -->
                <div class="carousel-item active" style="background-image: url(assets/airport.jpg);">
                    <div class="carousel-caption d-none d-md-block">
                    <h1>Weather-ball WIS</h1>
                    <h4>Airport weather operational conditions explorer and adviser</h4>
                    <a class="btn btn-dark btn-outline-light btn-lg" href="#explore">Get started</a>
                    </div>
                </div>
            <!-- Slide 2 -->
            <div class="carousel-item" style="background-image: url(assets/airport_2.jpg);">
                <div class="carousel-caption d-none d-md-block">
                    <h1>Weather-ball WIS</h1>
                    <h4>Airport weather operational conditions explorer and adviser</h4>
                    <a class="btn btn-dark btn-outline-light btn-lg" href="#explore">Get started</a>
                </div>
            </div>
            <!-- Slide 3 -->
            <div class="carousel-item" style="background-image: url(assets/airport_3.jpg);">
                <div class="carousel-caption d-none d-md-block">
                    <h1>Weather-ball WIS</h1>
                    <h4>Airport weather operational conditions explorer and adviser</h4>
                    <a class="btn btn-dark btn-outline-light btn-lg" href="#explore">Get started</a>
                </div>
            </div>
            </div> 
            <!-- End Carousel inner-->
        </div>
        <!--- End image slider -->
'''

#################################################################################################
#                                                Home                                           #
#################################################################################################

home = '''<div id="home">{navigation}
                         {image_slider}</div>'''.format(navigation=navigation, 
                                                        image_slider=image_slider)


#################################################################################################
#                                       Integration to dash                                     #
#################################################################################################

class CustomDash(dash.Dash):
    def interpolate_index(self, **kwargs):
        return '''<!DOCTYPE html>
        <html lang="en">
        {head}
        <body>
            {home}
            <!--// target DOM element where WeatherBlox will be rendered //-->
            {app_entry}
                        
            <!--- Script Source Files -->
            <script src="/assets/jquery-3.5.1.min.js"></script>
            <script src="/assets/bootstrap.min.js"></script>
            <script defer src="/assets/brands.min.js"></script>
            <script defer src="/assets/fontawesome.min.js"></script>
            
            
            <!--- End of Script Source Files -->     
            {config}
            {scripts}
            {renderer}   
        
        </body>
        </html>
        '''.format(
            head=head,
            home=home,
            app_entry=kwargs['app_entry'],
            config=kwargs['config'],
            scripts=kwargs['scripts'],
            renderer=kwargs['renderer'])
        


app = CustomDash(__name__, update_title=None)
server = app.server

#################################################################################################
#                                   Section 1 - Explore                                         #
#################################################################################################

#############################################
#                Section heading            #
#############################################
explore_heading = html.Div( 
        [
            html.Div(
                [
                    html.H3(children="Explore airports around Colombia", className="heading"),
                    html.Hr(className="my-4")
                ],
                className="col-12 text-center flex-start"
            ),
        ], 
        className="jumbotron container-fluid align-content-center"
        )


#############################################
#                airport data               #
#############################################

airport_data = html.Div( # Card for Airport data
        [
            html.Label(id='Altitude', children=f"Altitude (ft. asl.): {airports[airports['iata_code'] == 'BOG']['elevation_ft'].array[0]}"),
            html.Label(id='Location', children=f"Location: Lat: {airports[airports['iata_code'] == 'BOG']['latitude_deg'].array[0]},\
                                                 Lon: {airports[airports['iata_code'] == 'BOG']['longitude_deg'].array[0]}")
        ],
        className="card select-label"
        )


#############################################
#           airports dropdown list          #
#############################################

# Here is the control for the current airport

airports_dropdown = html.Div(
        [
            html.P(children="Click a location on the map or select it from the list",  className="block-title block-title-bordered"),
            dcc.Dropdown(id='Airports_list', placeholder='Select variable to see', 
                                options=[{'label':f"{airports[airports['name'] == name]['iata_code'].array[0]} - {name}",'value':airports[airports['name'] == name]['iata_code'].array[0]} for  name \
                                     in airports['name'].unique()],
                                value='BOG',
                                multi=False),
        ],
        className="card align-content-center text-center select-label"
)
                                

#############################################
#                section layout             #
#############################################

explore = html.Div(
    [
        explore_heading,
        html.Div( # Start of row
            [
                html.Div( # Left column for map
                    [
                        html.Div(id="Airport_pic", \
                            style={'background-image': 'url(/assets/BOG.jpg)'}),
                        html.H5(id='City', children=airports[airports['iata_code'] == 'BOG']['municipality'].array[0], className='back-title'),
                        airports_dropdown,
                        airport_data,
                    ],
                    className="col-sm-5"
                ),
                html.Div( # Right column
                    [
                        # Map
                        dcc.Graph(id='Airports_map', figure=airports_map)
                    ],
                    className="col-sm-7 d-flex align-items-center shaded-col"
                    )
            ], 
            className="row no-padding text-center"  # end of row
            )
    ],
    id="explore",
    className="offset"
)

#################################################################################################
#                                   Section 2 - analyze                                         #
#################################################################################################

#############################################
#                Section heading            #
#############################################

analyze_heading = html.Div(
            [
                html.Div(
                    [
                        html.H3(children="Analyze the evolution of climatological variables over time", className="heading"),
                        html.Hr(className="my-4"),
                    ],
                    className="col-sm-12 text-center"
                 ),
            ],
            className="jumbotron container-fluid align-content-center"
        )

#############################################
#                 Time slider               #
#############################################

time_slider = html.Div(
    [
        html.P(children="Select a period to explore", className="text-label"),
        dcc.RangeSlider(id='RS_period',  step=None,
                        min=0,
                        max=(airport_weather_data['year'].max() - airport_weather_data['year'].min()),
                        value=[0, (airport_weather_data['year'].max() - airport_weather_data['year'].min())],
                        marks={idx: {'label':f'{year}'} for idx, year in enumerate(airport_weather_data['year'].unique())}
        )
    ],
    className="card select-label"
)

#############################################
#               heatmaps layout             #
#############################################

heatmap_met_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.P(children="Distribution map for: ", className="text-label"),
                    ],
                    className="col-sm-6 back-title"
                ),
                html.Div(
                    [
                        dcc.Dropdown(id='Heatmap_met_list', placeholder='Select variable to see', 
                            options=[{'label': var, 'value': idx} for idx, var in enumerate(Met_vars)],
                            value=0,
                            multi=False)
                    ],
                    className="col-sm-6 dropdown"
                )
            ],
            className="row no-padding"
        ),
        dcc.Graph(id='Heatmap_met', figure=heatmap_met, className="heatmaps"),
        html.Div(
            html.P(children=["""This map shows the average behavior for the selected variable
                         along the month of the year (vertical axis) and the time along the day
                         (horizontal axis).""", html.Br(), html.Br(), """                        
                         See the list for available variables"""]),
            className="col-sm-12 select-label overlay map"
        )
    ],
    className="card"
)

heatmap_prob_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                         html.P(children="Frequency-based map for", className="text-label"),
                    ],
                    className="col-sm-6 back-title"
                ),
                html.Div(
                    [
                         dcc.Dropdown(id='Heatmap_prob_list', placeholder='Select variable to see',
                                options=[ {'label': var, 'value': idx} for idx, var in enumerate(Prob_vars)],
                                value=0,
                                multi=False),
                    ],
                    className="col-sm-6 dropdown"
                )
            ],
            className="row no-padding"
        ),      
        dcc.Graph(id='Heatmap_prob', figure=heatmap_prob, className="heatmaps"),
        html.Div(
            html.P(children=["""This map shows the frequency behavior for the selected variable
                         along the month of the year (vertical axis) and the time along the day
                         (horizontal axis).""", html.Br(), html.Br(), """                         
                         See the list for available variables."""]),
            className="col-sm-12 select-label overlay-map"
        )
    ],
    className="card"
)


#############################################
#               boxplot  layout             #
#############################################

boxplot_met_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                         html.P(children="frequency of distribution: ", className="text-label"),
                    ],
                    className="col-sm-6 back-title"
                ),
                html.Div(
                    [
                        dcc.Dropdown(id='Box_plot_met_list', placeholder='Select variable to see',
                            options=[ {'label': 'month', 'value': 0},
                                      {'label': 'hour', 'value': 1}],
                            value=0,
                            multi=False),
                    ],
                    className="col-sm-6 dropdown"
                )
            ],
            className="row no-padding"
        ),
        dcc.Graph(id='Box_plot_met', figure=box_met, className="heatmaps"),
        html.Div(
            html.P(children=["""Based on the variable selected for left map, this plot shows the distribution of the variable
                by month or hour along the period under analysis.""", html.Br(),html.Br(), """
                
                Time frames are divided into four categories:""", html.Br(),"""
                - Overnight: 12AM-6 AM""", html.Br(),"""
                - Morning: 6AM-12M""", html.Br(),"""
                - Afternoon: 12PM-6PM""", html.Br(),"""
                - Evening: 6PM-12AM."""]),
            className="col-sm-12 select-label overlay-map"
        )
    ],
    className="card"
)


#############################################
#                section layout             #
#############################################


analyze = html.Div(
    [
        analyze_heading, 
        html.Div( # Row 
            [
                html.Div( # outer left column 
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                html.H4("Exploring is the first step to knowledge, feel free to play around with the plots.", className="heading"),
                                html.P("""Heatmaps show the flow of a variable through time, for example average temperature
                                        through the day for a specific month in year.""", className="text-caption")
                                ],
                                className="mt-auto"
                            )
                        ],
                        className="container-img d-flex align-items-end flex-column text-center"
                    ),
                       
                    time_slider,             
                ],
                className="col-sm-6"
                ),
                html.Div( # Right column
                [
                    heatmap_prob_layout,
                ],
                className="col-sm-6 no-padding"
            )
            ],
            className="row no-padding"
        ),
        html.Div( # inner row 
        [   
                
            html.Div( # inner left column
                [
                    heatmap_met_layout,
                ],
                className="col-sm-6"
            ),
            html.Div( # inner right column
                [
                    boxplot_met_layout,
                ],
                className="col-sm-6 shaded-col"
            )
            ],
        className="row no-padding"
        ),
    ],
    id="analyze",
    className="offset align-content-center"
)



##################################################################################################
#                                   Section 3 - Inspect                                          #
##################################################################################################


#############################################
#                Section heading            #
#############################################

inspect_heading = html.Div(
        [
            html.Div(
                [
                    html.H3(children="Inspect about changes in future conditions", className="heading"),
                ],
                className="col-sm-12 text-center flex-start"
            ),
        ], 
        className="jumbotron container-fluid"
        )


#############################################
#         frequency map  layout             #
#############################################

frequency_map_layout = html.Div(
    [
        html.H5([
            html.Span("Non-operative conditions", className="inspect-title")
            ],
            className="inspect-border"
        ),
        dcc.Graph(id='Frequency_map', figure=frequency_map),   
        html.P("Frequencies at which horizontal and/or vertical visibility are below allowable limits for operation continuity"),
    ],
    className="d-block"
)



#############################################
#         weather widgets layout            #
#############################################

weather_widgets = html.Div(
    [
        html.Div(
            [
                html.Div(
                        [
                        ],
                        id="observations", 
                        className="aeris-wrapper"
                    )
            ],
            className="col-sm-5 align-content-center"
        ),
        html.Div(
            [
                html.H5([
                    html.Span("Wind distribution map", className="inspect-title")
                ],
                className="inspect-border"
                ),
                html.Iframe(src="https://www.meteoblue.com/en/weather/maps/widget?windAnimation=0&windAnimation=1&gust=0&gust=1&geoloc=4.70,-74.14&tempunit=C&windunit=kn&lengthunit=metric&zoom=10",\
                    className="meteomap", id="Weather_map", sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox"),
            ],
            className="col-sm-7"
        )
    ],
    className="row"
)

####################################################
#        map and frequency table layout            #
####################################################

map_freq_layout = html.Div(
    [
        html.Div(
            [
                html.Div( # 24h forecast
                    [
                    ],
                    id="forecast", 
                    className="aeris-wrapper"
                    ),
            ],
            className="col-sm-7"
        ),
        html.Div(
            [
                html.H5([
                    html.Span("Horizontal visibility @24h with forecast for 1h", className="inspect-title")
                ],
                className="inspect-border"
                ),

                dcc.Graph(id="H_vis_plot", figure=H_vis_plot)
            ],
            className="overlay-table"
        ),
        html.Div(
            [
                html.Div(
                    [
                        frequency_map_layout
                        
                    ],
                    className="col-sm-4"
                ),
                html.Div(
                    [
                        html.H5([
                            html.Span("Vertical visibility @24h with forecast for 1h", className="inspect-title")
                        ],
                        className="inspect-border"
                        ),
                        dcc.Graph(id="V_vis_plot", figure=V_vis_plot)
                    ],
                    className="col-sm-8"
                )
            ],
            className="row d-flex "
        )
    ],
    className="row"
)

#############################################
#                section layout             #
#############################################

inspect = html.Div(
    [
        inspect_heading,
        weather_widgets,
        map_freq_layout,      
    ],
    id="inspect",
    className="offset"
)




#################################################################################################
#                                   Section 4 - about                                           #
#################################################################################################
about_heading = html.Div(
    [
        html.Div(
        [
            html.Div(
                [
                    html.H3(children="About Weather-ball", className="heading")
                ],
                className="col-12 text-center flex-start"
            ),
        ], 
        className="jumbotron container-fluid"
        ),
    ],
    id="about",
    className="offset"
)



about = html.Div([
    about_heading,
    html.Div(
        [
        html.Div(
            [
                html.Iframe(src="https://view.genial.ly/5faf3a2ca083c60d0a8bed0e",\
                sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox", className="about")
            ],
            className="col-sm-10 d-flex align-items-center justify-content-center"
        )
        ],
        className="d-flex justify-content-center"
    )
])



#################################################################################################
#                                   Section 5 - Our Team                                        #
#################################################################################################

#############################################################
#                         People                            #
#############################################################


p_1 = "Luis M. Domínguez"
e_1 = html.P(["Industrial Engineer", html.Br(),"Spc. Strategic management"],className="text-center")
b_1 = html.Div(
    [
        html.P("Experience:", className="text-center"),
        html.Ul(
            [
            html.Li(
                html.P("Marketing")
            ),
            html.Li(
                html.P("BI tools design and development and data analytics for supporting business decision making")
            )
            ]
        ),
    ]
)

c_1 = html.Div(
            [
            html.A(
                [
                html.I(className="fab fa-linkedin fa-3x")
                ],
                href="https://www.linkedin.com/in/luisdominguezmoran", target="_blank"
            )
            ],
            className="d-flex align-items-end text-center justify-content-center card-footer"
        )




p_2 = "Leonardo Ramírez"
e_2 = html.P(["Mechatronic engineer", html.Br(), html.Br()], className="text-center") 
b_2 = html.Div(
    [
        html.P("Experience:", className="text-center"),
        html.Ul(
            [
            html.Li(
                html.P("Project Management/Agilism")
            ),
            html.Li(
                html.P("DB Design and Development")
            ),
            html.Li(
                html.P("RPA and AI tools with advanced forecasting and analytics models")
            )
            ]
        ),        
     ]
)
c_2 = html.Div(
            [
            html.A(
                [
                html.I(className="fab fa-linkedin fa-3x")
                ],
                href="https://www.linkedin.com/in/gleonardor-785880108", target="_blank"
            )
            ],
            className="d-flex align-items-end text-center justify-content-center card-footer"
        )


p_3 = "Pascual Ferrans PhD(e)"
e_3 = html.P(["Environmemtal Eng. BSc, MEng", html.Br(), html.Br()], className="text-center")  
b_3 = html.Div(
    [
        html.P("Experience:", className="text-center"),
        html.Ul(
            [
            html.Li(
                html.P("Operations Research")
            ),
            html.Li(
                html.P("Statistical data management")
            ),
            ]
        ),
    ]
)

c_3 = html.Div(
            [
            html.A(
                [
                html.I(className="fab fa-linkedin fa-3x")
                ],
                href="https://www.linkedin.com/in/pascual-ferrans-ram%C3%ADrez-654a86105/", target="_blank"
            )
            ],
            className="d-flex align-items-end text-center justify-content-center card-footer"
        )


p_4 = "Camilo Quiroga MSc."
e_4 = html.P(["Industrial Engineer", html.Br(), html.Br()], className="text-center") 
b_4 = html.Div(
    [
        html.P("Experience:", className="text-center"),
        html.Ul(
            [
            html.Li(
                html.P("Continuous improvement of processes")
            ),
            html.Li(
                html.P("Operations Control")
            ),
            html.Li(
                html.P("Advanced techniques in Statistics and optimization estrategies")
            ),
            ]
        ),
    ]
)

c_4 = html.Div(
            [
            html.A(
                [
                html.I(className="fab fa-linkedin fa-3x")
                ],
                href="https://www.linkedin.com/in/camiloquirogaing/", target="_blank"
            )
            ],
            className="d-flex align-items-end text-center justify-content-center card-footer"
        )


p_5 = "Daniel Mariño MSc(e)"
e_5 = html.P(["Economist", html.Br(), html.Br()], className="text-center") 
b_5 = html.Div(
    [
        html.P("Experience:", className="text-center"),
        html.Ul(
            [
            html.Li(
                html.P("Researching in labour markets, financial econometrics, risk and wellbeing")
            ),
            html.Li(
                html.P("Customer service analytics")
            ),
            ]
        ),
    ],
)

c_5 = html.Div(
            [
            html.A(
                [
                html.I(className="fab fa-linkedin fa-3x")
                ],
                href="https://www.linkedin.com/in/jdmarinou/", target="_blank"
            )
            ],
            className="d-flex align-items-end text-center justify-content-center card-footer"
        )

p_6 = "Manuel Serrano MBA-PMP"
e_6 = html.P(["Mechanical Engineer", html.Br(),"Industrial Designer"],className="text-center")
         
b_6 = html.Div(
    [
        html.P("Experience:", className="text-center"),
        html.Ul(
            [
            html.Li(
                html.P("Strategic consulting")
            ),
            html.Li(
                html.P("R&D and product development")
            ),
            ]
        ),
    ]
)

c_6 =html.Div(
            [
            html.A(
                [
                html.I(className="fab fa-linkedin fa-3x")
                ],
                href="https://www.linkedin.com/in/manuel-leonardo-serrano-rey-b7238811/", target="_blank"
            )
            ],
            className="d-flex align-items-end text-center justify-content-center card-footer"
        )

##################################################################
#                       Team section                             #
##################################################################

team = html.Div(
    [
        html.Div(
        [
            
            html.Div(
                [
                    html.H3(children="Our Team", className="heading"),
                    html.Hr(className="my-4"),
                    html.H5(children="Who made this a reality",  className="heading")
                ],
                className="col-12 text-center flex-start"
            ),
        ], 
        className="jumbotron container-fluid"
        ),
        html.Div( #row with 3 columns
            [
                html.Div(  # first column
                    [
                        html.Div( #Card
                            [
                                html.Img(src="assets/pic_1.png", className="card-img-top mx-auto"),
                                html.Div(
                                    [
                                        html.H5(p_1, className="text-center"),
                                        e_1,
                                        b_1,
                                    ],
                                    className="card-body"
                                ),
                                c_1
                            ],
                            className="card h-100 d-flex"
                        ),
                    ],
                    className="col-md-3 .card-padding"
                ),
                html.Div(  # second column
                [
                    html.Div( #Card
                        [
                            html.Img(src="assets/pic_2.png", className="card-img-top mx-auto"),
                            html.Div(
                                [
                                    html.H5(p_2, className="text-center"),
                                    e_2,
                                    b_2,
                                ],
                                className="card-body"
                            ),
                            c_2
                        ],
                        className="card h-100 d-flex"
                    ),
                ], 
                className="col-md-3 .card-padding"
                ),
                html.Div(  # third column
                [
                    html.Div( #Card
                        [
                            html.Img(src="assets/pic_3.png", className="card-img-top mx-auto"),
                            html.Div(
                                [
                                    html.H5(p_3, className="text-center"),
                                    e_3,
                                    b_3,
                                ],
                                className="card-body"
                            ),
                            c_3
                        ],
                        className="card h-100 d-flex"
                    ),
                ], 
                className="col-md-3 .card-padding"
                ),             
            ],
            className="row equal d-flex justify-content-center"
        ),
        html.Div( # Second row
            [
                html.Div(  # first column
                    [
                        html.Div( #Card
                            [
                                html.Img(src="assets/pic_4.png", className="card-img-top mx-auto"),
                                html.Div(
                                    [
                                        html.H5(p_4, className="text-center"),
                                        e_4,
                                        b_4,
                                    ],
                                    className="card-body"
                                ),
                                c_4
                            ],
                            className="card h-100 d-flex"
                        ),
                    ],
                    className="col-md-3 .card-padding"
                ),
                html.Div(  # second column
                [
                    html.Div( #Card
                        [
                            html.Img(src="assets/pic_5.png", className="card-img-top mx-auto"),
                            html.Div(
                                [
                                    html.H5(p_5, className="text-center"),
                                    e_5,
                                    b_5,
                                ],
                                className="card-body"
                            ),
                            c_5
                        ],
                        className="card h-100 d-flex"
                    ),
                ], 
                className="col-md-3 .card-padding"
                ),
                html.Div(  # third column
                [
                    html.Div( #Card
                        [
                            html.Img(src="assets/pic_6.png", className="card-img-top mx-auto"),
                            html.Div(
                                [
                                    html.H5(p_6, className="text-center"),
                                    e_6,
                                    b_6,
                                ],
                                className="card-body"
                            ),
                            c_6
                        ],
                        className="card h-100 d-flex"
                    ),
                ],
                className="col-md-3 .card-padding"
                ),
            ],
            className="row equal d-flex justify-content-center"
        )
    ],
    id="team",
    className="offset"
)


#################################################################################################
#                                   Section 6 - Contact us                                      #
#################################################################################################

contact = html.Footer(
    [
        html.H4("Our Sponsors", className="heading text-center"),
        html.Div(
            [
                html.Div(html.Img(src="assets/ds4a_logo.png"), id="logo_1"),
                html.Div(html.Img(src="assets/mintic_logo.png"), id="logo_2"),
                html.Div(html.Img(src="assets/c1_logo.png"),id="logo_3"),
            ],
            className="d-flex px-md-5 justify-content-center align-items-center",
            id="sponsors"

        ),
    ]
)

################################################################################################################
#                                            Main   DASH                                                       #
################################################################################################################

app.layout = html.Div(
    [
        visdcc.Run_js(id = 'javascript'),
        dcc.Interval(
            id='interval-component-5min',
            interval=5*60*1000, # in milliseconds
            n_intervals=0
        ),
        explore,
        analyze,
        inspect,
        about,
        team,
        contact

    ]
)


####################################################
#                   Callbacks                      #
####################################################

''' Inputs from Map '''

# Update maps and info 


# From the map

@app.callback(
    dash.dependencies.Output('Airports_list', 'value'),
    [dash.dependencies.Input('Airports_map', 'clickData')],
    prevent_initial_call=True)
def update_airport_from_map(clickData):
    # clickData is the data for the airport the user clicked to
    value = clickData['points'][0]["customdata"][-1]
    return value

# From the Dropdown menu

@app.callback(
    [dash.dependencies.Output('City', 'children'),
     dash.dependencies.Output('Location', 'children'),
     dash.dependencies.Output('Altitude', 'children'),
     dash.dependencies.Output('Heatmap_met_list', 'value'),
     dash.dependencies.Output('Heatmap_prob_list', 'value'),
     dash.dependencies.Output('Box_plot_met_list', 'value'),
     dash.dependencies.Output('Airports_map', 'figure'),
     dash.dependencies.Output('observations', 'children'),  # Div for Aerisweather
     dash.dependencies.Output('forecast', 'children'), # Div for Aerisweather
     dash.dependencies.Output('Weather_map', 'src'), # Div for MeteoBlue
     dash.dependencies.Output('Airport_pic', 'style'), # Div for Airport Picture
     dash.dependencies.Output('Frequency_map', 'figure'),
     dash.dependencies.Output('H_vis_plot', 'figure'),
     dash.dependencies.Output('V_vis_plot', 'figure')
    ],
    [dash.dependencies.Input('Airports_list', 'value')],
    [dash.dependencies.State('Airports_map', 'figure')],
    prevent_initial_call=True)
def update_airport_from_dropdown(value, fig):
    # Value is the iata Code from the airport

    ### Update General info
    City = airports[airports['iata_code'] == value]['municipality'].array[0]
    Latitude = airports[airports['iata_code'] == value]['latitude_deg'].array[0]
    Longitude = airports[airports['iata_code'] == value]['longitude_deg'].array[0]
    fig['layout']['mapbox']['center'] = {'lat': Latitude, 'lon':Longitude}
    fig['layout']['mapbox']['zoom'] = 12

    Location = f"Location: Lat: {np.round(Latitude,4)}, Lon: {np.round(Longitude,4)}"
    Altitude = f"Altitude (ft. asl): {airports[airports['iata_code'] == value]['elevation_ft'].array[0]}"
       
    # Update Data and Maps     

    # Update meteorological with default values
    heatmap_met_list_value = 0
    heatmap_prob_list_value = 0
    box_plot_met_list = 0


    # Div containers for metereological data
    observations = ""
    forecast = ""

    # Update Iframe for wind map
    zoom = 12

    src = "https://www.meteoblue.com/en/weather/maps/widget?windAnimation=0&windAnimation=1&gust=0&gust=1&tempunit=C&windunit=kn&lengthunit=metric&" + \
          "zoom=" + str(zoom) + "#coords=" + str(zoom) + "/" + str(np.round(Latitude,3)) + "/" + str(np.round(Longitude,3)) + \
          "&map=windAnimation~rainbow~auto~10%20m%20above%20gnd~none"

    # Airport pic
    airport_pic_style = {'background-image': 'url(/assets/' + value +'.jpg)'}

    # Filter data for the airport selected by the user - 
    # required for prediction plots
    airport_weather_data = set_current_airport(weather_data, value)

    # Update frequency table
    op_conditions = operation_conditions(value) 
    non_op_freq = frequency_table(airport_weather_data, op_conditions)
    frequency_map = create_frequency_map(non_op_freq)

    # Update prediction plots
    model_results = model_predict(value,engine)
    H_vis_plot, V_vis_plot = create_predict_plots(model_results,op_conditions)


    return City, Location, Altitude, heatmap_met_list_value, heatmap_prob_list_value, box_plot_met_list, fig,\
           observations, forecast, src, airport_pic_style, frequency_map, H_vis_plot, V_vis_plot 
    


''' Aeris weather divs'''

@app.callback(
     dash.dependencies.Output('javascript', 'run'),
    [dash.dependencies.Input('observations', 'children'),
     dash.dependencies.Input('forecast', 'children'),
     dash.dependencies.Input('interval-component-5min', 'n_intervals')],
     dash.dependencies.State('Airports_list', 'value'))
def update_aeris_data(observations, forecast, intervals, value):

# Java script for metereological data Aerisweather
    Latitude = airports[airports['iata_code'] == value]['latitude_deg'].array[0]
    Longitude = airports[airports['iata_code'] == value]['longitude_deg'].array[0]

    map_point = str(np.round(Latitude,2)) + "," + str(np.round(Longitude,2))

    aeris_data= '''
        $('#forecast').empty();
        $('#observations').empty();

        const aeris = new AerisWeather('lAcjWvB3gsckg4vjDBxS0', 'u7u5ZDzEkbFs1krtiXiv8RXLrMwfzszboVvQNYhv');
        aeris.on('ready', () => {
            // create desired WeatherBlox instance
            var daynight = new aeris.wxblox.views.DayNightForecast('#forecast', {
                metric: true
            });

            var observations = new aeris.wxblox.views.Observations('#observations', {
                advisories: {
                    enabled: true
                },
                threats: {
                    enabled: true
                },
                metric: true
            });

            var params = { p: "''' + map_point + '''"}

            // load data and render the view for a specific location
            daynight.load(params);
            observations.load(params);   
        });
    
    '''
    return aeris_data


''' Inputs from heatmaps and lists'''

# Metereologogical heatmap 
@app.callback(
    dash.dependencies.Output('Heatmap_met', 'figure'),
    [dash.dependencies.Input('Heatmap_met_list', 'value'),
    dash.dependencies.Input('RS_period', 'value')],
    dash.dependencies.State('Airports_list', 'value'),
    prevent_initial_call=True)
def update_heatmap_met_from_dropdown(value, period, airport):

    # Filter data for the airport selected by the user
    airport_weather_data = set_current_airport(weather_data, airport)

    # Define period for the data
    airport_weather_data_period = adjust_period(period[0], period[1], airport_weather_data)

    # Compute data for selected meteorological variable
    met_var = Met_vars[value]
    met_unit = Met_vars_units[value]
    
    heatmap_met_data = compute_meteorological_heatmap(airport_weather_data_period, met_var)
    heatmap_met = create_meteorological_heatmap(heatmap_met_data, met_var, met_unit) 
    
    return heatmap_met

#  Probabilistic heatmap 

@app.callback(
    dash.dependencies.Output('Heatmap_prob', 'figure'),
    [dash.dependencies.Input('Heatmap_prob_list', 'value'),
     dash.dependencies.Input('RS_period', 'value')],
    dash.dependencies.State('Airports_list', 'value'),
    prevent_initial_call=True)
def update_heatmap_prob_from_dropdown(value, period, airport):

    # Filter data for the airport selected by the user
    airport_weather_data = set_current_airport(weather_data, airport)

    # Define period for the data
    airport_weather_data_period = adjust_period(period[0], period[1], airport_weather_data)

    # Compute data for selected variable
    prob_var = Prob_vars[value]
    heatmap_prob_data = compute_probabilistic_heatmap(airport_weather_data_period, prob_var)
    heatmap_prob = create_probabilistic_heatmap(heatmap_prob_data, prob_var) 
    
    return heatmap_prob



@app.callback(
    dash.dependencies.Output('Box_plot_met', 'figure'),
    [dash.dependencies.Input('Box_plot_met_list', 'value'),
     dash.dependencies.Input('Heatmap_met_list', 'value'),
     dash.dependencies.Input('RS_period', 'value')],
    dash.dependencies.State('Airports_list', 'value'),
    prevent_initial_call=True)
def update_heatmap_prob_from_dropdown(value_box, value_met, period, airport):

    Box_vars = {0 : 'month', 1: 'hour'}

    # Filter data for the airport selected by the user
    airport_weather_data = set_current_airport(weather_data, airport)

    # Define period for the data
    airport_weather_data_period = adjust_period(period[0], period[1], airport_weather_data)

    # Compute data for selected variable
    box_var = Box_vars[value_box] # variable for boxplot
    met_var = Met_vars[value_met] # variable from met heatmap

    box_met_data = compute_metereological_boxplot(airport_weather_data_period, met_var, box_var)
    box_met = create_metereological_boxplot(box_met_data, met_var, box_var)
    
    return box_met


''' End callbacks '''

if __name__ == '__main__':
    app.run_server(debug=True)
