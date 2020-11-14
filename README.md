# WEATHER-BALL APP

## What is Weather-ball
Weather-ball is an app mainly developed with [Dash](https://plotly.com/dash/), it was made for airports and airlines and its a hand-tool for analyst that need to make decision in critical moments, for example to take off an aircraft that could be in the air wasting gas and time for the bad weather in the arrival airport. This tool show historical data for the airport and could help the analyst doing the flight plans. 

This app is develop with the follows motivations

  - **Airlines**: Mitigate delays caused by adverse weather conditions, by avoiding scheduling flights when there is a high probability of these types of events, such as    reduced visibility and thunder, therefore guaranteeing punctuality. Reducing gasoline consumption by keeping a flight on ground that was scheduled to a destination where there may be weather conditions that would restrict the aircraft from landing. 
  - **Airports**: When there are adverse conditions surrounding an airport, they get saturated. Therefore it’s relevant to have an adequate weather management tool that will enable them to safely and efficiently execute their different processes. 
  
Please be free to visit [Weather-Ball](http://www.weather-ball.com)
  
## What can you find here

For this repository you cand find all the resources (for Research and develop) and code that helps to build the app.

  - **Resources**: Here you cand find all the notebooks, documents, and information that was collected in the firsts step.
  - **App**: Here you can find the code developed in python for the Front-end and pipelines that made this app run, the pipelines are scripts to automate the ingest of data from the APIs, and the front-end is the script in charge to consume the data from the postgresql instance, load the model for the correspondent airport city and show all the historical data.
