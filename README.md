<img style="float: right; width:200px;" src="App/Front/assets/wb_logo.png">

# WEATHER-BALL APP   
## What is Weather-ball
Weather-ball is an app mainly developed with [Dash](https://plotly.com/dash/), it was made for airports and airlines and it is a handy-tool for analysts that need to make decisions in critical moments, as for example, to decide whether or not to take off an aircraft that could be otherwise in the air wasting gas and time for the bad weather during its arrival to an airport. This tool shows historical data for the airport and could help analysts doing the flight plans. 

This app was developed having in mind the benefits for the following aviation actors:

  - **Airlines**: Mitigate delays caused by adverse weather conditions, by avoiding scheduling flights when there is a high probability of these types of events, such as reduced visibility and thunder, therefore guaranteeing punctuality. Reducing gasoline consumption by keeping a flight on ground that was scheduled to a destination where there may be weather conditions that would restrict the aircraft from landing. 
  - **Airports**: When there are adverse conditions surrounding an airport, they get saturated. Therefore it’s relevant to have an adequate weather management tool that will enable them to safely and efficiently execute their different processes. 
  
Please be free to visit [Weather-Ball](https://weather-ball.com)
  
## What can you find here

Inside this repository you can find all resources (for Research and development) and code used to built the app.

  - **Resources**: Here you cand find all notebooks, documents, and information that was collected in the first step.
  - **App**: Here you can find the code developed in Python for the Front-end and pipelines that made this app run, the pipelines are scripts to automate the ingest of data from the APIs, and the front-end is the script in charge to consume the data from the postgresql instance, load the model for the correspondent airport city and show all the historical data.
  
  **Libraries** the required librarys to run this app are in the file requeriments.txt, you can install with the command pip install -r requeriments.txt
