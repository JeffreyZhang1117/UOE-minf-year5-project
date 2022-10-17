# from help_function.airspeck_plot import GOOGLE_MAPS_API_KEY
from PIL import Image
from io import BytesIO
import requests
import plotly.express as px
GOOGLE_MAPS_API_KEY = "AIzaSyDBzpTwCrP8_oyCtJIJYQUfp3JAFDEvZYY"
def plot_subject_locations(markers, map_parameters, api_key=GOOGLE_MAPS_API_KEY):
    """
    :param markers: a dictionary, key: value where:
    key = subject ID + other info (e.g. home, work etc)
    value = {
    'size': size,
    'color': color, 
    'label': One alphanumeric character,
    'location': (lat, lon) -> tuple
    }
    :param map_parameters: A dictionary containing information such as map size, map type and scale. 
    """
    # parse the marker list
    marker_list = []
    
    for subject_id, marker_dict in markers.items():
        lat, lon = marker_dict['location']
        new_str = f"size:{marker_dict['size']}%7Ccolor:{marker_dict['color']}%7Clabel:{marker_dict['label']}%7C{lat},{lon}"
        marker_list.append(new_str)
    
    map_parameters['key'] = api_key
    map_parameters['markers'] = marker_list
    
    url = 'http://maps.googleapis.com/maps/api/staticmap?'
    
    # form the URL
    # we do it this way because when applying multiple markers with different style
    # we must append multiple 'marker' tags to the URL
    for key, val in map_parameters.items():
        if type(val) != list:
            url += f"&{key}={val}"
        else:
            for elem in val:
                url+= f"&{key}={elem}"
            
    print(f"Final url = {url}")
    
    response = requests.get(url)
    response.raise_for_status()
    
    return Image.open(BytesIO(response.content))


###########plot iteractive map by using plotly
# fig = px.scatter_mapbox(asthma_airspeck_data, 
#                         lat=asthma_airspeck_data['gpsLatitude'], 
#                         lon=asthma_airspeck_data['gpsLongitude'],
#                        color=asthma_airspeck_data['pm2_5'], size_max=10,
#                        range_color=[0, 100], opacity=1, zoom=20,
#                        hover_name=asthma_airspeck_data['subject_id'])
# fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=10,
#                   mapbox_center = {"lat": 51.508610, "lon": -0.163611})
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.update_layout(
#     coloraxis_colorbar=dict(
#         title="PM2.5",
#     ),
# )
# fig.show()
# fig.write_html("./plots/inhale_asthma/asthma_inhale_interactive_map_all.html")