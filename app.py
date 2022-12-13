#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template,jsonify, request
from model.model import *
import pandas as pd
import re
import numpy as np
from flask.json import JSONEncoder
from datetime import date
from collections import Counter
from wordcloud import STOPWORDS

DEVELOPMENT_ENV  = True

app = Flask(__name__)

app_data = {
    "name":         "BIS634 Final Project",
    "html_title":   "COVID-19 Vaccination Tweets Research",
    "project_name": "BIS634 Final Project",
}
app.config['JSON_SORT_KEYS'] = False

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 3,   
    output_attentions = False,
    output_hidden_states = False,
)
#load the pretrained model
state_dict = torch.load('model/checkpoint.pth',  map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

vax_data = pd.read_csv('model/vax_data_cleaned.csv')
vax_data['date'] = pd.to_datetime(vax_data['date'], errors='coerce').dt.strftime('%Y-%m-%d')
vax_data = vax_data.dropna()
all_vax = ['covaxin', 'sinopharm', 'sinovac', 'moderna', 'pfizer', 'biontech', 'oxford', 'astrazeneca', 'sputnik']
countries=['india','usa','canada','germany','spain','pakistan','uk','brazil','russia','italy','australia','france','argentina','uae','israel','mexico','japan']
states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
city_names = ["Aberdeen", "Abilene", "Akron", "Albany", "Albuquerque", "Alexandria", "Allentown", "Amarillo", "Anaheim", "Anchorage", "Ann Arbor", "Antioch", 
"Apple Valley", "Appleton", "Arlington", "Arvada", "Asheville", "Athens", "Atlanta", "Atlantic City", "Augusta", "Aurora", "Austin", "Bakersfield", "Baltimore", 
"Barnstable", "Baton Rouge", "Beaumont", "Bel Air", "Bellevue", "Berkeley", "Bethlehem", "Billings", "Birmingham", "Bloomington", "Boise", "Boise City", 
"Bonita Springs", "Boston", "Boulder", "Bradenton", "Bremerton", "Bridgeport", "Brighton", "Brownsville", "Bryan", "Buffalo", "Burbank", "Burlington", "Cambridge", 
"Canton", "Cape Coral", "Carrollton", "Cary", "Cathedral City", "Cedar Rapids", "Champaign", "Chandler", "Charleston", "Charlotte", "Chattanooga", "Chesapeake", 
"Chicago", "Chula Vista", "Cincinnati", "Clarke County", "Clarksville", "Clearwater", "Cleveland", "College Station", "Colorado Springs", "Columbia", "Columbus", 
"Concord", "Coral Springs", "Corona", "Corpus Christi", "Costa Mesa", "Dallas", "Daly City", "Danbury", "Davenport", "Davidson County", "Dayton", "Daytona Beach", 
"Deltona", "Denton", "Denver", "Des Moines", "Detroit", "Downey", "Duluth", "Durham", "El Monte", "El Paso", "Elizabeth", "Elk Grove", "Elkhart", "Erie", "Escondido", 
"Eugene", "Evansville", "Fairfield", "Fargo", "Fayetteville", "Fitchburg", "Flint", "Fontana", "Fort Collins", "Fort Lauderdale", "Fort Smith", "Fort Walton Beach", 
"Fort Wayne", "Fort Worth", "Frederick", "Fremont", "Fresno", "Fullerton", "Gainesville", "Garden Grove", "Garland", "Gastonia", "Gilbert", "Glendale", "Grand Prairie", 
"Grand Rapids", "Grayslake", "Green Bay", "GreenBay", "Greensboro", "Greenville", "Gulfport-Biloxi", "Hagerstown", "Hampton", "Harlingen", "Harrisburg", "Hartford", 
"Havre de Grace", "Hayward", "Hemet", "Henderson", "Hesperia", "Hialeah", "Hickory", "High Point", "Hollywood", "Honolulu", "Houma", "Houston", "Howell", "Huntington", 
"Huntington Beach", "Huntsville", "Independence", "Indianapolis", "Inglewood", "Irvine", "Irving", "Jackson", "Jacksonville", "Jefferson", "Jersey City", "Johnson City", 
"Joliet", "Kailua", "Kalamazoo", "Kaneohe", "Kansas City", "Kennewick", "Kenosha", "Killeen", "Kissimmee", "Knoxville", "Lacey", "Lafayette", "Lake Charles", "Lakeland",
"Lakewood", "Lancaster", "Lansing", "Laredo", "Las Cruces", "Las Vegas", "Layton", "Leominster", "Lewisville", "Lexington", "Lincoln", "Little Rock", "Long Beach",
"Lorain", "Los Angeles", "Louisville", "Lowell", "Lubbock", "Macon", "Madison", "Manchester", "Marina", "Marysville", "McAllen", "McHenry", "Medford", "Melbourne",
"Memphis", "Merced", "Mesa", "Mesquite", "Miami", "Milwaukee", "Minneapolis", "Miramar", "Mission Viejo", "Mobile", "Modesto", "Monroe", "Monterey", "Montgomery",
"Moreno Valley", "Murfreesboro", "Murrieta", "Muskegon", "Myrtle Beach", "Naperville", "Naples", "Nashua", "Nashville", "New Bedford", "New Haven", "New London",
"New Orleans", "New York", "New York City", "Newark", "Newburgh", "Newport News", "Norfolk", "Normal", "Norman", "North Charleston", "North Las Vegas", "North Port",
"Norwalk", "Norwich", "Oakland", "Ocala", "Oceanside", "Odessa", "Ogden", "Oklahoma City", "Olathe", "Olympia", "Omaha", "Ontario", "Orange", "Orem", "Orlando",
"Overland Park", "Oxnard", "Palm Bay", "Palm Springs", "Palmdale", "Panama City", "Pasadena", "Paterson", "Pembroke Pines", "Pensacola", "Peoria", "Philadelphia",
"Phoenix", "Pittsburgh", "Plano", "Pomona", "Pompano Beach", "Port Arthur", "Port Orange", "Port Saint Lucie", "Port St. Lucie", "Portland", "Portsmouth",
"Poughkeepsie", "Providence", "Provo", "Pueblo", "Punta Gorda", "Racine", "Raleigh", "Rancho Cucamonga", "Reading", "Redding", "Reno", "Richland", "Richmond",
"Richmond County", "Riverside", "Roanoke", "Rochester", "Rockford", "Roseville", "Round Lake Beach", "Sacramento", "Saginaw", "Saint Louis", "Saint Paul",
"Saint Petersburg", "Salem", "Salinas", "Salt Lake City", "San Antonio", "San Bernardino", "San Buenaventura", "San Diego", "San Francisco", "San Jose", "Santa Ana",
"Santa Barbara", "Santa Clara", "Santa Clarita", "Santa Cruz", "Santa Maria", "Santa Rosa", "Sarasota", "Savannah", "Scottsdale", "Scranton", "Seaside", "Seattle",
"Sebastian", "Shreveport", "Simi Valley", "Sioux City", "Sioux Falls", "South Bend", "South Lyon", "Spartanburg", "Spokane", "Springdale", "Springfield", "St. Louis",
"St. Paul", "St. Petersburg", "Stamford", "Sterling Heights", "Stockton", "Sunnyvale", "Syracuse", "Tacoma", "Tallahassee", "Tampa", "Temecula", "Tempe", "Thornton",
"Thousand Oaks", "Toledo", "Topeka", "Torrance", "Trenton", "Tucson", "Tulsa", "Tuscaloosa", "Tyler", "Utica", "Vallejo", "Vancouver", "Vero Beach", "Victorville",
"Virginia Beach", "Visalia", "Waco", "Warren", "Washington", "Waterbury", "Waterloo", "West Covina", "West Valley City", "Westminster", "Wichita", "Wilmington",
"Winston", "Winter Haven", "Worcester", "Yakima", "Yonkers", "York", "Youngstown"]
states.extend(city_names)
states.append('usa')
states = [rf'\b{x.lower()}\b' for x in states]

@app.route('/')
def index():
    return render_template('index.html', app_data=app_data)


@app.route('/data')
def program():
    return render_template('data.html', app_data=app_data)

@app.route('/wordcloud')
def wordcloud():
    return render_template('wordcloud.html', app_data=app_data)

@app.route('/wordcloud_api/<string:vax_name>')
def wordcloud_api(vax_name):
    if vax_name=='all':
        vax_name=vax_data
    else :   
        vax_name=vax_data[vax_data['text'].str.lower().str.contains(vax_name)]
    
    user_text = ' '.join(vax_name['text'])
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    user_text = url_pattern.sub(r'', user_text)
    user_text = re.sub('\S*@\S*\s?', '', user_text)
    user_text = re.sub('\s+', ' ', user_text)
    user_text = re.sub("\'", "", user_text)
    user_text = re.sub("#", "", user_text)
    user_text = re.sub(r'[^\w\s]', '', user_text.lower())

    stopwords = set(STOPWORDS)
    stopwords.update(["t", "co", "https", "amp", "U", "vaccine", "covid", "covid19",
                    'will', 'got', 'vaccination', 'vaccinated', 'vaccines', 'covidvaccine',
                    'coronavirus'])
   
    counts = Counter([item for item in user_text.split() if item not in stopwords])
    myDict = dict(counts.most_common(50))
    l = []
    for key, value in myDict.items():
        l.append({'x':key,
                 'value':value})
    return jsonify(l)

@app.route('/wordcloud_api/senti/<string:senti>')
def wordcloud_api_senti(senti):
    if senti == 'all':
        s = vax_data
    else:
        s=vax_data[vax_data['Sentiment'].map({-1:'negative',0:'neutral',1:'positive'}) == senti]
    
    user_text = ' '.join(s['text'])
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    user_text = url_pattern.sub(r'', user_text)
    user_text = re.sub('\S*@\S*\s?', '', user_text)
    user_text = re.sub('\s+', ' ', user_text)
    user_text = re.sub("\'", "", user_text)
    user_text = re.sub("#", "", user_text)
    user_text = re.sub(r'[^\w\s]', '', user_text.lower())

    stopwords = set(STOPWORDS)
    stopwords.update(["t", "co", "https", "amp", "U", "vaccine", "covid", "covid19",
                    'will', 'got', 'vaccination', 'vaccinated', 'vaccines', 'covidvaccine',
                    'coronavirus'])
   
    counts = Counter([item for item in user_text.split() if item not in stopwords])
    myDict = dict(counts.most_common(50))
    l = []
    for key, value in myDict.items():
        l.append({'x':key,
                 'value':value})
    return jsonify(l)


@app.route('/wordcloud_api/country/<string:country>')
def wordcloud_api_country(country):
    if country=='all':
        count=vax_data
    elif country == 'usa':
        count=vax_data[vax_data['user_location'].str.lower().str.contains('|'.join(states))]
    elif country == 'uk':
        loc = ['uk', 'england']
        loc = [rf'\b{x.lower()}\b' for x in loc]
        count=vax_data[vax_data['user_location'].str.lower().str.contains('|'.join(loc))]
    else : 
        count=vax_data[vax_data['user_location'].str.lower().str.contains(country)]
    
    user_text = ' '.join(count['text'])
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    user_text = url_pattern.sub(r'', user_text)
    user_text = re.sub('\S*@\S*\s?', '', user_text)
    user_text = re.sub('\s+', ' ', user_text)
    user_text = re.sub("\'", "", user_text)
    user_text = re.sub("#", "", user_text)
    user_text = re.sub(r'[^\w\s]', '', user_text.lower())

    stopwords = set(STOPWORDS)
    stopwords.update(["t", "co", "https", "amp", "U", "vaccine", "covid", "covid19",
                    'will', 'got', 'vaccination', 'vaccinated', 'vaccines', 'covidvaccine',
                    'coronavirus'])
   
    counts = Counter([item for item in user_text.split() if item not in stopwords])
    myDict = dict(counts.most_common(50))
    l = []
    for key, value in myDict.items():
        l.append({'x':key,
                 'value':value})
    return jsonify(l)


@app.route('/timeseries')
def timeseries():
    return render_template('timesseries.html', app_data=app_data)

@app.route('/timeseries_api/varianceVStime/<string:vax_name>')
def timeseries_api_varianceVStime(vax_name):
    if vax_name=='all':
        vax_name=vax_data
    else :   
        vax_name=vax_data[vax_data['text'].str.lower().str.contains(vax_name)]

    temp=pd.DataFrame()
    temp['date'] = sorted(vax_name['date'].unique())
    senti=list()

    for date in temp['date']:
        senti.append(vax_name[vax_name['date']==date].Sentiment.mean())

    temp['Sentiment']=senti
    return jsonify([{
        "x": temp.date.to_list(),
        "y": temp.Sentiment.to_list(),
        "mode": "lines+markers",
        "type": "scatter"
    }])

@app.route('/timeseries_api/varianceVStime_Country/<string:country>')
def timeseries_api_varianceVStime_Country(country):
    if country=='all':
        count=vax_data
    elif country == 'usa':
        count=vax_data[vax_data['user_location'].str.lower().str.contains('|'.join(states))]
    elif country == 'uk':
        loc = ['uk', 'england']
        loc = [rf'\b{x.lower()}\b' for x in loc]
        count=vax_data[vax_data['user_location'].str.lower().str.contains('|'.join(loc))]
    else : 
        count=vax_data[vax_data['user_location'].str.lower().str.contains(country)]

    temp=pd.DataFrame()
    temp['date'] = sorted(count['date'].unique())
    senti=list()

    for date in temp['date']:
        senti.append(count[count['date']==date].Sentiment.mean())

    temp['Sentiment']=senti
    return jsonify([{
        "x": temp.date.to_list(),
        "y": temp.Sentiment.to_list(),
        "mode": "lines+markers",
        "type": "scatter"
    }])

@app.route('/timeseries_api/SDT/<string:senti>')
def timeseries_api_SDT(senti):
    if senti == 'all':
        s = vax_data
    else:
        s=vax_data[vax_data['Sentiment'].map({-1:'negative',0:'neutral',1:'positive'}) == senti]
    temp = s.groupby(['date'])['date'].count().reset_index(name="count")
    return jsonify([{
        "x": temp.date.to_list(),
        "y": temp['count'].tolist(),
        "type": "bar"
    }])

@app.route('/model_page', methods=["GET", "POST"])
def model_page():
    return render_template('model.html', app_data=app_data)

@app.route('/model_result')
def model_result():
    user_text = request.args.get('user_text')
    dict = model_api(user_text).get_json()
    return render_template('model.html', app_data=app_data, user_text = user_text, dict = dict)


@app.route('/model_api/<string:user_text>')
def model_api(user_text):
    prediction,prob = predict_sentiment(model,tokenizer,user_text)
    label,negative,neutral,positive = get_sentiment(prediction,prob)
    return jsonify({"Predicted Sentiment": label,
                    "negative probability": float(negative[1]),
                    "neutral probability": float(neutral[1]),
                    "positive probability": float(positive[1])})

@app.route('/contact')
def contact():
    return render_template('contact.html', app_data=app_data)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

if __name__ == '__main__':
    app.run(debug=DEVELOPMENT_ENV)