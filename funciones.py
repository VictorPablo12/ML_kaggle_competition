import re
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN

def bath_clean(x):
    bath_n=0
    patron='[0-9]+'
    try:
        bath_n=int(re.findall(patron, x)[0])
    except:
        bath_n=0
        
    return bath_n

def cats_clean(x):
    
    return str(x).replace('[', '').replace(']', '').replace("'", '').split(',')


def clean(df):
       
    #Casteo datetime:
    df['host_since'] = df['host_since'].astype(str) #casteo datetime
    df['host_since'] = pd.to_datetime(df['host_since'], format='%Y-%m-%d')
    
    #Creo nueva columna y dropeo la existente:                                                                
    df['host_since_year'] = pd.DatetimeIndex(df['host_since']).year 
    df['host_since_year'].head()  
    
    #Cluster latitud,longitud
    hdbscan=HDBSCAN(min_cluster_size=3,min_samples=2,cluster_selection_epsilon=0.5,
                    allow_single_cluster=False,prediction_data=True,leaf_size=30)
    
    locs = df[['longitude', 'latitude']] #Dataframe solo con latitudes y longitudes
    hdbscan.fit(locs)
    df['label_ubication']=hdbscan.fit_predict(locs)
    
    #Baños:
    df['bathroom']=df.bathrooms_text.apply(bath_clean)
    
    #Categories:-->Quito esto por la importancia de las columnas del report de H2O
    #df['cats']=df.host_verifications
    #df.cats=df.cats.apply(cats_clean)
    #
    #df.cats=df.cats.apply(lambda x: x[0])
    #df=df.reset_index(drop=True)
    #df=pd.concat([df, pd.get_dummies(df.cats).reset_index(drop=True)], axis=1)
    
    #Dumies room_type:
    #pd.get_dummies(df, columns = ['room_type'])
    #df=pd.concat([df, pd.get_dummies(df, columns = ['room_type']).reset_index(drop=True)], axis=1)
    #Casting beds:
    df.beds = df.beds.fillna(0)
    df.beds = df.beds.astype(int)
    
    #Rellenando Super_host:
    df.host_is_superhost= df.host_is_superhost.fillna('f')
    df=pd.concat([df, pd.get_dummies(df['room_type']).reset_index(drop=True)], axis=1)
    
    
    #Dropeo todo lo que no quiero:
    drop=['id','host_identity_verified','neighbourhood_group_cleansed',
    'calendar_updated','bathrooms','host_neighbourhood','host_about','listing_url','scrape_id','last_scraped',
    'name','description','neighborhood_overview','picture_url','host_id','host_url','host_name',
    'host_location','host_response_time','host_response_rate','host_acceptance_rate',
    'host_since','review_scores_communication',
    'review_scores_location','review_scores_value','review_scores_checkin','review_scores_accuracy',
    'review_scores_cleanliness','first_review','last_review','reviews_per_month','host_thumbnail_url','host_picture_url','host_total_listings_count','host_has_profile_pic',
    'neighbourhood','neighbourhood_cleansed','property_type','bathrooms_text','amenities',
    'minimum_nights','maximum_nights','minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights',
    'maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm','has_availability',
    'availability_30','availability_60','availability_90','availability_365','calendar_last_scraped',
    'number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d','review_scores_rating',
    'license','instant_bookable','host_verifications','bedrooms','room_type','calculated_host_listings_count',
    'calculated_host_listings_count_shared_rooms','calculated_host_listings_count_private_rooms','bathroom',
    'label_ubication','calculated_host_listings_count_entire_homes','host_listings_count','latitude','geo'    ,'longitude']
    for i in drop:
        df = df.drop(i, axis=1)

    df = df.fillna(0)

    df.host_is_superhost = df.host_is_superhost.map(dict(t=1, f=0))


        
    return df

def export(nombre, nombre_modelo):
    print("exporting...")
    predict = h2o.as_list(nombre)
    predict.to_csv(("data/{}.csv".format(nombre)))#Exportamos a CSV
    sample = pd.read_csv('data/sample.csv') 
    sample.price = predict.predict  #Cambiamos columna price por la Series de Pandas que tenemos
    sample.to_csv('data/modelo{}_predict.csv'.format(nombre_modelo), index = False)
    print ('Export done!' )