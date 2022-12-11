import duckdb
import pandas as pd
import json
import shutil
import os
import gzip
import numpy as np

names = ["AirBnB_WashDC/11Jun22 neighbourhoods.geojson", "AirBnB_WashDC/14Sep22 neighbourhoods.geojson", "AirBnB_WashDC/15Dec21 neighbourhoods.geojson", "AirBnB_WashDC/19Mar22 neighbourhoods.geojson"]

class DatawareHouse:
    def __init__(self) -> None:
        self.con = duckdb.connect(database='ps6.duckdb', read_only=False)
    
    def createAllListings(self):
        self.con.execute('DROP TABLE IF EXISTS all_listings;')
        self.con.execute("CREATE TABLE all_listings AS SELECT * FROM read_csv_auto('AirBnB_WashDC/14Sep22 listings 2.csv');")
        self.con.execute("ALTER TABLE all_listings DROP source;")
        self.con.execute("INSERT INTO all_listings SELECT * FROM read_csv_auto('AirBnB_WashDC/11Jun22 listings 2.csv');")
        self.con.execute("INSERT INTO all_listings SELECT * FROM read_csv_auto('AirBnB_WashDC/15Dec21 listings 2.csv');")
        self.con.execute("INSERT INTO all_listings SELECT * FROM read_csv_auto('AirBnB_WashDC/19Mar22 listings 2.csv');")

        all_listings_df = self.con.execute("SELECT * from all_listings").df()
        all_listings_df.price = all_listings_df.price.replace('\$|,', '', regex=True).astype(float)
        all_listings_df['host_acceptance_rate'] = all_listings_df['host_acceptance_rate'].replace('N/A',np.NaN)
        all_listings_df['host_acceptance_rate'] = all_listings_df['host_acceptance_rate'].replace('%', '', regex=True).astype(float)
        all_listings_df['host_response_rate'] = all_listings_df['host_response_rate'].replace('N/A',np.NaN)
        all_listings_df['host_response_rate'] = all_listings_df['host_response_rate'].replace('%', '', regex=True).astype(float)
        self.con.execute('DROP TABLE IF EXISTS all_listings;')
        self.con.execute("CREATE TABLE all_listings AS SELECT * FROM all_listings_df")
        
        # Create the view for the lastest listings
        self.con.execute('drop view if exists last_scraped')
        self.con.execute('create view last_scraped as select id, max(calendar_last_scraped) as last_scraped from all_listings group by id;')

        self.con.execute('drop view if exists latest_listings')
        self.con.execute('create view latest_listings as select all_listings.* from all_listings inner join last_scraped on all_listings.id = last_scraped.id and all_listings.calendar_last_scraped = last_scraped.last_scraped; ')
        print("all_listing tables have been created")

    def createNeighborhood(self):
        self.con.execute('DROP TABLE IF EXISTS neighborhoods;')
        self.con.execute('CREATE TABLE neighborhoods'
                    '(neighbourhood_group VARCHAR,'
                    'neighbourhood VARCHAR PRIMARY KEY,'
                    'type_outer VARCHAR,'
                    'feature_type VARCHAR,'
                    'geometry_type VARCHAR,'
                    'coordinates DOUBLE[][])')
        df = pd.read_csv("AirBnB_WashDC/11Jun22 neighbourhoods.csv")
        total = pd.DataFrame()
        last = pd.DataFrame(columns=["type_outer", "feature_type", "geometry_type", "coordinates", "neighborhood", "neighborhood group"])
        diff = False
        for n in names:
            with open(n, 'r') as f:
                data = json.load(f)
            
                row = []
                type_outer = ""
                type_outer = data["type"]
                features = []
                feature_type = ""
                geometry, properties = [], []
                for y in range(0, len(data["features"])):
                    features = data["features"][y]
                    feature_type = features["type"]
                    geometry = features["geometry"]
                    properties = features["properties"]
                    geometry_type, neigh, neigh_group = "", "", ""
                    coords = []
                    geometry_type = geometry["type"]
                    coords = geometry["coordinates"][0][0]
                    neigh = properties["neighbourhood"]
                    neigh_group = properties["neighbourhood_group"]
                    
                    if len(last) == 0:
                        row.append([type_outer, feature_type, geometry_type, coords, neigh, neigh_group])
                        diff = True
                        
                    elif list(last[last["neighborhood"] == neigh]["coordinates"])[0] != list(coords):
                        lastcoords = list(last[last["neighborhood"] == neigh]["coordinates"])[0]
                        diff = True
                        row.append([type_outer, feature_type, geometry_type, coords, neigh, neigh_group])
                
                if diff == True:
                    last = pd.DataFrame(row, columns=["type_outer", "feature_type", "geometry_type", "coordinates", "neighborhood", "neighborhood group"])
                    json_df = pd.DataFrame(row, columns=["type_outer", "feature_type", "geometry_type", "coordinates", "neighborhood", "neighborhood group"])
                    diff = False
                    
                    pd.concat([df, pd.read_csv(n.replace("geojson", "csv"))])
                    df.drop_duplicates()
                    
                    new_df = df.join(json_df.set_index("neighborhood"), on="neighbourhood")
                    
                    total = pd.concat([total, new_df])
        total = total.drop(['neighborhood group'], axis=1)
        self.con.execute("INSERT INTO neighborhoods SELECT * FROM total;")
        print("neighborhoods table has been created")


    def createReviews(self):
        self.con.execute('DROP TABLE IF EXISTS reviews;')
        self.con.execute('CREATE TABLE reviews'
                    '(listing_id BIGINT,'
                    'id BIGINT PRIMARY KEY,'
                    'date DATE,'
                    'reviewer_id INTEGER,'
                    'reviewer_name VARCHAR,'
                    'comments VARCHAR)')
        reviews_df = pd.read_csv('AirBnB_WashDC/14Sep22 reviews 2.csv')
        pd.concat([reviews_df, pd.read_csv('AirBnB_WashDC/11Jun22 reviews 2.csv')])
        pd.concat([reviews_df, pd.read_csv('AirBnB_WashDC/15Dec21 reviews 2.csv')])
        pd.concat([reviews_df, pd.read_csv('AirBnB_WashDC/19Mar22 reviews 2.csv')])
        reviews_df.drop_duplicates(subset=['id'])
        self.con.execute("INSERT INTO reviews SELECT * FROM reviews_df;")
    
    def createCalendar(self):
        self.unzipCalendar('AirBnB_WashDC/15Dec21 calendar.csv.gz', "12_2021")
        self.unzipCalendar('AirBnB_WashDC/19Mar22 calendar.csv.gz', "03_2022")
        self.unzipCalendar('AirBnB_WashDC/11Jun22 calendar.csv.gz', "06_2022")
        self.unzipCalendar('AirBnB_WashDC/14Sep22 calendar.csv.gz', "09_2022")

        self.cleancsv('calendar12_2021.csv','$')
        self.cleancsv('calendar03_2022.csv','$')
        self.cleancsv('calendar06_2022.csv','$')
        self.cleancsv('calendar09_2022.csv','$')

        self.con.execute('CREATE OR REPLACE TABLE calendar'
            '(listing_id BIGINT,'
            'date DATE,'
            'available VARCHAR,'
            'price INTEGER,'
            'adjusted_price INTEGER,'
            'minimum_nights INTEGER,'
            'maximum_nights INTEGER);')
        self.con.execute("COPY calendar FROM 'AirBnB_WashDC/calendar12_2021.csv' (AUTO_DETECT TRUE);")
        self.con.execute("INSERT INTO calendar SELECT * FROM read_csv('AirBnB_WashDC/calendar09_2022.csv', delim=',', header=True, columns={'listing_id': 'BIGINT', 'date': 'DATE', 'available': 'VARCHAR', 'price': 'INTEGER', 'adjusted_price': 'INTEGER', 'minimum_nights': 'INTEGER', 'maximum_nights': 'INTEGER'});")
        self.con.execute("INSERT INTO calendar SELECT * FROM read_csv_auto('AirBnB_WashDC/calendar03_2022.csv');")
        self.con.execute("INSERT INTO calendar SELECT * FROM read_csv_auto('AirBnB_WashDC/calendar06_2022.csv');")
        print("calendar tables have been created")

    def createCrime(self):
        self.con.execute('DROP TABLE IF EXISTS crimes;')
        self.con.execute("CREATE TABLE crimes as SELECT * FROM read_csv('AirBnB_WashDC/dc-crimes-search-results-V3.csv', dateformat='%m/%d/%Y, %H:%M:%S %p', AUTO_DETECT=TRUE);")

        self.con.execute('drop view if exists neighborhood_crimes')
        self.con.execute('create view neighborhood_crimes as Select neighborhood_name, '
                'count(neighborhood_name) FILTER (WHERE offense = \'homicide\') as homicides, '
                'count(neighborhood_name) FILTER (WHERE offense = \'robbery\') as robberies, '
                'count(neighborhood_name) FILTER (WHERE offense = \'assault w/dangerous weapon\') as assaults, '
                'count(neighborhood_name) FILTER (WHERE offense = \'theft f/auto\') as theft_from_auto, '
                'count(neighborhood_name) FILTER (WHERE offense = \'theft/other\') as other_thefts, '
                'count(neighborhood_name) FILTER (WHERE offense = \'motor vehicle theft\') as vehicle_theft, '
                'count(neighborhood_name) FILTER (WHERE offense = \'burglary\') as burglaries, '
                'count(neighborhood_name) FILTER (WHERE offense = \'sex abuse\') as sex_abuses, '
                'count(neighborhood_name) FILTER (WHERE offense = \'arson\') as arsons,'
                'count(neighborhood_name) as total_crimes '
                'from crimes group by neighborhood_name;')

        self.con.execute('drop view if exists neighborhood_crimes')
        self.con.execute('create view neighborhood_crimes as Select neighborhood_name, '
                'count(neighborhood_name) FILTER (WHERE offense = \'homicide\') as homicides, '
                'count(neighborhood_name) FILTER (WHERE offense = \'robbery\') as robberies, '
                'count(neighborhood_name) FILTER (WHERE offense = \'assault w/dangerous weapon\') as assaults, '
                'count(neighborhood_name) FILTER (WHERE offense = \'theft f/auto\') as theft_from_auto, '
                'count(neighborhood_name) FILTER (WHERE offense = \'theft/other\') as other_thefts, '
                'count(neighborhood_name) FILTER (WHERE offense = \'motor vehicle theft\') as vehicle_theft, '
                'count(neighborhood_name) FILTER (WHERE offense = \'burglary\') as burglaries, '
                'count(neighborhood_name) FILTER (WHERE offense = \'sex abuse\') as sex_abuses, '
                'count(neighborhood_name) FILTER (WHERE offense = \'arson\') as arsons,'
                'count(neighborhood_name) FILTER (WHERE offensegroup = \'violent\') as violent_crimes,'
                'count(neighborhood_name) FILTER (WHERE offensegroup = \'property\') as property_crimes,'
                'count(neighborhood_name) as total_crimes '
                'from crimes group by neighborhood_name;')
        self.con.execute('drop view if exists full_latest_listings')
        self.con.execute('create view full_latest_listings as select * from latest_listings left join neighborhood_crimes on latest_listings.neighbourhood_cleansed = neighborhood_crimes.neighborhood_name;')
        print("crime tables have been created")

    def unzipCalendar(self, file, uniqueId):
        new_file = 'calendar' + uniqueId + '.csv'
        with gzip.open(file, 'r') as f_in, open(new_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(new_file)

    def cleanPrices(self, filename):
        df = pd.read_csv(filename)
        df['price'] = df['price'].str.replace(',', '').astype(float)
        df['adjusted_price'] = df['adjusted_price'].str.replace(',', '').astype(float)
        df.to_csv(filename, encoding='utf-8', index=False)
    
    def cleancsv(self, filename, match):
        original_file = filename
        temp_file = "temp.csv"
        
        string_to_delete = [match]
        with open(original_file, "r") as input:
            with open(temp_file, "w") as output:
                for line in input:
                    for word in string_to_delete:
                        line = line.replace(word, "")
                    output.write(line)
        os.replace('temp.csv', filename)
        self.cleanPrices(filename)
        print(filename, ' cleaned')


if __name__ == "__main__":
    datawarehouse = DatawareHouse()
    datawarehouse.createAllListings()
    datawarehouse.createNeighborhood()
    datawarehouse.createReviews()
    datawarehouse.createCalendar()
    datawarehouse.createCrime()