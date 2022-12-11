import pandas as pd

crime_data = pd.read_csv("dc-crimes-search-results.csv")
#print(crime_data)

neighborhood_data = pd.read_csv("Neighborhood_Clusters.csv")
#print(neighborhood_data)

neighborhood_map = {}
for index,row in neighborhood_data.iterrows():
    cluster = row['NAME'].lower()
    name = row['NBH_NAMES']
    neighborhood_map[cluster] = name
#print(neighborhood_map)

neighborhood = []
#crime_data["NEIGHBORHOOD_NAME"] = ""
for index,row in crime_data.iterrows():
    cluster = row["NEIGHBORHOOD_CLUSTER"]
    if cluster in neighborhood_map:
        print(neighborhood_map[cluster])
        neighborhood.append(neighborhood_map[cluster])
    else:
        neighborhood.append("N/A")
crime_data["NEIGHBORHOOD_NAME"] = neighborhood
print(crime_data)
crime_data.to_csv("new_crime.csv")
