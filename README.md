# Final Report
Problem Set 6 / Group Project  
EN.685.648.81.FA22 Data Science   
December 11, 2022  


# Group 1
* Sakib Ahmed
* David Arteche
* Tanner Amundsen
* Eva Jin
* Emily Russell
* Yisheng Tang


# Get Step
# Primary Dataset
All primary dataset files were downloaded from http://insideairbnb.com/get-the-data/ and stored in the AirBnB_WashDC/ subdirectory. Each compressed file was extracted to this same directory, with detailed listing files resulting in an appended " 2" in the filename due to having the same name as the summary listing sources (e.g. 11Jun22 listings 2.csv for the detailed listing and 11Jun22 listings.csv for the summary listing).


## Secondary Dataset ##

The secondary dataset we chose is the Crime data for Washington D.C. from 10/1/2021 to 10/1/2022. The data was originally from Metropolitan Police Department of Washington D.C. It's available to on [crimecards](https://crimecards.dc.gov/all:crimes/all:weapons/dated::10012021:10012022/citywide:heat). The dataset include many variables in it such as the type of the crime, the location where the crime happened, the start time and the end time for the crime. We also refer to the cluster id in the dataset and mapped to the neighborhood we have from the airbnb dataset. We believe there might be some correlation between the total crimes and the price of the airbnb listing.

A script was written to bring this data in as the file dc-crimes-search-results-V3.csv. This is also stored in the AirBnB_WashDC/ subdirectory

## datawarehouse.py
This script can be run by itself from the CLI, or from a notebook using  
%run -i 'datawarehouse.py'

# Ask
queries.ipynb includes a CoNVO and a CLD

# Explore
Single-variable and pairwise analysis was done for the listings data set and the crime data set.  

# Model
baseline.ipynb includes the baseline/null model and the distributional model  

regression.ipynb includes the regression model


# Report
The working notebooks were summarized in the report.ipynb notebook. This notebook can run on its own, from start to finish, with the data setup mentioned in the __Get Step__ above.


# Peer Review
* Sakib Ahmed     - 
* David Arteche   - 
* Tanner Amundsen - 
* Eva Jin         - 
* Emily Russell   - 
* Yisheng Tang    - 