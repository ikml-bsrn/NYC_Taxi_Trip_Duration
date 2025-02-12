# The Taxi Fares Predictor

This project aims to create a machine learning model which predicts fares from relevant attributes, a predictive model popularly and widely used by ride-hailing companies such as Uber, Veezu, and many more. This project serves as a passion project, where novelty is not particularly emphasised, but serves as a way for me to gain experience and apply my knowledge in practical context using real-world data.

In this project, attributes include, but not limited to:
- pick-up and drop-off dates/times
- pick-up and drop-off locations
- trip distances
- itemised fares
- rate types
- payment types
- driver-reported passenger counts
- tip amount
- airport fee

# Project Details

**Problem**:

**Research Questions**:

**Data**: The data used in the attached datasets were collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorised under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). This dataset is an open dataset, originally found in the Microsoft Learn's Open Datasets page but obtained from NYC Taxi & Limousine Commission (TLC) (refer to link below):

  Microsoft Learn - https://learn.microsoft.com/en-gb/azure/open-datasets/dataset-taxi-yellow?tabs=azureml-opendatasets

  NYC TLC - https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page


**Analysis**:

**Results**:

# Dataset Details

- VendorID: A code indicating the LPEP provider that provided the record. 
    1= Creative Mobile Technologies, LLC; 
    2= VeriFone Inc.

- tpep_pickup_datetime: The date and time when the meter was engaged.

- tpep_dropoff_datetime: The date and time when the meter was disengaged.

- passenger_count: The number of passengers in the vehicle. This is a driver-entered value.

- trip_distance: The elapsed trip distance in miles reported by the taximeter.

- RatecodeID: The final rate code in effect at the end of the trip. 
    1= Standard rate; 
    2= JFK; 
    3= Newark; 
    4= Nassau or Westchester; 
    5= Negotiated fare; 
    6= Group ride.

-  store_and_fwd_flag: This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, also known as “store and forward,” because the vehicle did not have a connection to the server. 
    Y= store and forward trip; 
    N= not a store and forward trip.

- PULocationID: TLC Taxi Zone in which the taximeter was engaged.  

- DOLocationID: TLC Taxi Zone in which the taximeter was disengaged.   

- payment_type: A numeric code signifying how the passenger paid for the trip. 
    1= Credit card; 
    2= Cash; 
    3= No charge; 
    4= Dispute; 
    5= Unknown; 
    6= Voided trip  

- fare_amount: The time-and-distance fare calculated by the meter.

- extra: Miscellaneous extras and surcharges. Currently, this only includes the $0.50 and $1 rush hour and overnight charges.

- mta_tax: $0.50 MTA tax that is automatically triggered based on the metered rate in use.

- tip_amount: This field is automatically populated for credit card tips. Cash tips are not included.

- tolls_amount: Total amount of all tolls paid in trip.

- improvement_surcharge: $0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015.

- total_amount: The total amount charged to passengers. Does not include cash tips.

- congestion_surcharge:

- Airport_fee
