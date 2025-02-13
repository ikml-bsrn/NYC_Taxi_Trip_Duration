# NYC Taxi Trip Duration Estimator

This project aims to create a machine learning model which predicts the estimated trip duration for taxis in New York using relevant features from a dataset by NYC TLC. This system is widely used in many ride-hailing companies such as Uber and Veezu, and has been useful for passengers and drivers in many circumstances.

This project serves as a passion project, where novelty is not particularly emphasised, but serves as a way to gain experience and apply my knowledge in practical context using real-world data.
Project Details
Problem: Users don’t usually know when exactly they will arrive at their destination point in taxis.

**Objectives**: 
To predict how long a trip will take based on pickup/dropoff time, peak hours, and distance.
To determine the peak times in a day and busy days in a week, which will be used as a Feature in the model.

**Data**: The data used in the attached datasets were collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorised under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). This dataset is an open dataset, originally found in the Microsoft Learn's Open Datasets page but obtained from NYC Taxi & Limousine Commission (TLC) (refer to link below):
Microsoft Learn - https://learn.microsoft.com/en-gb/azure/open-datasets/dataset-taxi-yellow?tabs=azureml-opendatasets 
NYC TLC - https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page 

**Analysis**:
Feature engineering: Create a new Feature called ‘Peak Hours’
More soon… :)

**Results**: More soon… :)

## Dataset Details

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

## Initial Analysis on Features
### Feature Selection
#### Relevant Attributes

##### Pickup & Drop Off Time:
Why? Peak hours, day of the week, and seasonality affect trip duration.
Feature Engineering:
Extract hour of the day (0-23) → Helps identify rush hours.
Extract day of the week (0-6) → Weekends vs. weekdays.
Extract month/season → Weather and tourism seasons.

##### Trip Distance:
Why? Core factor in determining duration.
Feature Engineering:
Keep as-is or normalize using min-max scaling.


##### Passenger Count (Maybe)
Why? Could impact weight, vehicle performance, or likelihood of shared rides.
Feature Engineering:
Try including it as-is and analyze feature importance later.


##### Extra (Rush Hour Indicator)
Why? It explicitly marks rush-hour trips ($1 rush-hour charge).
Feature Engineering:
Binary feature: 1 = Rush Hour, Other = Non-Rush Hour.
Could be merged into the "Peak Time" feature.

##### Congestion Surcharge (Maybe)
Why? Might be correlated with congestion, affecting duration.
Feature Engineering:
Keep as-is and test importance.
Could be combined with peak time data to infer high-traffic zones.

#### Features To Drop

**RatecodeID**: Mostly redundant unless you're distinguishing airport rides.

**Payment Type**: Not related to trip duration.

**Tolls Amount**: More related to fare prediction than duration.

**Mta_tax**: More related to fare prediction than duration.

**Tip_amount**: More related to fare prediction than duration.

**Total_amount**: More related to fare prediction than duration.

**Airport_fee**: Not related to trip duration.

**Store_and_fwd_flag**: Not related to trip duration.

**PULocationID**, **DOLocationID**: Just knowing the taxi zones doesn’t account for traffic conditions, road closures, or actual route taken. Additionally, it may introduce noise, since for example, two trips starting from the same pickup and drop-off zone might have very different durations depending on route choice. 
But I will run feature importance analysis just to be sure.

More updates coming soon! :)

