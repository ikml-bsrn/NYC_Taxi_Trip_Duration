# NYC Taxi Trip Duration Prediction

This project aims to create a machine learning model which predicts the estimated trip duration for taxis in New York using relevant features from a dataset by NYC TLC. This system is widely used in many ride-hailing companies such as Uber and Veezu, and has been useful for passengers and drivers in many circumstances.

This project serves as a passion project, where novelty is not particularly emphasised, but serves as a way to gain experience and apply my knowledge in practical context using real-world data.

**Project Details**
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

**Results**: See section below.

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

## Correlation Matrix
![image](https://github.com/user-attachments/assets/4ae8836f-61a9-4818-9c5f-0771db9fc465)

## Simple Analysis on Peak Hours & Busy Days
The 'hour' and 'weekday' features are created via data transformation from the 'tpep_pickup_datetime' feature available from the dataset. The following graph shows the peak hours and days of the week for NYC Taxis, where the dataset ranges from 2002 to 2024.

![image](https://github.com/user-attachments/assets/20e74317-7b5e-4ff1-878a-2c25cc201c60)

![image](https://github.com/user-attachments/assets/ce42d61d-ba69-47f2-b2dd-c27ef4352e84)


# Discussions

![image](https://github.com/user-attachments/assets/64125d8c-3241-4861-a3ed-5b28758fc15c)

The observed pattern is a result of capping outliers in the 'trip_distance' feature during data processing. However, instead of capping these outliers, it would be more effective to remove them. This is because capped outliers could reduce the model's accuracy, particularly for predicting trip durations near the 6-7 mile range. The wide variation in trip durations within this range may introduce inconsistencies, making it harder for the model to learn meaningful patterns. Removing these outliers ensures a more reliable relationship between trip distance and trip duration.

![image](https://github.com/user-attachments/assets/1a31cab2-a7e0-43ab-a2f8-3ba0f03e9a8a)

A distinct cluster of data points near 0 miles exhibits significant variation in trip durations. Upon further analysis, it was found that these data points correspond to instances where the trip distance is recorded as 0, which is invalid. These entries must be removed to prevent introducing noise and skewing the model's predictions.

After removing outliers instead of capping them, and manually removing invalid data points and additional outliers,the model performance increased by 3.3%. For further discussions, please refer to the Jupyter Notebook. :)

# Results

## 26/2 Update

- Added new feature 'congestion_index' (further analysis on its importance will be conducted)

- Built and tested Simple Regression model (MSE: 0.69)

- Built and tested Random Forest Regression (r2-score: 0.75)

- Built and tested XGBoost Regression (r2-score:0.7615)

- Utilised cross-validation and hyperparameter tuning for RF and XGBoost models.

- **Model performance increased by 8.86% (based on r2_score) using XGBoost compared to Simple Regression.**

- **Limitation** found: Max CPU usage and took a long time to compute

- **Solution** to limitation: Used **RandomizedSearch** instead of **GridSearch**, and used **sampling** of **10**% for Hyperparameter Tuning, and reduced the number of hyperparameter values.

## 24/3 Update
The Mean Squared Error (MSE) was used to evaluate the performance of various regression models in predicting the trip duration. The results are as follows:

- Simple Regression MSE: 32.37 minutes
- Random Forest MSE: 28.09 minutes
- XGBoost MSE: 27.89 minutes

Based on the results, it is clear that XGBoost and Random Forest outperform Simple Regression, with XGBoost being the best-performing model in this case. Despite this improvement, the prediction errors are still relatively high, indicating that the models are not fully capturing the complexity of the relationships in the data.

Thus, the predictions are highly error-prone and the models are struggling to account for the underlying patterns. This suggests that the data may contain more complex relationships, which the simpler models like Simple Regression are unable to capture effectively.

To address the high prediction errors, I decided to explore a **Deep Neural Network (DNN)** as an alternative modeling approach. Given that the existing models (Simple Regression, Random Forest, and XGBoost) have shown limitations, it is reasonable to believe that DNNs, with their ability to model intricate relationships through multiple layers of neurons, could offer a more robust and flexible solution.

- Details:
-     Sample Size: 3,000
-     Epochs: 100
-     Batch size: 32
-     Layers: 4
-     Included Dropout layers

- Obtained Mean MAE: 4.1602 (+/- 0.1572)

Training & validation MAE
- ![image](https://github.com/user-attachments/assets/2505079e-ddf5-4cbc-876c-4b37fa8497fd)
Training & validation Loss
- ![image](https://github.com/user-attachments/assets/791fd4ae-3ed5-4940-9cb1-2aeb5413f8f6)



Based on the Mean MAE of the DNN model, it outperformed previous models significantly. This shows the ability of the model to tackle the complexity of the data and make more accurate predictions.

## Update 2: 24/3 
- Added new feature called ‘route’ (from ‘PULocationID’ and ‘DOLocationID’)
- ‘Route’ encoded to ‘route_freq’ using Frequency Encoding for non-DNN models
- ‘Route’ encoded to ‘route_encoded’ using LabelEncoding for DNN model

**Results**:
- Simple Regression MSE: 34.09 minutes (slightly worse)
- Random Forest MSE: 26.98 minutes (improved)
- XGBoost MSE: 27.81 minutes (minutely improved)
- DNN MSE: 3.95 minutes (improved by 5.3%)
