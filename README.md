# NYC Taxi Trip Duration Prediction

This project builds a machine learning system which predicts **taxi trip duration in New York City**, using real-world trip data from the NYC Taxi and Limousine Commission (TLC) and integrated **historical hourly weather data** such as precipitation, temperature, and weather conditions using Open-Meteo API.

Such predictive systems are widely used by ride-hailing platforms like Uber, Lyft, and Veezu to improve **ETA accuracy**, support **driver dispatching**, and enhance **passenger experience**.

## Project Details

### Problem
Passengers and drivers often don’t know how long a taxi trip will take. Duration is influenced by a mix of factors: **distance**, **traffic**, **time of day**, and **weather conditions**, and failing to account for these leads to inaccurate ETAs.

### Objectives
The objective of this project is to:
- Predict trip duration based on:
  - **Pickup and dropoff time**
  - **Trip distance**
  - **Weather conditions** (temperature, rain, snow, etc.)
  - **Time-based features** (rush hour, day of week, month)
- Extract and use features like:
  - Peak hours during the day
  - Busy days in a week
- Evaluate the impact of **external weather data** on model performance.

## Data Sources

- **NYC Yellow Taxi Trip Data**  
  Provided by technology vendors under the TPEP/LPEP programs and collected by NYC TLC.  
  - [Microsoft Learn – NYC Yellow Taxi Dataset](https://learn.microsoft.com/en-gb/azure/open-datasets/dataset-taxi-yellow?tabs=azureml-opendatasets)  
  - [NYC TLC Official Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- **Open-Meteo Historical Weather Data**  
  Hourly weather data retrieved from Open-Meteo API and matched to yellow taxi trip records based on pickup date and hour. Features include:
  - `temperature`
  - `precipitation`
  - `weather_code`
 
## Methodology (to be updated)
### Feature Selection
#### Relevant Attributes

| **Feature**                           | **Reason for Inclusion**                                                              | **Planned Transformations**                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Pickup & Drop-off Time**            | Captures temporal patterns: peak hours, weekdays vs weekends, and seasonal variations | - Extract **hour** (0–23)<br>- Extract **day of week** (0–6) |
| **Trip Distance**                     | Core determinant of trip duration                                                     | - Use as-is or apply **min-max scaling**                                                   |
| **Passenger Count** *(optional)*      | May affect vehicle performance or indicate shared rides                               | - Include as-is initially<br>- Evaluate **feature importance**                             |
| **Extra (Rush Hour Flag)**            | Direct indicator of rush-hour periods (\$1 surcharge)                                 | - Convert to **binary** (1 = Rush Hour, 0 = Not)<br>- May merge with peak time             |
| **Congestion Surcharge** *(optional)* | Possible proxy for traffic conditions or high-demand zones                            | - Use as-is<br>- Consider merging with **peak hour indicators**                            |

#### Features To Drop

**RatecodeID**: Mostly redundant unless you're distinguishing airport rides.

**Payment Type**: Not related to trip duration.

**Tolls Amount**: More related to fare prediction than duration.

**Mta_tax**: More related to fare prediction than duration.

**Tip_amount**: More related to fare prediction than duration.

**Total_amount**: More related to fare prediction than duration.

**Airport_fee**: Not related to trip duration.

**Store_and_fwd_flag**: Not related to trip duration.

**PULocationID**, **DOLocationID**: Just knowing the taxi zones doesn’t account for traffic conditions, road closures, or actual route taken. Additionally, it may introduce noise, since for example, two trips starting from the same pickup and drop-off zone might have very different durations depending on route choice. However this is kept to create the 'route' feature.

## Correlation Matrix
![image](https://github.com/user-attachments/assets/fa4258df-a86e-49ae-8018-907eb66f0384)

## Analysis on Peak Hours & Busy Days
The 'hour' and 'weekday' features are created via data transformation from the 'tpep_pickup_datetime' feature available from the dataset. The following graph shows the peak hours and days of the week for NYC Taxis, where the dataset ranges from 2002 to 2024.

![image](https://github.com/user-attachments/assets/20e74317-7b5e-4ff1-878a-2c25cc201c60)

![image](https://github.com/user-attachments/assets/ce42d61d-ba69-47f2-b2dd-c27ef4352e84)

# Results

After resolving feature scaling issues and integrating weather data, model MAEs improved significantly.

![image](https://github.com/user-attachments/assets/88241d16-b593-45f1-a0e3-70e3a87b3691)

Based on the figure above:
- **Random Forest** performs the best compared to other models, both with and without the integration of weather data.
- **Random Forest** and **XGBoost** improved by over **6%** with weather data. This indicates that weather contributes valuable predictive signal.
- **Simple Regression** showed marginal improvement (+0.5%), which may suggest limited ability to leverage weather.
- Surprisingly, the **Deep Learning model** slightly worsened (−1.06%), possibly due to overfitting, tabular limitations or tuning challenges.

These results affirm that **tree-based models effectively incorporate weather features** in short-trip taxi duration prediction, while simpler or less-optimised models struggle to extract signal from added complexity.

# Discussions

During the data cleaning and transformation process, I stumbled across two key issues which taught me the impacts of capping outliers.

![image](https://github.com/user-attachments/assets/64125d8c-3241-4861-a3ed-5b28758fc15c)

The observed pattern is a result of capping outliers in the 'trip_distance' feature during data processing. However, instead of capping these outliers, it would be more effective to remove them. This is because capped outliers could reduce the model's accuracy, particularly for predicting trip durations near the 6-7 mile range. The wide variation in trip durations within this range may introduce inconsistencies, making it harder for the model to learn meaningful patterns. Removing these outliers ensures a more reliable relationship between trip distance and trip duration.

![image](https://github.com/user-attachments/assets/1a31cab2-a7e0-43ab-a2f8-3ba0f03e9a8a)

A distinct cluster of data points near 0 miles exhibits significant variation in trip durations. Upon further analysis, it was found that these data points correspond to instances where the trip distance is recorded as 0, which is invalid. These entries must be removed to prevent introducing noise and skewing the model's predictions.

After removing outliers instead of capping them, and manually removing invalid data points and additional outliers,the model performance increased by 3.3%. For more details, please refer to the Jupyter Notebook. :)

# Update Logs

## 26/2 Update

- Added new feature 'congestion_index' (further analysis on its importance will be conducted)

- Built and tested Simple Regression model (r2-score: 0.69)

- Built and tested Random Forest Regression (r2-score: 0.75)

- Built and tested XGBoost Regression (r2-score:0.7615)

- Utilised cross-validation and hyperparameter tuning for RF and XGBoost models.

- **Model performance increased by 8.86% (based on r2_score) using XGBoost compared to Simple Regression.**

- **Limitation** found: Max CPU usage and took a long time to compute

- **Solution** to limitation: Used **RandomizedSearch** instead of **GridSearch**, and used **sampling** of **10**% for Hyperparameter Tuning, and reduced the number of hyperparameter values.

## 24/3 Update
The Mean Absolute Error (MAE) was used to evaluate the performance of various regression models in predicting the trip duration. Though Mean Squared Error (MSE) puts higher penalty on large errors, MAE allows for a simpler and easier interpretation and reporting. Hence, the results are as follows:

- Simple Regression MAE: 32.37 minutes
- Random Forest MAE: 28.09 minutes
- XGBoost MAE: 27.89 minutes

Based on the results, it is clear that XGBoost and Random Forest outperform Simple Regression, with XGBoost being the best-performing model in this case. Despite this improvement, the prediction errors are still relatively high, which indicates that the models are not fully capturing the complexity of the relationships in the data.

Thus, the predictions are highly error-prone and the models are struggling to account for the underlying patterns. This suggests that the data may contain more complex relationships, which the simpler models like Simple Regression are unable to capture effectively.

To address the high prediction errors, I decided to explore a **Deep Neural Network (DNN)** as an alternative modeling approach. Given that the existing models (Simple Regression, Random Forest, and XGBoost) have shown limitations, it is reasonable to believe that DNNs, with their ability to model intricate relationships through multiple layers of neurons, could offer a more robust and flexible solution.

- Details:
-     Sample Size: 3,000
-     Epochs: 100
-     Batch size: 32
-     Layers: 4
-     Included Dropout layers

- Obtained Mean MAE: 4.1602 (+/- 0.1572) minutes

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
- Simple Regression MAE: 34.09 minutes (slightly worse)
- Random Forest MAE: 26.98 minutes (improved)
- XGBoost MAE: 27.81 minutes (minutely improved)
- DNN MAE: 3.95 minutes (improved by 5.3%)

## Update 3: 18/6
- Integrated real historical weather data using Open-Meteo API
- Scaled 'trip_duration_in_mins' and 'trip_distance'

### Impact of Weather Data on Model Performance

To evaluate the contribution of real-world weather data, I trained all models both **with** and **without** weather features (e.g., temperature, precipitation, weather_code).

**Results**:

- Without Weather Features:

  - Simple Regression: 0.5801
  - Random Forest: 0.494
  - XGBoost: 0.514
  - Deep Learning model Mean MAE: 0.568

- With Weather Features:
  
  - Simple Regression: 0.577 (+0.54%)
  - Random Forest: 0.465 (+6.24%)
  - XGBoost: 0.484 (+6.20%)
  - Deep Learning model Mean MAE: 0.574 (+1.06%)
