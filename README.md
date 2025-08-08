# Sprint 14 - Project: Used Car Value Prediction - Rusty Bargain

The project tests machine learning skills by creating a predictive model to estimate the market value of used cars.

##  Project Description

Rusty Bargain, a used car sales service, is developing an app that allows users to find out the market value of their vehicles. The task is to build a machine learning model that predicts the price of a car based on its characteristics.

### Model objectives:
- High **prediction quality**
- Good **prediction speed**
- **Efficiency in training time**

##  Project Structure

1. **Data Exploration**
- Loading and analyzing the dataset `/datasets/car_data.csv`
   - Data cleaning and preprocessing

2. **Model training**
   - Linear regression (sanity check)
   - Decision tree
   - Random forest
   - LightGBM (with hyperparameter tuning)
   - CatBoost and XGBoost (optional)

3. **Evaluation**
   - Metric used: **RECM (Root Mean Squared Error)**
   - Comparison of prediction quality and speed
   - Training time analysis

4. **Categorical Feature Encoding**
   - Appropriate encoding for each algorithm
   - OHE for XGBoost, native encoding for LightGBM and CatBoost

5. **Optimization**
   - Hyperparameter tuning
   - Evaluation of variables to avoid failures
   - Measurement of cell execution time

---

##  Dataset

**File:** `car_data.csv`

| Column           | Description |
|-------------------|-------------|
| DateCrawled       | Profile download date |
| VehicleType       | Body type |
| RegistrationYear  | Year of registration |
| Gearbox           | Gearbox type |
| Power             | Power (HP) |
| Model             | Vehicle model |
| Mileage           | Mileage (km) |
| RegistrationMonth | Month of registration |
| FuelType          | Fuel type |
| Brand             | Vehicle brand |
| NotRepaired       | Repair status |
| DateCreated       | Profile creation date |
| NumberOfPictures  | Number of photos |
| PostalCode        | Postal code |
| LastSeen          | Last user activity |
| **Price**         | **Target: Price (in euros)** |

---

##  Checklist

- Compliance with instructions
- Data preparation and cleaning
- Model variety and tuning
- Code organization and clarity
- Project structure
- Findings and conclusions

---

##  Tools

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Sklearn
- Lightgbm
- Catboost
- Xgboost
- Time
