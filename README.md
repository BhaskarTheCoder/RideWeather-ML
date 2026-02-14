# RideWeather-ML

**Course:** CSCE 5215 Machine Learning Project  
**Semester:** Fall 2023

## 📋 Project Overview

This project investigates how weather conditions impact ride-hailing services (Uber and Lyft) in Boston, Massachusetts. Using machine learning techniques, we analyze the relationship between various weather parameters and ride service characteristics to predict cab types and understand pricing patterns.

## 🎯 Objectives

- Analyze the correlation between weather conditions and ride service patterns
- Build predictive models to classify cab types (Uber vs Lyft) based on various features
- Evaluate the impact of weather parameters on ride pricing and availability
- Compare multiple machine learning algorithms for optimal performance

## 📊 Dataset

The project uses two primary datasets from Kaggle:

1. **Ride Data (`cab_rides.csv`)**: Contains 693,071 ride records with features including:
   - Distance, cab type, timestamp
   - Source and destination locations
   - Price and surge multiplier
   - Product ID and ride name

2. **Weather Data (`weather.csv`)**: Contains 6,276 weather records with features including:
   - Temperature, humidity, wind speed
   - Cloud cover, atmospheric pressure
   - Rain measurements
   - Location and timestamp

**Data Source:** [Uber & Lyft Cab Prices Dataset](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices)

## 🔧 Technologies Used

- **Python 3.x**
- **Libraries:**
  - pandas - Data manipulation and analysis
  - numpy - Numerical computations
  - matplotlib & seaborn - Data visualization
  - scikit-learn - Machine learning algorithms and metrics
  - datetime - Time series handling

## 🚀 Features

### Data Preprocessing
- Timestamp conversion to human-readable format
- Feature engineering (date, time, weekday extraction)
- Data merging based on location and timestamp
- Handling missing values
- Feature scaling and normalization

### Exploratory Data Analysis
- Correlation analysis between features
- Distribution analysis of ride prices
- Weather pattern visualization
- Temporal analysis of ride patterns

### Machine Learning Models

#### 1. **Logistic Regression**
- Binary classification (Uber vs Lyft)
- Performance metrics evaluation
- Confusion matrix analysis

#### 2. **Support Vector Machine (SVM)**
- Non-linear classification
- Kernel-based approach
- Model performance comparison

#### 3. **K-Nearest Neighbors (KNN)**
- Instance-based learning
- Hyperparameter tuning using GridSearchCV
- Optimal parameter selection:
  - n_neighbors: 148
  - weights: 'distance'
  - metric: 'manhattan'

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (Before Merge) | 52.62% | 52.33% | 37.89% | 51.64% |
| KNN (Before Merge) | 56.56% | 56.57% | 55.26% | 56.56% |
| **KNN (After Tuning)** | **67.45%** | **67.46%** | **62.37%** | **67.36%** |

## 📈 Key Findings

1. **Weather Impact**: Weather conditions show measurable correlation with ride patterns and pricing
2. **Model Performance**: KNN with hyperparameter tuning achieved the best performance (67.45% accuracy)
3. **Feature Importance**: Distance, time of day, and weather conditions are significant predictors
4. **Optimal Configuration**: Manhattan distance metric with distance-based weighting performs best for this dataset

## 📁 Project Structure

```
├── ML_Final_Project.ipynb    # Main Jupyter notebook with complete analysis
├── ML_Final_Project.pdf       # Project documentation (PDF)
├── ML_Final_Project.pptx      # Project presentation
├── README.md                  # Project documentation
└── Images/
    ├── Results/               # Model performance visualizations
    └── Visualization/         # EDA and data visualization plots
```

## 🔬 Methodology

1. **Data Collection & Cleaning**
   - Load ride and weather datasets
   - Handle missing values
   - Convert timestamps to datetime format

2. **Feature Engineering**
   - Extract temporal features (date, time, weekday)
   - Create location-based features
   - Merge datasets on common attributes

3. **Exploratory Data Analysis**
   - Statistical analysis of features
   - Correlation heatmaps
   - Distribution plots
   - Time series analysis

4. **Model Development**
   - Train-test split (70-30)
   - Model training with multiple algorithms
   - Hyperparameter tuning using GridSearchCV
   - Cross-validation (5-fold)

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix analysis
   - Model comparison

## 📊 Visualizations

The project includes comprehensive visualizations:
- Correlation heatmaps
- Price distribution plots
- Weather pattern analysis
- Confusion matrices for model evaluation
- Time series plots
- Feature importance charts

## 🎓 Results & Conclusions

- Successfully demonstrated the impact of weather conditions on ride-hailing services
- KNN with optimized hyperparameters achieved the best classification performance
- Weather features contribute significantly to predicting cab type and pricing
- The merged dataset (combining ride and weather data) improved model performance
- Hyperparameter tuning significantly enhanced model accuracy (from 56.56% to 67.45%)

## 🔮 Future Work

- Incorporate additional features (traffic data, events, holidays)
- Explore deep learning models for improved accuracy
- Real-time prediction system implementation
- Extended geographical coverage
- Seasonal pattern analysis
- Dynamic pricing prediction

## 👥 Contributors

This project was completed as part of the CSCE 5215 Machine Learning course at the University of North Texas.

## 📝 License

This project is available for educational purposes.

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- University of North Texas - Department of Computer Science and Engineering
- Course Instructor and Teaching Assistants

---

**Note:** This project demonstrates the application of machine learning techniques to real-world data analysis, focusing on the intersection of weather patterns and ride-hailing services.
