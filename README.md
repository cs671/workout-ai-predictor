# 🏋️ Workout Progression Predictor

> An application that predicts optimal weights for your next workout session using machine learning on personal training data.

## 🎯 Project Overview

This project demonstrates an end-to-end machine learning solution that analyzes personal workout data to predict optimal weight progressions. Using data exported from the Strong fitness app, I built a Random Forest model that provides personalized weight recommendations with ±7.5kg average accuracy.

### 🔍 Key Features
- **Personalized Predictions**: Tailored weight recommendations for 8 different exercises
- **Real-Time Insights**: Interactive Streamlit dashboard with live predictions  
- **Smart Analysis**: Considers rest days, RPE, training volume, and exercise history
- **High Accuracy**: 7.48kg mean absolute error across all exercises
- **Production Ready**: Deployed web application with intuitive UI

## 📊 Business Impact

- **Injury Prevention**: Reduces risk of overtraining by recommending appropriate progressions
- **Training Optimization**: Maximizes strength gains through data-driven weight selection
- **Time Efficiency**: Eliminates guesswork in workout planning
- **Personalization**: Adapts to individual training patterns and recovery rates

## 🛠️ Technical Implementation

### Data Pipeline
```
Raw Strong App Data → Cleaning & Feature Engineering → Session Aggregation → ML Training → Deployment
```

### Machine Learning Approach
- **Algorithm**: Random Forest Regressor (200 estimators)
- **Features**: Previous weight, rest days, RPE, volume, session count, exercise type
- **Training Data**: 255 workout sessions across 8 exercises
- **Validation**: Stratified train-test split with exercise-specific evaluation

### Model Performance
| Exercise | MAE (kg) | Test Samples |
|----------|----------|--------------|
| Pec Deck | 2.69 | 4 |
| Lying Leg Curl | 5.19 | 5 |
| Chest Press | 4.93 | 7 |
| Preacher Curl | 7.09 | 7 |
| **Overall** | **7.48** | **255** |

## 🚀 Live Demo

**Try the app**: [(https://workout-progression-predictor.streamlit.app/)](https://workout-progression-predictor2.streamlit.app/)]

### Sample Prediction
Input your last session data and get instant recommendations:
- Exercise: Chest Press (Machine)
- Previous Weight: 70kg
- Days Rest: 3
- RPE: 7.5
- **Prediction**: 72.5kg (+2.5kg progression)

## 📁 Repository Structure
```
workout-progression-predictor/
├── app.py                          # Streamlit application
├── model_training.py               # Complete training pipeline
├── data/
│   └── strong.csv                  # Raw workout data
├── models/
│   ├── best_progression_model.pkl  # Trained Random Forest
│   ├── exercise_label_encoder.pkl  # Exercise encoder
│   └── model_info.json            # Model metadata
├── notebooks/
│   └── exploratory_analysis.ipynb # Data exploration
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## 🔧 Installation & Usage

### Quick Start
```bash
git clone https://github.com/cs671/workout-progression-predictor
cd workout-progression-predictor
pip install -r requirements.txt
streamlit run app.py
```

### Training Your Own Model
```python
# Load your Strong app export
python model_training.py --data_path your_strong_export.csv
```

## 📈 Key Insights

### Feature Importance Analysis
1. **Previous Weight** (46.8%) - Strongest predictor of next session weight
2. **Training Volume** (24.3%) - Higher volume indicates readiness for progression  
3. **Exercise Type** (9.1%) - Different exercises have unique progression patterns
4. **Session Number** (7.2%) - Experience level affects progression rate

### Training Patterns Discovered
- **Beginner Effect**: First 10 sessions show faster progression (avg +5kg/session)
- **Recovery Impact**: 3-4 day rest periods optimize progression vs. fatigue
- **RPE Sweet Spot**: Sessions with RPE 7-8 lead to best next-session performance

## 🎓 Skills Demonstrated

### Data Science
- **Data Preprocessing**: Cleaned and aggregated 4,000+ workout records
- **Feature Engineering**: Created lag features, session metrics, and temporal variables
- **Exploratory Analysis**: Uncovered training patterns and progression insights
- **Model Selection**: Compared Random Forest, XGBoost, and Linear models

### Machine Learning
- **Regression Modeling**: Optimized hyperparameters for minimal prediction error
- **Cross-Validation**: Stratified splits ensuring exercise representation
- **Model Evaluation**: Exercise-specific performance metrics and error analysis
- **Production ML**: Model serialization, versioning, and deployment

### Software Engineering  
- **Full-Stack Development**: End-to-end application from data to deployment
- **Web Development**: Interactive Streamlit dashboard with real-time predictions
- **Version Control**: Git workflow with feature branches and documentation
- **Cloud Deployment**: Production app hosting on Streamlit Cloud

## 🔮 Future Enhancements

- [ ] **Multi-Exercise Sessions**: Predict optimal weights for entire workouts
- [ ] **Injury Prevention**: Integrate biomechanical risk factors
- [ ] **Social Features**: Compare progress with similar athletes
- [ ] **Advanced ML**: Deep learning for complex progression patterns
- [ ] **Mobile App**: Native iOS/Android application

## 📚 Technologies Used

**Machine Learning**: scikit-learn, pandas, numpy  
**Visualization**: plotly, matplotlib, seaborn  
**Web App**: Streamlit, HTML/CSS  
**Development**: Python 3.11, Jupyter, Google Colab  
**Deployment**: Streamlit Cloud, GitHub Actions  
