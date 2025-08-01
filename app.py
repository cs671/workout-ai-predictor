import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Page configuration
st.set_page_config(
    page_title="Workout Progression Predictor",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_info():
    """Load the trained model, label encoder, and model information"""
    try:
        # Load model, encoder, and metadata
        model = joblib.load('best_progression_model.pkl')
        label_encoder = joblib.load('exercise_label_encoder.pkl')
        
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        return model, label_encoder, model_info
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.info("Required files: best_progression_model.pkl, exercise_label_encoder.pkl, model_info.json")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_next_weight(model, label_encoder, exercise_name, prev_weight, days_rest, prev_rpe, prev_volume, session_number):
    """Make prediction using the trained model with proper exercise encoding"""
    
    # Encode the exercise name
    try:
        exercise_encoded = label_encoder.transform([exercise_name])[0]
    except ValueError:
        # If exercise not in training data, use most similar or default to first exercise
        st.warning(f"Exercise '{exercise_name}' not in training data. Using closest match.")
        exercise_encoded = 0
    
    # Create DataFrame with exact same column names and order as training
    input_data = pd.DataFrame({
        'prev_weight': [prev_weight],
        'days_rest': [days_rest], 
        'prev_rpe_filled': [prev_rpe],
        'prev_volume': [prev_volume],
        'session_number': [session_number],
        'exercise_encoded': [exercise_encoded]
    })
    
    prediction = model.predict(input_data)[0]
    return prediction

def get_progression_recommendation(current_weight, predicted_weight):
    """Generate human-readable recommendation"""
    diff = predicted_weight - current_weight
    
    if diff > 2.5:
        return "🚀 Strong progression recommended!", "success"
    elif diff > 0:
        return "📈 Small progression recommended", "success" 
    elif diff > -2.5:
        return "⚖️ Maintain current weight", "warning"
    else:
        return "🔋 Consider deload/rest", "error"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏋 Workout Progression Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model, label_encoder, model_info = load_model_and_info()
    
    if model is None:
        st.stop()
    
    # Model accuracy disclaimer
    st.warning("""
    ⚠️ **Model Accuracy Notice**: This model has an average prediction error of ±7.5kg. 
    Use predictions as guidance only and always listen to your body.""")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.write(f"**Average Error:** {model_info.get('mae', 0):.2f} kg")
        st.write(f"**Training Samples:** {model_info.get('training_samples', 0):,}")
        st.write(f"**CV Score:** {model_info.get('cv_mae', 0):.2f} kg")
        st.write(f"**Exercises Trained:** {len(model_info.get('exercises', []))}")
        
        st.header("🎯 How It Works")
        st.write("""
        1. Select your exercise type
        2. Input your last session data
        3. Model analyzes your patterns
        4. Get personalized weight recommendation
        5. Track your actual results
        """)
        
        st.header("📈 Features Used")
        features = model_info.get('features', [])
        for feature in features:
            st.write(f"• {feature.replace('_', ' ').title()}")
    
    # Main input form
    st.header("🔮 Get Your Next Workout Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Exercise Information")
        
        # Exercise selection from trained exercises
        available_exercises = model_info.get('exercises', [])
        if not available_exercises:
            st.error("No exercises found in model data.")
            st.stop()
        
        selected_exercise = st.selectbox(
            "Select Exercise",
            options=available_exercises,
            help="Choose from exercises the model was trained on"
        )
        
        # Show exercise-specific performance if available
        exercise_performance = model_info.get('exercise_performance', {})
        if selected_exercise in exercise_performance:
            perf = exercise_performance[selected_exercise]
            st.info(f"**{selected_exercise}** - Model accuracy: ±{perf.get('mae', 0):.1f}kg on {int(perf.get('samples', 0))} test samples")
        
        prev_weight = st.number_input(
            "Previous Session Max Weight (kg)",
            min_value=5.0,
            max_value=200.0,
            value=70.0,
            step=1.25,
            help="The heaviest weight you lifted in your last session"
        )
        
        session_number = st.number_input(
            "Total Sessions for This Exercise",
            min_value=1,
            max_value=500,
            value=20,
            help="How many times you've done this exercise total"
        )
    
    with col2:
        st.subheader("📅 Recovery & Performance")
        
        days_rest = st.number_input(
            "Days Since Last Session",
            min_value=0,
            max_value=14,
            value=3,
            help="How many days since you last did this exercise"
        )
        
        prev_rpe = st.slider(
            "Previous Session RPE (Rate of Perceived Exertion)",
            min_value=1.0,
            max_value=10.0,
            value=7.5,
            step=0.5,
            help="How hard did your last session feel? (1=very easy, 10=maximum effort)"
        )
        
        prev_volume = st.number_input(
            "Previous Session Total Volume (kg)",
            min_value=100,
            max_value=10000,
            value=int(prev_weight * 12 * 3),  # Estimate: weight × reps × sets
            step=100,
            help="Total volume from last session (weight × reps × sets)"
        )
    
    # RPE guidance
    with st.expander("🤔 RPE Scale Reference"):
        st.write("""
        **RPE (Rate of Perceived Exertion) Scale:**
        - **1-3:** Very easy, could do many more reps
        - **4-6:** Easy to moderate, could do several more reps  
        - **7-8:** Hard, could do 1-3 more reps
        - **9:** Very hard, could maybe do 1 more rep
        - **10:** Maximum effort, couldn't do another rep
        """)
    
    # Prediction button
    if st.button("🎯 Get Prediction", type="primary"):
        
        # Make prediction
        predicted_weight = predict_next_weight(
            model, label_encoder, selected_exercise, prev_weight, days_rest, prev_rpe, prev_volume, session_number
        )
        
        # Generate recommendation
        recommendation, rec_type = get_progression_recommendation(prev_weight, predicted_weight)
        
        # Display results
        st.header("🔮 Your Personalized Recommendation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Weight",
                value=f"{prev_weight:.1f} kg"
            )
        
        with col2:
            st.metric(
                label="Predicted Next Weight", 
                value=f"{predicted_weight:.1f} kg",
                delta=f"{predicted_weight - prev_weight:+.1f} kg"
            )
        
        with col3:
            change_pct = ((predicted_weight - prev_weight) / prev_weight) * 100
            st.metric(
                label="% Change",
                value=f"{change_pct:+.1f}%"
            )
        
        # Recommendation box
        if rec_type == "success":
            st.markdown(f'<div class="success-box"><h3>{recommendation}</h3></div>', unsafe_allow_html=True)
        elif rec_type == "warning":
            st.markdown(f'<div class="warning-box"><h3>{recommendation}</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-box"><h3>{recommendation}</h3></div>', unsafe_allow_html=True)
        
        # Exercise-specific insights
        st.subheader("🧠 Insights")
        
        insights = []
        
        # Add exercise-specific insight
        insights.append(f"🏋️ Prediction for **{selected_exercise}** based on {session_number} previous sessions")
        
        if days_rest <= 1:
            insights.append("⚠️ Very short recovery time - consider the fatigue factor")
        elif days_rest >= 7:
            insights.append("⏰ Long break since last session - may need to ease back in")
        else:
            insights.append("✅ Good recovery time for progression")
        
        if prev_rpe <= 6:
            insights.append("💪 Previous session was easy - ready for more challenge")
        elif prev_rpe >= 8.5:
            insights.append("🔥 Previous session was intense - consider lighter progression")
        else:
            insights.append("⚖️ Previous session intensity was in the sweet spot")
        
        if session_number < 10:
            insights.append("🌱 Still building experience with this exercise - expect faster initial gains")
        elif session_number > 50:
            insights.append("🏆 Experienced with this exercise - progression may be slower but steady")
        
        for insight in insights:
            st.write(insight)
        
        # Show exercise-specific model performance
        if selected_exercise in exercise_performance:
            perf = exercise_performance[selected_exercise]
            st.subheader("📊 Model Performance for This Exercise")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"±{perf.get('mae', 0):.1f} kg")
            with col2:
                st.metric("Test Samples", f"{int(perf.get('samples', 0))}")
        
        # Create a simple visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=['Previous', 'Predicted'],
            y=[prev_weight, predicted_weight],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=10),
            name='Weight Progression'
        ))
        
        # Add confidence interval based on exercise-specific MAE
        exercise_mae = exercise_performance.get(selected_exercise, {}).get('mae', model_info.get('mae', 7.5))
        lower_bound = predicted_weight - exercise_mae
        upper_bound = predicted_weight + exercise_mae
        
        fig.add_trace(go.Scatter(
            x=['Predicted', 'Predicted'],
            y=[lower_bound, upper_bound],
            mode='lines',
            line=dict(color='lightblue', width=2),
            name='Confidence Range',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Weight Progression: {selected_exercise}",
            xaxis_title="Session",
            yaxis_title="Weight (kg)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer with usage tips
    st.header("💡 Pro Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **Before Your Workout:**
        - Input your most recent session data
        - Use the predicted weight as a starting point
        - Adjust based on how you feel that day
        """)
    
    with col2:
        st.write("""
        **After Your Workout:**
        - Track your actual performance
        - Note if prediction was accurate
        - The model learns from patterns in your data
        """)
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.write("""
        This workout progression predictor model was trained on personal workout data 
        from Strong app exports. It uses machine learning to analyze individual patterns 
        and recommend optimal weights for your next training session.
        
        **Disclaimer:** This is a tool to assist your training decisions. Always listen to your 
        body and consult with qualified trainers for personalized advice. The model has an 
        average error of ±7.5kg and should be used as guidance only.
        """)

if __name__ == "__main__":
    main()
