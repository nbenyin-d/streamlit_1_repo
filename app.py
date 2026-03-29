"""
AGRICULTURAL DISEASE DIAGNOSIS CHATBOT - STREAMLIT VERSION
===========================================================

This is a complete, production-ready Streamlit application for crop disease diagnosis.

Features:
- Crop selection (Maize, Cassava, Tomato)
- Symptom description input
- AI-powered diagnosis using SetFit model
- Treatment recommendations from dataset
- Prevention measures
- Confidence scores
- Alternative diagnoses

Author: Agricultural AI Research Project
Date: 2024
"""

import streamlit as st
from setfit import SetFitModel
import pickle
import json
import numpy as np
import torch
import pandas as pd
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Crop Disease Diagnosis Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .treatment-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .prevention-box {
        background-color: #FFF3E0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFEBEE;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """
    Load SetFit model, label mappings, and recommendations.
    Cached to avoid reloading on every interaction.
    """
    try:
        # Load SetFit model
        model = SetFitModel.from_pretrained("Tree-Diagram/setfit_crop_disease_model_v1")
        
        # Load label mappings
        with open('label_mappings.pkl', 'rb') as f:
            label_info = pickle.load(f)
        
        # Load recommendations
        with open('diagnosis_recommendations.json', 'r') as f:
            recommendations = json.load(f)
        
        return model, label_info, recommendations, None
    
    except Exception as e:
        return None, None, None, str(e)

# Initialize
model, label_info, recommendations, error = load_model_and_data()

if error:
    st.error(f"""
    ⚠️ **Error loading model**: {error}
    
    Please ensure:
    1. Model folder 'setfit_crop_disease_model/' exists
    2. 'label_mappings.pkl' file exists
    3. 'diagnosis_recommendations.json' file exists
    """)
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🌾 Crop Disease Diagnosis Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-Powered Tool for Farmers • Instant Diagnosis • Evidence-Based Recommendations</p>',
    unsafe_allow_html=True
)

# ============================================================================
# SIDEBAR - CROP SELECTION & INFO
# ============================================================================

st.sidebar.header("🌱 Step 1: Select Your Crop")

crop = st.sidebar.selectbox(
    "Which crop are you diagnosing?",
    ["MAIZE", "CASSAVA", "TOMATO"],
    help="Choose the crop you're experiencing problems with"
)

# Display crop-specific icon
crop_icons = {
    "MAIZE": "🌽",
    "CASSAVA": "🥔", 
    "TOMATO": "🍅"
}

st.sidebar.markdown(f"### Selected: {crop_icons[crop]} {crop}")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Information")
st.sidebar.info(f"""
**Model Details:**
- Type: SetFit (Few-Shot Learning)
- Training: 8 samples per disease
- Total Diseases: {label_info['n_classes']}
- Supported Crops: 3

**Data Source:**
- Ghana Ministry of Agriculture
- Plant Clinic Records
- Expert-Validated Recommendations
""")

# Usage statistics (if you want to track)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
if 'diagnosis_count' not in st.session_state:
    st.session_state.diagnosis_count = 0
st.sidebar.metric("Diagnoses Today", st.session_state.diagnosis_count)

# ============================================================================
# MAIN INPUT AREA
# ============================================================================

st.markdown("---")
st.header(f"📝 Step 2: Describe Symptoms for {crop_icons[crop]} {crop}")

col1, col2 = st.columns([2, 1])

with col1:
    symptom_description = st.text_area(
        "**Symptom Description**",
        height=200,
        placeholder=f"Describe what you see on your {crop.lower()} plant...\n\nExample: The plant shows yellowing leaves, bore holes, visible insects, and frass on the leaves. Some leaves are chewed.",
        help="Provide as much detail as possible about what you observe"
    )
    
    # Character counter
    char_count = len(symptom_description)
    if char_count > 0:
        st.caption(f"Characters: {char_count} {'✓' if char_count >= 20 else '(minimum 20 recommended)'}")

with col2:
    st.markdown("### 💡 What to Include")
    st.markdown("""
    **Describe these if visible:**
    
    🔍 **Leaf Symptoms**
    - Color changes
    - Spots or lesions
    - Holes or damage
    
    🐛 **Pest Signs**
    - Insects present
    - Eggs or larvae
    - Frass or webbing
    
    🌿 **Plant Condition**
    - Wilting
    - Stunted growth
    - Deformed parts
    
    📍 **Distribution**
    - Isolated or widespread
    - Which plant parts affected
    """)

# Example descriptions
with st.expander("📋 Click here for example descriptions"):
    st.markdown("""
    **Example 1 - Fall Armyworm (Maize)**:
    > "The maize plant shows bore holes in the leaves with visible caterpillars. There is frass (insect droppings) on the whorl and chewed leaf edges. I can see the insects feeding during early morning."
    
    **Example 2 - Cassava Mosaic Virus**:
    > "The cassava leaves show yellow and green mosaic patterns. New leaves are distorted and smaller than normal. The plant growth is stunted."
    
    **Example 3 - Tomato Early Blight**:
    > "The tomato plant has dark brown spots on lower leaves with yellow halos around them. Older leaves are turning yellow and dying. Spots have concentric rings like a target."
    """)

# ============================================================================
# DIAGNOSIS BUTTON & PROCESSING
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    diagnose_button = st.button(
        "🔬 Diagnose Disease",
        type="primary",
        use_container_width=True
    )

if diagnose_button:
    # Validation
    if not symptom_description.strip():
        st.error("❌ Please enter a symptom description before diagnosing!")
        st.stop()
    
    if len(symptom_description.strip()) < 20:
        st.warning("⚠️ Your description is very short. Please provide more details for better accuracy.")
        st.stop()
    
    # Processing
    with st.spinner("🔍 Analyzing symptoms... Please wait."):
        try:
            # Make prediction
            prediction = model.predict([symptom_description])
            
            # Get probabilities
            try:
                probs = model.predict_proba([symptom_description])
                
                # Convert to numpy if tensor
                if torch.is_tensor(probs):
                    probs = probs.cpu().numpy()
                else:
                    probs = np.array(probs)
                
                # Handle shape
                if len(probs.shape) == 1:
                    # Single prediction, create one-hot
                    probs_2d = np.zeros((1, label_info['n_classes']))
                    probs_2d[0, int(prediction[0])] = 1.0
                    probs = probs_2d
                
                # Get top 3 predictions
                top_3_indices = np.argsort(-probs[0])[:3]
                top_3_probs = probs[0][top_3_indices]
                
            except Exception as e:
                # Fallback: use prediction only
                top_3_indices = [int(prediction[0])]
                top_3_probs = [1.0]
            
            # Get diagnosis names
            idx_to_diagnosis = label_info['idx_to_diagnosis']
            top_diagnoses = [
                (idx_to_diagnosis[int(idx)], float(prob)) 
                for idx, prob in zip(top_3_indices, top_3_probs)
            ]
            
            # Filter by selected crop if possible
            primary_diagnosis = top_diagnoses[0][0]
            confidence = top_diagnoses[0][1]
            
            # Update count
            st.session_state.diagnosis_count += 1
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            st.success("✅ Analysis Complete!")
            
            st.markdown("---")
            
            # Primary diagnosis
            st.markdown("## 🎯 Diagnosis Results")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f'<div class="diagnosis-box">', unsafe_allow_html=True)
                st.markdown(f"### 🔬 **{primary_diagnosis}**")
                st.markdown(f"**Crop**: {crop_icons[crop]} {crop}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Confidence indicator
                st.metric(
                    "Confidence",
                    f"{confidence*100:.1f}%",
                    delta=None
                )
                
                # Confidence color indicator
                if confidence >= 0.7:
                    st.success("High confidence")
                elif confidence >= 0.5:
                    st.warning("Moderate confidence")
                else:
                    st.error("Low confidence - consult expert")
            
            # Alternative diagnoses
            if len(top_diagnoses) > 1 and confidence < 0.9:
                st.markdown("### 🔄 Alternative Possibilities")
                st.info("Consider these if symptoms don't match perfectly:")
                
                for i, (diag, prob) in enumerate(top_diagnoses[1:], 2):
                    if prob > 0.1:  # Only show if >10% probability
                        st.write(f"{i}. **{diag}** - {prob*100:.1f}% probability")
            
            # ================================================================
            # RECOMMENDATIONS
            # ================================================================
            
            st.markdown("---")
            st.markdown("## 💊 Treatment & Prevention")
            
            if primary_diagnosis in recommendations:
                rec = recommendations[primary_diagnosis]
                
                # Current treatment
                st.markdown('<div class="treatment-box">', unsafe_allow_html=True)
                st.markdown("### 🚑 Immediate Action Required")
                st.markdown(rec['current_treatment'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Prevention
                st.markdown('<div class="prevention-box">', unsafe_allow_html=True)
                st.markdown("### 🛡️ Prevention for Future Crops")
                st.markdown(rec['prevention'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional information
                st.markdown("### 📋 Additional Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Treatment Type**: {rec.get('recommendation_type', 'General')}")
                
                with col2:
                    validity = rec.get('validity', 'Unknown')
                    if 'ACCEPT' in validity.upper():
                        st.success(f"✅ {validity}")
                    else:
                        st.info(f"ℹ️ {validity}")
                
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning(f"""
                ⚠️ **No specific recommendations found** for {primary_diagnosis}.
                
                **Next Steps:**
                1. Consult your local agricultural extension officer
                2. Take photos of affected plants
                3. Bring leaf samples for laboratory confirmation
                4. Isolate affected plants if possible
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ================================================================
            # DISCLAIMER & FEEDBACK
            # ================================================================
            
            st.markdown("---")
            
            # Disclaimer
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("""
            ### ⚠️ Important Disclaimer
            
            This AI diagnostic tool is designed to **assist farmers** with preliminary disease identification. 
            However, it should **not replace** professional agricultural advice.
            
            **We recommend:**
            - ✅ Use this as a first step for quick guidance
            - ✅ Consult agricultural extension officers for confirmation
            - ✅ Seek laboratory testing for serious or persistent issues
            - ✅ Follow local agricultural guidelines and regulations
            
            **Accuracy Note**: AI predictions are based on symptom patterns in training data and may not 
            cover all possible diseases or regional variations.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feedback section
            st.markdown("---")
            st.markdown("### 📢 Was this diagnosis helpful?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("👍 Very Helpful"):
                    st.success("Thank you for your feedback!")
                    # Log positive feedback
            
            with col2:
                if st.button("😐 Somewhat Helpful"):
                    st.info("Thank you! We'll continue improving.")
                    # Log neutral feedback
            
            with col3:
                if st.button("👎 Not Helpful"):
                    feedback = st.text_input("What could be improved?")
                    if feedback:
                        st.warning("Thank you for helping us improve!")
                        # Log negative feedback with comment
            
        except Exception as e:
            st.error(f"""
            ❌ **An error occurred during diagnosis:**
            
            {str(e)}
            
            Please try again or contact support if the problem persists.
            """)
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Agricultural Disease Diagnosis Assistant</strong></p>
    <p>Powered by SetFit Few-Shot Learning | Trained on Ghana Ministry of Agriculture Data</p>
    <p>Developed for Agricultural AI Research | © 2024</p>
    <p style='font-size: 0.8rem; margin-top: 10px;'>
        For technical support or to report issues, contact your agricultural extension office
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - HELP & RESOURCES
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")

with st.sidebar.expander("ℹ️ How to Use"):
    st.markdown("""
    **Step-by-Step Guide:**
    
    1. **Select your crop** from the dropdown above
    2. **Describe symptoms** in detail in the text box
    3. **Click 'Diagnose Disease'** button
    4. **Review diagnosis** and confidence score
    5. **Follow recommendations** provided
    6. **Consult expert** if unsure or for serious cases
    """)

with st.sidebar.expander("🆘 Common Issues"):
    st.markdown("""
    **Problem**: Low confidence score
    - **Solution**: Provide more detailed symptoms
    
    **Problem**: Wrong diagnosis
    - **Solution**: Check alternative diagnoses listed
    
    **Problem**: No recommendations
    - **Solution**: Contact agricultural extension officer
    """)

with st.sidebar.expander("🌍 Language Support"):
    st.markdown("""
    **Currently Available**: English
    
    **Coming Soon**:
    - French (Français)
    - Twi
    - Ewe
    
    Contact us to request additional languages.
    """)

# Session state for tracking
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()
