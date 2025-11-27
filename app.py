"""
Streamlit web application for Brain Tumor MRI Classification.
Provides an interactive interface for inference on uploaded MRI images.
"""

import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import config


# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path):
    """
    Load the trained model with caching.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded Keras model
    """
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array ready for prediction
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array, image


def predict(model, processed_image):
    """
    Make prediction on the preprocessed image.
    
    Args:
        model: Trained Keras model
        processed_image: Preprocessed image array
    
    Returns:
        Predictions array
    """
    predictions = model.predict(processed_image, verbose=0)
    return predictions[0]


def plot_predictions(predictions, class_names):
    """
    Create an interactive bar chart of predictions using Plotly.
    
    Args:
        predictions: Prediction probabilities
        class_names: List of class names
    
    Returns:
        Plotly figure object
    """
    # Create color map
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=predictions * 100,
            marker_color=colors,
            text=[f'{p*100:.2f}%' for p in predictions],
            textposition='outside',
            hovertemplate='%{x}<br>Probability: %{y:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Prediction Probabilities',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title='Tumor Type',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 105],
        height=400,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    
    return fig


def main():
    """
    Main Streamlit application.
    """
    # Sidebar
    with st.sidebar:
        st.title("🧠 Brain Tumor Classifier")
        st.markdown("---")
        
        st.markdown("### 📋 Project Information")
        st.markdown("""
        This application uses deep learning to classify brain MRI scans 
        into four categories:
        
        - **Glioma Tumor**
        - **Meningioma Tumor**
        - **Pituitary Tumor**
        - **No Tumor**
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔬 Model Selection")
        
        # Check available models
        available_models = {}
        if os.path.exists(config.CUSTOM_CNN_MODEL_PATH):
            available_models['Custom CNN'] = config.CUSTOM_CNN_MODEL_PATH
        if os.path.exists(config.TRANSFER_MODEL_PATH):
            available_models[f'Transfer Learning ({config.PRETRAINED_MODEL})'] = config.TRANSFER_MODEL_PATH
        
        if not available_models:
            st.error("⚠️ No trained models found!")
            st.info("Please train the models first using: `python train.py`")
            return
        
        model_choice = st.selectbox(
            "Select Model:",
            list(available_models.keys())
        )
        
        model_path = available_models[model_choice]
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.info("""
        **Note:** This is an educational and research demonstration. 
        It should NOT be used for clinical diagnosis or medical decision-making. 
        Always consult qualified healthcare professionals for medical advice.
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Class Labels")
        for i, class_name in enumerate(config.CLASS_NAMES):
            st.markdown(f"**{i}.** {class_name.replace('_', ' ').title()}")
    
    # Main content
    st.title("🧠 Brain Tumor MRI Classification")
    st.markdown("### Upload an MRI scan to classify the tumor type")
    
    # Load model
    with st.spinner(f"Loading {model_choice}..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    st.success(f"✅ {model_choice} loaded successfully!")
    
    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan in JPG or PNG format"
    )
    
    if uploaded_file is not None:
        try:
            # Display original image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                # Image info
                st.markdown(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                st.markdown(f"**Mode:** {image.mode}")
            
            # Preprocess and predict
            with st.spinner("🔄 Processing image and making prediction..."):
                processed_image, resized_image = preprocess_image(image)
                predictions = predict(model, processed_image)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions)
            predicted_class_name = config.CLASS_NAMES[predicted_class_idx]
            confidence = predictions[predicted_class_idx] * 100
            
            with col2:
                st.markdown("### 🎯 Prediction Result")
                
                # Display prediction with color coding
                if predicted_class_name == 'no_tumor':
                    result_color = "#2ecc71"  # Green
                    icon = "✅"
                else:
                    result_color = "#e74c3c"  # Red
                    icon = "⚠️"
                
                st.markdown(
                    f"""
                    <div style="
                        background-color: {result_color}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        text-align: center;
                        margin-bottom: 20px;
                    ">
                        <h2 style="color: white; margin: 0;">
                            {icon} {predicted_class_name.replace('_', ' ').title()}
                        </h2>
                        <h3 style="color: white; margin: 10px 0 0 0;">
                            Confidence: {confidence:.2f}%
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display preprocessed image
                st.markdown("**Preprocessed Image (224x224)**")
                st.image(resized_image, use_container_width=True)
            
            # Prediction probabilities
            st.markdown("---")
            st.markdown("### 📊 Detailed Prediction Probabilities")
            
            # Interactive plot
            fig = plot_predictions(predictions, 
                                 [name.replace('_', ' ').title() for name in config.CLASS_NAMES])
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities table
            st.markdown("#### Probability Breakdown")
            prob_data = {
                'Class': [name.replace('_', ' ').title() for name in config.CLASS_NAMES],
                'Probability': [f'{p*100:.2f}%' for p in predictions],
                'Confidence': predictions
            }
            
            import pandas as pd
            df = pd.DataFrame(prob_data)
            df = df.sort_values('Confidence', ascending=False).reset_index(drop=True)
            
            # Style the dataframe
            st.dataframe(
                df[['Class', 'Probability']],
                use_container_width=True,
                hide_index=True
            )
            
            # Additional information
            st.markdown("---")
            st.markdown("### 💡 Interpretation Guide")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("#### 🔴 Glioma")
                st.markdown("Most common and aggressive brain tumor in adults")
            
            with col2:
                st.markdown("#### 🟠 Meningioma")
                st.markdown("Usually benign tumor in the meninges")
            
            with col3:
                st.markdown("#### 🟡 Pituitary")
                st.markdown("Tumor in the pituitary gland")
            
            with col4:
                st.markdown("#### 🟢 No Tumor")
                st.markdown("No tumor detected in the scan")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.info("Please make sure you uploaded a valid image file.")
    
    else:
        # Display example/instructions when no file is uploaded
        st.info("👆 Please upload a brain MRI image to get started")
        
        st.markdown("---")
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. **Select a model** from the sidebar (Custom CNN or Transfer Learning)
        2. **Upload an MRI image** using the file uploader above
        3. **View the prediction** and confidence scores
        4. **Analyze the probability distribution** across all classes
        
        The model will classify the image into one of four categories:
        - Glioma Tumor
        - Meningioma Tumor
        - Pituitary Tumor
        - No Tumor
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Custom CNN")
            st.markdown("""
            - Deep convolutional architecture
            - Trained from scratch
            - Optimized for MRI classification
            """)
        
        with col2:
            st.markdown(f"#### Transfer Learning ({config.PRETRAINED_MODEL})")
            st.markdown("""
            - Pre-trained on ImageNet
            - Fine-tuned for brain tumors
            - Generally higher accuracy
            """)


if __name__ == "__main__":
    main()
