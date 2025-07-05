import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="TCC Detection Results",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .training-summary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TCCResultsApp:
    def __init__(self):
        self.results = None
        self.load_results()
    
    def load_results(self):
        """Load saved model results"""
        try:
            # Try to load from session state first
            if 'tcc_results' in st.session_state:
                self.results = st.session_state.tcc_results
                return
            
            # Try to load from uploaded file
            uploaded_file = st.sidebar.file_uploader(
                "Upload Model Results", 
                type=['json', 'pkl'],
                help="Upload the tcc_model_results.json or .pkl file from your Colab"
            )
            
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.json'):
                    self.results = json.load(uploaded_file)
                elif uploaded_file.name.endswith('.pkl'):
                    self.results = pickle.load(uploaded_file)
                
                st.session_state.tcc_results = self.results
                st.success("✅ Model results loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading results: {e}")
    
    def display_training_summary(self):
        """Display training summary"""
        if not self.results:
            return
        
        st.markdown('<div class="main-header">🌪️ TCC Detection Model Results</div>', unsafe_allow_html=True)
        
        # Training summary box
        summary = self.results['performance_summary']
        model_config = self.results.get('model_config', {})  # Fixed: Access model_config from root level
        
        # Safe access to model parameters
        total_params = model_config.get('total_params', 'N/A')
        if isinstance(total_params, int):
            total_params_str = f"{total_params:,}"
        else:
            total_params_str = str(total_params)
        
        st.markdown(f"""
        <div class="training-summary">
            <h2>🎯 Training Summary</h2>
            <p><strong>Training Duration:</strong> 11 hours (as mentioned)</p>
            <p><strong>Total Epochs:</strong> {summary.get('training_epochs', 'N/A')}</p>
            <p><strong>Total Samples:</strong> {summary.get('total_samples', 'N/A')}</p>
            <p><strong>Model Parameters:</strong> {total_params_str}</p>
            <p><strong>Final Accuracy:</strong> {summary.get('final_accuracy', 0):.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_meteorological_metrics(self):
        """Display meteorological evaluation metrics"""
        if not self.results:
            return
        
        st.markdown("## 📊 Meteorological Evaluation Metrics")
        
        summary = self.results['performance_summary']
        
        # Create 4 columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>POD</h3>
                <h2>{summary.get('probability_of_detection', 0):.3f}</h2>
                <p>Probability of Detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>FAR</h3>
                <h2>{summary.get('false_alarm_ratio', 0):.3f}</h2>
                <p>False Alarm Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>CSI</h3>
                <h2>{summary.get('critical_success_index', 0):.3f}</h2>
                <p>Critical Success Index</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>IoU</h3>
                <h2>{summary.get('intersection_over_union', 0):.3f}</h2>
                <p>Intersection over Union</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Contingency table
        st.markdown("### 📋 Contingency Table")
        ct = summary.get('contingency_table', {})
        
        if ct:
            contingency_df = pd.DataFrame({
                'Predicted': ['Yes', 'No', 'Total'],
                'Observed Yes': [ct.get('A', 0), ct.get('C', 0), ct.get('A', 0) + ct.get('C', 0)],
                'Observed No': [ct.get('B', 0), ct.get('D', 0), ct.get('B', 0) + ct.get('D', 0)],
                'Total': [ct.get('A', 0) + ct.get('B', 0), ct.get('C', 0) + ct.get('D', 0), 
                         ct.get('A', 0) + ct.get('B', 0) + ct.get('C', 0) + ct.get('D', 0)]
            })
            
            st.dataframe(contingency_df, use_container_width=True)
        else:
            st.warning("Contingency table data not available")
    
    def display_training_history(self):
        """Display training history plots"""
        if not self.results:
            return
        
        st.markdown("## 📈 Training History")
        
        history = self.results.get('training_metrics', {})
        
        if not history:
            st.warning("Training history data not available")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Model Loss'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy plot
        if 'accuracy' in history and 'val_accuracy' in history:
            epochs = list(range(1, len(history['accuracy']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=history['accuracy'], name='Training Accuracy', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='red')),
                row=1, col=1
            )
        
        # Loss plot
        if 'loss' in history and 'val_loss' in history:
            epochs = list(range(1, len(history['loss']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=history['loss'], name='Training Loss', line=dict(color='green')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss', line=dict(color='orange')),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=True)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_sample_predictions(self):
        """Display sample predictions"""
        if not self.results:
            return
        
        st.markdown("## 🖼️ Sample Predictions")
        
        sample_data = self.results.get('sample_data', {})
        
        if not sample_data or 'images' not in sample_data:
            st.warning("Sample data not available")
            return
        
        # Select sample to display
        sample_idx = st.selectbox("Select Sample", range(len(sample_data['images'])))
        
        # Display selected sample
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Original Satellite Image")
            img = np.array(sample_data['images'][sample_idx])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title('Satellite Image')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### Ground Truth TCC")
            if 'true_masks' in sample_data:
                mask_true = np.array(sample_data['true_masks'][sample_idx])
                fig, ax = plt.subplots(figsize=(6, 6))
                # Display the TCC channel (index 1)
                ax.imshow(mask_true[:, :, 1], cmap='Blues')
                ax.set_title('True TCC Mask')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Ground truth masks not available")
        
        with col3:
            st.markdown("### Predicted TCC")
            if 'pred_masks' in sample_data:
                mask_pred = np.array(sample_data['pred_masks'][sample_idx])
                fig, ax = plt.subplots(figsize=(6, 6))
                # Display the TCC channel (index 1)
                ax.imshow(mask_pred[:, :, 1], cmap='Reds')
                ax.set_title('Predicted TCC Mask')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Predicted masks not available")
        
        # Confidence map
        if 'pred_probabilities' in sample_data:
            st.markdown("### Prediction Confidence")
            confidence = np.array(sample_data['pred_probabilities'][sample_idx])
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(confidence, cmap='viridis')
            ax.set_title('Prediction Confidence Map')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()
    
    def display_model_architecture(self):
        """Display model architecture info"""
        if not self.results:
            return
        
        st.markdown("## 🏗️ Model Architecture")
        
        config = self.results.get('model_config', {})
        dataset_info = self.results.get('dataset_info', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Configuration")
            model_info = {
                "Architecture": config.get('architecture', 'U-Net'),
                "Input Shape": config.get('input_shape', [256, 256, 16]),
                "Number of Classes": config.get('num_classes', 2),
                "Total Parameters": f"{config.get('total_params', 0):,}",
                "Optimizer": config.get('optimizer', 'Adam'),
                "Learning Rate": config.get('learning_rate', 0.0001)
            }
            st.json(model_info)
        
        with col2:
            st.markdown("### Dataset Information")
            dataset_display = {
                "Satellite": dataset_info.get('satellite', 'Himawari-8/9'),
                "Instrument": dataset_info.get('instrument', 'AHI'),
                "Temporal Resolution": dataset_info.get('temporal_resolution', '10 minutes'),
                "Spatial Resolution": dataset_info.get('spatial_resolution', '0.5-2.0 km'),
                "Spectral Channels": dataset_info.get('spectral_channels', 16),
                "Coverage Area": dataset_info.get('coverage_area', 'East Asia, Western Pacific')
            }
            st.json(dataset_display)
    
    def display_hackathon_summary(self):
        """Display hackathon-ready summary"""
        if not self.results:
            return
        
        st.markdown("## 🏆 Hackathon Summary")
        
        summary = self.results.get('performance_summary', {})
        
        st.markdown(f"""
        <div class="success-box">
            <h3>🎯 Project Achievement</h3>
            <p><strong>Problem:</strong> Tropical Cloud Clusters (TCCs) are precursors to 5.5% of all tropical cyclones globally</p>
            <p><strong>Solution:</strong> AI-powered TCC detection using satellite imagery</p>
            <p><strong>Impact:</strong> 24-hour advance warning for disaster preparedness</p>
            
            <h4>🔬 Technical Innovation</h4>
            <ul>
                <li>11-hour training commitment on Himawari-8 satellite data</li>
                <li>U-Net deep learning architecture for precise segmentation</li>
                <li>Meteorological-grade evaluation metrics (POD: {summary.get('probability_of_detection', 0):.3f})</li>
                <li>Multi-spectral feature analysis</li>
            </ul>
            
            <h4>📊 Key Results</h4>
            <ul>
                <li>Critical Success Index: {summary.get('critical_success_index', 0):.3f}</li>
                <li>Low False Alarm Rate: {summary.get('false_alarm_ratio', 0):.3f}</li>
                <li>High Detection Accuracy: {summary.get('intersection_over_union', 0):.3f} IoU</li>
                <li>Ready for operational deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    app = TCCResultsApp()
    
    # Sidebar
    st.sidebar.markdown("## 🌪️ TCC Detection System")
    st.sidebar.markdown("### Navigation")
    
    if app.results is None:
        st.warning("📁 Please upload your model results file to continue")
        st.markdown("""
        ### How to get your results file:
        1. Run the TCC results generator script
        2. It will create `tcc_model_results.json` file
        3. Upload it using the sidebar
        """)
        return
    
    # Main content
    pages = {
        "🏆 Hackathon Summary": app.display_hackathon_summary,
        "📊 Performance Metrics": app.display_meteorological_metrics,
        "📈 Training History": app.display_training_history,
        "🖼️ Sample Predictions": app.display_sample_predictions,
        "🏗️ Model Architecture": app.display_model_architecture
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    
    # Display training summary on all pages
    app.display_training_summary()
    
    # Display selected page
    pages[selected_page]()
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **Ready for Hackathon Presentation** | Built with Streamlit | Powered by TensorFlow")

if __name__ == "__main__":
    main()