import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from io import StringIO

# Set style for all plots
plt.style.use('seaborn')
sns.set_theme()

# Page configuration
st.set_page_config(page_title="Proyecto 3: Customer Segmentation Analysis", layout="wide")

# Title and description
st.title("Proyecto 3: Customer Segmentation Analysis")
st.markdown("""
### Description:
This application performs customer segmentation using K-means clustering based on customer purchasing behavior.
It helps understand how to group customers with similar characteristics for targeted marketing strategies.

### Requirements:
- **Tools:** Python, Scikit-learn, pandas, Matplotlib, seaborn
- **Skills:** Data preprocessing, feature engineering, clustering, data visualization
- **Dataset:** "Mall Customer Segmentation Data" from Kaggle
""")

# Function to load data
@st.cache_data
def load_data():
    # You'll need to modify this path to where your data is stored
    try:
        data = pd.read_csv('data/Mall_Customers.csv')
        return data
    except:
        st.error("Please ensure the data file is in the correct location")
        return None

# Function to create distribution plots
def create_distribution_plots(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.histplot(data=data, x='Age', bins=20, ax=axes[0])
    axes[0].set_title('Age Distribution')
    
    sns.histplot(data=data, x='Annual Income (k$)', bins=20, ax=axes[1])
    axes[1].set_title('Income Distribution')
    
    sns.histplot(data=data, x='Spending Score (1-100)', bins=20, ax=axes[2])
    axes[2].set_title('Spending Score Distribution')
    
    plt.tight_layout()
    return fig

# Function to create gender analysis plots
def create_gender_plots(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.boxplot(data=data, x='Gender', y='Age', ax=axes[0])
    axes[0].set_title('Age by Gender')
    
    sns.boxplot(data=data, x='Gender', y='Annual Income (k$)', ax=axes[1])
    axes[1].set_title('Income by Gender')
    
    sns.boxplot(data=data, x='Gender', y='Spending Score (1-100)', ax=axes[2])
    axes[2].set_title('Spending Score by Gender')
    
    plt.tight_layout()
    return fig

# Load the data
data = load_data()

if data is not None:
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "EDA", "Clustering", "3D Analysis", "Customer Profiles"])

    with tab1:
        st.header("Data Overview")
        st.write("First few rows of the dataset:")
        st.dataframe(data.head())
        
        st.write("Data Description:")
        st.dataframe(data.describe())

        st.write("Data Info:")
        buffer = StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Distribution plots using the new function
        st.subheader("Distribution of Key Features")
        fig1 = create_distribution_plots(data)
        st.pyplot(fig1)
        
        # Gender analysis using the new function
        st.subheader("Analysis by Gender")
        fig2 = create_gender_plots(data)
        st.pyplot(fig2)

    with tab3:
        st.header("Customer Clustering")
        
        # Prepare data for clustering
        X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Elbow method and silhouette analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Elbow Method")
            inertias = []
            K = range(1, 11)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(K, inertias, 'bx-')
            ax3.set_xlabel('k')
            ax3.set_ylabel('Inertia')
            ax3.set_title('Elbow Method for Optimal k')
            st.pyplot(fig3)
        
        with col2:
            st.subheader("Silhouette Analysis")
            silhouette_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                score = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(score)
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.plot(range(2, 11), silhouette_scores, 'rx-')
            ax4.set_xlabel('k')
            ax4.set_ylabel('Silhouette Score')
            ax4.set_title('Silhouette Analysis')
            st.pyplot(fig4)
        
        # Perform clustering with optimal k=5
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Visualize clusters
        st.subheader("Customer Segments")
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        scatter = ax5.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'],
                            c=clusters, cmap='viridis')
        ax5.set_xlabel('Annual Income (k$)')
        ax5.set_ylabel('Spending Score (1-100)')
        ax5.set_title('Customer Segments')
        plt.colorbar(scatter)
        st.pyplot(fig5)

    with tab4:
        st.header("3D Analysis")
        
        # 3D clustering
        X_3d = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
        X_3d_scaled = StandardScaler().fit_transform(X_3d)
        kmeans_3d = KMeans(n_clusters=5, random_state=42)
        clusters_3d = kmeans_3d.fit_predict(X_3d_scaled)
        
        # Create 3D scatter plot using Plotly
        fig6 = px.scatter_3d(
            data_frame=data,
            x='Age',
            y='Annual Income (k$)',
            z='Spending Score (1-100)',
            color=clusters_3d,
            title='3D Customer Segmentation',
            labels={'color': 'Cluster'}
        )
        st.plotly_chart(fig6)

    with tab5:
        st.header("Customer Profiles")
        
        # Add cluster labels to the data
        data['Cluster'] = clusters
        
        # Calculate cluster profiles
        profiles = data.groupby('Cluster').agg({
            'Age': ['mean', 'min', 'max'],
            'Annual Income (k$)': ['mean', 'min', 'max'],
            'Spending Score (1-100)': ['mean', 'min', 'max'],
            'Gender': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        st.write("Cluster Profiles:")
        st.dataframe(profiles)
        
        # Visualization of cluster characteristics
        st.subheader("Cluster Characteristics")
        fig7, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sns.boxplot(data=data, x='Cluster', y='Age', ax=axes[0])
        axes[0].set_title('Age Distribution by Cluster')
        
        sns.boxplot(data=data, x='Cluster', y='Annual Income (k$)', ax=axes[1])
        axes[1].set_title('Income Distribution by Cluster')
        
        sns.boxplot(data=data, x='Cluster', y='Spending Score (1-100)', ax=axes[2])
        axes[2].set_title('Spending Score Distribution by Cluster')
        
        plt.tight_layout()
        st.pyplot(fig7)

else:
    st.error("Unable to load data. Please check your data source.")

# Add footer
st.markdown("---")
st.markdown("Created by: dsrojo10")
st.markdown(f"Last updated: 2025-01-16 01:30:12 UTC")