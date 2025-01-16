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

# Page configuration
st.set_page_config(page_title="Customer Segmentation Analysis", layout="wide")

# Title and description
st.title("Customer Segmentation Analysis")
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

# Load the data
data = load_data()

if data is not None:
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "EDA", "Clustering", "3D Analysis", "Customer Profiles"])

    with tab1:
        st.header("Data Overview")
        st.write("First few rows of the dataset:")
        st.dataframe(data.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Description:")
            st.dataframe(data.describe())
        with col2:
            st.write("Data Info:")
            buffer = StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())

    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Distribution plots
        st.subheader("Distribution of Key Features")
        fig1 = plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(data=data, x='Age', bins=20)
        plt.title('Age Distribution')
        
        plt.subplot(1, 3, 2)
        sns.histplot(data=data, x='Annual Income (k$)', bins=20)
        plt.title('Income Distribution')
        
        plt.subplot(1, 3, 3)
        sns.histplot(data=data, x='Spending Score (1-100)', bins=20)
        plt.title('Spending Score Distribution')
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Gender analysis
        st.subheader("Analysis by Gender")
        fig2 = plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.boxplot(data=data, x='Gender', y='Age')
        plt.title('Age by Gender')
        
        plt.subplot(1, 3, 2)
        sns.boxplot(data=data, x='Gender', y='Annual Income (k$)')
        plt.title('Income by Gender')
        
        plt.subplot(1, 3, 3)
        sns.boxplot(data=data, x='Gender', y='Spending Score (1-100)')
        plt.title('Spending Score by Gender')
        
        plt.tight_layout()
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
            
            fig3 = plt.figure(figsize=(10, 6))
            plt.plot(K, inertias, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            st.pyplot(fig3)
        
        with col2:
            st.subheader("Silhouette Analysis")
            silhouette_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                score = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(score)
            
            fig4 = plt.figure(figsize=(10, 6))
            plt.plot(range(2, 11), silhouette_scores, 'rx-')
            plt.xlabel('k')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis')
            st.pyplot(fig4)
        
        # Perform clustering with optimal k=5
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Visualize clusters
        st.subheader("Customer Segments")
        fig5 = plt.figure(figsize=(12, 8))
        scatter = plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'],
                            c=clusters, cmap='viridis')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.title('Customer Segments')
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
        fig7 = plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.boxplot(data=data, x='Cluster', y='Age')
        plt.title('Age Distribution by Cluster')
        
        plt.subplot(1, 3, 2)
        sns.boxplot(data=data, x='Cluster', y='Annual Income (k$)')
        plt.title('Income Distribution by Cluster')
        
        plt.subplot(1, 3, 3)
        sns.boxplot(data=data, x='Cluster', y='Spending Score (1-100)')
        plt.title('Spending Score Distribution by Cluster')
        
        plt.tight_layout()
        st.pyplot(fig7)

else:
    st.error("Unable to load data. Please check your data source.")

# Add footer
st.markdown("---")
st.markdown("Created by: dsrojo10")
st.markdown(f"Last updated: 2025-01-16 01:30:12 UTC")