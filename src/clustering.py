import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def perform_clustering(df, n_clusters=3):
    """
    Performs K-Means clustering on customer behavior.
    """
    print("[Clustering] Segmenting customers...")
    
    # Aggregate by Customer
    sales_cust = df[df['CAUSALE_DEL_MOVIMENTO'] == 'VEN'].groupby('CODICE_CLIENTE')['QUANTITA_MOVIMENTATA'].sum()
    returns_cust = df[df['CAUSALE_DEL_MOVIMENTO'] == 'RES'].groupby('CODICE_CLIENTE')['QUANTITA_MOVIMENTATA'].sum()
    
    customer_features = pd.DataFrame({'Total_Sales': sales_cust, 'Total_Returns': returns_cust}).fillna(0)
    
    # Feature Engineering for Clustering
    # Return Rate (Absolute value of returns / Sales)
    # Returns are negative, so we use abs() or -
    # If returns are stored as negative numbers, standard practice is abs(returns) / sales
    customer_features['Return_Rate'] = abs(customer_features['Total_Returns']) / (customer_features['Total_Sales'] + 1e-6)
    
    # Log Transform to handle skewness
    # Adding 1 to handle 0s, and using abs() for returns since log can't handle negative
    X = customer_features.copy()
    X['Log_Sales'] = np.log1p(X['Total_Sales'].clip(lower=0))
    X['Log_Returns'] = np.log1p(abs(X['Total_Returns']))
    X['Log_Return_Rate'] = np.log1p(X['Return_Rate'])
    
    features_for_clustering = ['Log_Sales', 'Log_Returns', 'Log_Return_Rate']
    X_train = X[features_for_clustering].fillna(0) # Handles cases where log might produce NaN/Inf if not handled
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_features['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add Log features to result for plotting
    customer_features['Log_Sales'] = X['Log_Sales']
    customer_features['Log_Returns'] = X['Log_Returns']
    customer_features['Log_Return_Rate'] = X['Log_Return_Rate']
    
    # Calculate PCA coordinates for Streamlit plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    customer_features['PCA1'] = X_pca[:, 0]
    customer_features['PCA2'] = X_pca[:, 1]
    
    print(f"[Clustering] Clusters distribution:\n{customer_features['Cluster'].value_counts()}")
    return customer_features, kmeans

def plot_clusters(customer_features, output_dir):
    """
    Visualizes clusters using PCA.
    """
    print("[Clustering] Generating cluster plot...")
    features = ['Log_Sales', 'Log_Returns', 'Log_Return_Rate']
    X = customer_features[features]
    
    # PCA coordinate (se già non presenti, le calcoliamo, ma ora sono calcolate in perform_clustering)
    if 'PCA1' not in customer_features.columns:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        customer_features['PCA1'] = X_pca[:, 0]
        customer_features['PCA2'] = X_pca[:, 1]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=customer_features, palette='viridis', alpha=0.6)
    plt.title('Customer Segments (PCA)')
    plt.savefig(os.path.join(output_dir, 'customer_clusters.png'))
    plt.close()

def assign_cluster_labels(customer_df):
    """
    Assigns descriptive labels to clusters based on their characteristics.
    """
    # Calculate mean stats per cluster
    cluster_stats = customer_df.groupby('Cluster')[['Total_Sales', 'Return_Rate']].mean()
    
    labels = {}
    remaining_clusters = list(cluster_stats.index)
    
    # Identify High Returners: Max Return Rate
    high_return_cluster = cluster_stats.loc[remaining_clusters, 'Return_Rate'].idxmax()
    labels[high_return_cluster] = "Alto Tasso di Reso"
    remaining_clusters.remove(high_return_cluster)
    
    # Identify Best Customers: Max Sales among remaining
    best_customer_cluster = cluster_stats.loc[remaining_clusters, 'Total_Sales'].idxmax()
    labels[best_customer_cluster] = "Top Clients (Alto Spendente)"
    remaining_clusters.remove(best_customer_cluster)
    
    # Assign Standard label to the last remaining one
    labels[remaining_clusters[0]] = "Standard / Basso Valore"
            
    # Apply labels
    customer_df['Cluster_Label'] = customer_df['Cluster'].map(labels)
    return customer_df, labels
