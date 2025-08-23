from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import json
import plotly
import base64
from io import BytesIO
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Global variable to store data
data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.endswith('.csv'):
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load the data
        data = pd.read_csv(filepath)
        
        # Basic data info
        info = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'missing_values': data.isnull().sum().to_dict(),
            'head': data.head().to_html(classes='table table-striped')
        }
        
        return jsonify({'success': True, 'info': info})
    
    return jsonify({'error': 'Please upload a CSV file'})

@app.route('/demographic_analysis')
def demographic_analysis():
    global data
    if data is None:
        return jsonify({'error': 'No data loaded'})
    
    # Create demographic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Key Demographic Variables', fontsize=16)
    
    sns.set_style("whitegrid")
    
    # Age distribution
    sns.countplot(ax=axes[0, 0], x='Age', data=data, palette='coolwarm')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Gender distribution
    sns.countplot(ax=axes[0, 1], x='Gender', data=data, palette='coolwarm')
    axes[0, 1].set_title('Gender Distribution')
    
    # Education Level distribution
    sns.countplot(ax=axes[1, 0], x='Education Level', data=data, palette='coolwarm')
    axes[1, 0].set_title('Education Level Distribution')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Income Level distribution
    sns.countplot(ax=axes[1, 1], x='Income Level', data=data, palette='coolwarm')
    axes[1, 1].set_title('Income Level Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

@app.route('/behavior_analysis')
def behavior_analysis():
    global data
    if data is None:
        return jsonify({'error': 'No data loaded'})
    
    # Create behavior analysis plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('User Online Behavior and Ad Interaction Metrics', fontsize=16)
    
    # Time spent online weekday
    sns.histplot(ax=axes[0, 0], x='Time Spent Online (hrs/weekday)', data=data, bins=20, kde=True, color='skyblue')
    axes[0, 0].set_title('Weekday Time Online')
    
    # Time spent online weekend
    sns.histplot(ax=axes[0, 1], x='Time Spent Online (hrs/weekend)', data=data, bins=20, kde=True, color='orange')
    axes[0, 1].set_title('Weekend Time Online')
    
    # Likes and reactions
    sns.histplot(ax=axes[1, 0], x='Likes and Reactions', data=data, bins=20, kde=True, color='green')
    axes[1, 0].set_title('Likes and Reactions')
    
    # Click-through rates
    sns.histplot(ax=axes[1, 1], x='Click-Through Rates (CTR)', data=data, bins=20, kde=True, color='red')
    axes[1, 1].set_title('CTR')
    
    # Conversion rates
    sns.histplot(ax=axes[2, 0], x='Conversion Rates', data=data, bins=20, kde=True, color='purple')
    axes[2, 0].set_title('Conversion Rates')
    
    # Ad interaction time
    sns.histplot(ax=axes[2, 1], x='Ad Interaction Time (sec)', data=data, bins=20, kde=True, color='brown')
    axes[2, 1].set_title('Ad Interaction Time')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

@app.route('/interests_analysis')
def interests_analysis():
    global data
    if data is None:
        return jsonify({'error': 'No data loaded'})
    
    # Split and count interests
    interests_list = data['Top Interests'].str.split(', ').sum()
    interests_counter = Counter(interests_list)
    
    interests_df = pd.DataFrame(interests_counter.items(), columns=['Interest', 'Frequency'])
    interests_df = interests_df.sort_values(by='Frequency', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Interest', data=interests_df.head(10), palette='coolwarm')
    plt.title('Top 10 User Interests', fontsize=16)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot': plot_url})

@app.route('/clustering_analysis')
def clustering_analysis():
    global data
    if data is None:
        return jsonify({'error': 'No data loaded'})
    
    # Prepare features for clustering
    features = ['Age', 'Gender', 'Income Level', 'Time Spent Online (hrs/weekday)',
                'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']
    
    X = data[features].copy()
    
    # Preprocessing
    numeric_features = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)',
                        'Likes and Reactions', 'Click-Through Rates (CTR)']
    categorical_features = ['Age', 'Gender', 'Income Level']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=5, random_state=42))
    ])
    
    pipeline.fit(X)
    data['Cluster'] = pipeline.named_steps['cluster'].labels_
    
    # Cluster analysis
    cluster_means = data.groupby('Cluster')[numeric_features].mean()
    
    for feature in categorical_features:
        mode_series = data.groupby('Cluster')[feature].agg(lambda x: x.mode()[0])
        cluster_means[feature] = mode_series
    
    # Create radar chart
    features_to_plot = numeric_features
    labels = np.array(features_to_plot)
    
    radar_df = cluster_means[features_to_plot].reset_index()
    
    # Normalize data for radar chart
    radar_df_normalized = radar_df.copy()
    for feature in features_to_plot:
        radar_df_normalized[feature] = (
            radar_df[feature] - radar_df[feature].min()) / (
            radar_df[feature].max() - radar_df[feature].min())
    
    # Segment names
    segment_names = ['Weekend Warriors', 'Engaged Professionals', 'Low-Key Users',
                     'Active Explorers', 'Budget Browsers']
    
    # Create radar chart
    fig = go.Figure()
    
    for i, segment in enumerate(segment_names):
        values = radar_df_normalized.iloc[i][features_to_plot].values.tolist()
        values += [values[0]]  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels.tolist() + [labels[0]],
            fill='toself',
            name=segment
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='User Segments Profile'
    )
    
    # Convert to JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Cluster summary table
    cluster_summary = cluster_means.round(2).to_html(classes='table table-striped')
    
    return jsonify({
        'radar_chart': graphJSON,
        'cluster_summary': cluster_summary
    })

if __name__ == '__main__':
    app.run(debug=True)