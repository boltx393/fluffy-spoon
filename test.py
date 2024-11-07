import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load and merge relevant NBA datasets"""
    print("Loading NBA datasets...")
    
    # Load all datasets
    games = pd.read_csv('games.csv')
    games_details = pd.read_csv('games_details.csv')
    players = pd.read_csv('players.csv')
    teams = pd.read_csv('teams.csv')
    
    print("\nDataset shapes:")
    print(f"Games: {games.shape}")
    print(f"Games Details: {games_details.shape}")
    print(f"Players: {players.shape}")
    print(f"Teams: {teams.shape}")
    
    return games, games_details, players, teams

def preprocess_game_data(games, games_details, teams):
    """Preprocess game-level data for analysis"""
    print("\n1. Preprocessing Game Data:")
    
    # Merge games with teams data
    games_processed = games.merge(teams[['TEAM_ID', 'NICKNAME']], 
                                left_on='HOME_TEAM_ID', 
                                right_on='TEAM_ID', 
                                suffixes=('', '_HOME'))
    games_processed = games_processed.merge(teams[['TEAM_ID', 'NICKNAME']], 
                                         left_on='VISITOR_TEAM_ID', 
                                         right_on='TEAM_ID', 
                                         suffixes=('_HOME', '_AWAY'))
    
    # Calculate point difference and create target for classification
    games_processed['POINT_DIFF'] = games_processed['PTS_HOME'] - games_processed['PTS_AWAY']
    games_processed['HOME_TEAM_WIN'] = (games_processed['POINT_DIFF'] > 0).astype(int)
    
    # Create features for game prediction
    games_processed['SEASON_MONTH'] = pd.to_datetime(games_processed['GAME_DATE_EST']).dt.month
    
    # Handle missing values
    games_processed = games_processed.dropna(subset=['PTS_HOME', 'PTS_AWAY'])
    
    print(f"\nProcessed games shape: {games_processed.shape}")
    return games_processed

def preprocess_player_data(games_details, players):
    """Preprocess player-level data for analysis"""
    print("\n2. Preprocessing Player Data:")
    
    # Merge games_details with players data
    player_stats = games_details.merge(players[['PLAYER_ID', 'PLAYER_NAME']], 
                                     on='PLAYER_ID', 
                                     how='left')
    
    # Calculate efficiency metrics
    player_stats['EFFICIENCY'] = (
        player_stats['PTS'] + 
        player_stats['REB'] + 
        player_stats['AST'] + 
        player_stats['STL'] + 
        player_stats['BLK'] - 
        (player_stats['FGA'] - player_stats['FGM']) - 
        (player_stats['FTA'] - player_stats['FTM']) - 
        player_stats['TO']
    )
    
    # Group by player and calculate averages
    player_averages = player_stats.groupby('PLAYER_ID').agg({
        'PLAYER_NAME': 'first',
        'MIN': 'mean',
        'PTS': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'EFFICIENCY': 'mean'
    }).reset_index()
    
    # Remove players with too few minutes
    player_averages = player_averages[player_averages['MIN'] >= 10]
    
    print(f"\nProcessed player stats shape: {player_averages.shape}")
    return player_averages

def perform_game_classification(games_processed):
    """Predict home team wins using game features"""
    print("\n3. Game Outcome Classification:")
    
    # Prepare features
    features = ['SEASON_MONTH', 'HOME_TEAM_WINS', 'VISITOR_TEAM_WINS']
    X = games_processed[features]
    y = games_processed['HOME_TEAM_WIN']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nHome Team Win Prediction Accuracy: {accuracy:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(clf.coef_[0])
    })
    print("\nFeature Importance for Game Prediction:")
    print(feature_importance.sort_values('Importance', ascending=False))
    
    return clf

def perform_scoring_regression(player_averages):
    """Predict player scoring using other statistics"""
    print("\n4. Player Scoring Regression:")
    
    # Prepare features
    features = ['MIN', 'REB', 'AST', 'STL', 'BLK']
    X = player_averages[features]
    y = player_averages['PTS']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train regressor
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = reg.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = reg.score(X_test, y_pred)
    
    print(f"\nScoring Prediction RMSE: {rmse:.2f}")
    print(f"R-squared score: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': reg.coef_
    })
    print("\nFeature Importance for Scoring Prediction:")
    print(feature_importance.sort_values('Coefficient', ascending=False))
    
    return reg

def perform_player_clustering(player_averages):
    """Cluster players based on their playing style"""
    print("\n5. Player Style Clustering:")
    
    # Prepare features for clustering
    features = ['PTS', 'REB', 'AST']
    X = player_averages[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    silhouette_scores = []
    K = range(2, 6)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    optimal_k = K[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    player_averages['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(player_averages['PTS'], 
                         player_averages['REB'],
                         c=player_averages['Cluster'],
                         cmap='viridis',
                         alpha=0.6)
    
    plt.xlabel('Points per Game')
    plt.ylabel('Rebounds per Game')
    plt.title('NBA Player Clusters based on Performance Metrics')
    plt.colorbar(scatter, label='Player Type')
    
    # Analyze clusters
    cluster_stats = player_averages.groupby('Cluster')[features].mean()
    print("\nCluster Characteristics:")
    print(cluster_stats)
    
    return kmeans

def main():
    # Load data
    games, games_details, players, teams = load_data()
    
    # Preprocess data
    games_processed = preprocess_game_data(games, games_details, teams)
    player_averages = preprocess_player_data(games_details, players)
    
    # Perform analyses
    game_classifier = perform_game_classification(games_processed)
    scoring_regressor = perform_scoring_regression(player_averages)
    player_clusters = perform_player_clustering(player_averages)
    
    # Save visualization
    plt.savefig('player_clusters.png')
    plt.close()
    
    print("\nAnalysis complete! Check 'player_clusters.png' for the clustering visualization.")

if __name__ == "__main__":
    main()