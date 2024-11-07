import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix, roc_auc_score, classification_report)
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def load_data():
    """Load and merge relevant NBA datasets"""
    print("Loading NBA datasets")
    
    games = pd.read_csv('games.csv')
    games_details = pd.read_csv('games_details.csv', low_memory=False)
    players = pd.read_csv('players.csv')
    teams = pd.read_csv('teams.csv')
    
    print("\nDataset shapes:")
    print(f"Games: {games.shape}")
    print(f"Games Details: {games_details.shape}")
    print(f"Players: {players.shape}")
    print(f"Teams: {teams.shape}")
    
    return games, games_details, players, teams

def preprocess_data(games, games_details, players, teams):
    """Comprehensive data preprocessing"""
    print("\n1. Data Preprocessing:")
    
    # Create game features
    games_processed = games.copy()
    games_processed['POINT_DIFF'] = games_processed['PTS_home'] - games_processed['PTS_away']
    games_processed['SEASON_MONTH'] = pd.to_datetime(games_processed['GAME_DATE_EST']).dt.month
    
    # Calculate rolling averages for team statistics
    for team_id in games_processed['HOME_TEAM_ID'].unique():
        team_games = games_processed[games_processed['HOME_TEAM_ID'] == team_id]
        games_processed.loc[games_processed['HOME_TEAM_ID'] == team_id, 'ROLLING_PTS_HOME'] = \
            team_games['PTS_home'].rolling(window=5, min_periods=1).mean()
    
    # Feature engineering
    features = ['SEASON_MONTH', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 
               'FG3_PCT_home', 'AST_home', 'REB_home', 'ROLLING_PTS_HOME']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    games_processed[features] = imputer.fit_transform(games_processed[features])
    
    # Scale features
    scaler = StandardScaler()
    games_processed[features] = scaler.fit_transform(games_processed[features])
    
    print("\nFeatures created:", features)
    print("\nMissing values after preprocessing:", games_processed[features].isnull().sum().sum())
    
    return games_processed, features

def random_forest_classification(games_processed, features):
    """Random Forest Classification with comprehensive evaluation"""
    print("\n2. Random Forest Classification:")
    
    X = games_processed[features]
    y = games_processed['HOME_TEAM_WINS']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Grid Search for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Performance metrics
    print("\nRandom Forest Performance Metrics:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_rf, X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('rf_confusion_matrix.png')
    plt.close()
    
    return best_rf, feature_importance

def logistic_regression(games_processed, features):
    """Logistic Regression with comprehensive evaluation"""
    print("\n3. Logistic Regression:")
    
    X = games_processed[features]
    y = games_processed['HOME_TEAM_WINS']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Grid Search for hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_lr = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_lr.predict(X_test)
    y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
    
    # Performance metrics
    print("\nLogistic Regression Performance Metrics:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_lr

def kmodes_clustering(games_processed, features):
    """K-Modes Clustering with evaluation"""
    print("\n4. K-Modes Clustering:")
    
    # Prepare data for clustering
    X = games_processed[features].copy()
    
    # Convert numerical features to categorical for K-Modes
    for feature in features:
        X[feature] = pd.qcut(X[feature], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Find optimal number of clusters using elbow method
    cost = []
    K = range(2, 7)
    for k in K:
        kmode = KModes(n_clusters=k, init='Huang', n_init=5, random_state=42)
        kmode.fit(X)
        cost.append(kmode.cost_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, cost, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Curve for K-Modes Clustering')
    plt.savefig('kmodes_elbow.png')
    plt.close()
    
    # Select optimal k (you may need to adjust this based on the elbow curve)
    optimal_k = K[np.argmin(np.diff(cost)) + 1]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Perform final clustering
    kmodes = KModes(n_clusters=optimal_k, init='Huang', n_init=5, random_state=42)
    clusters = kmodes.fit_predict(X)
    
    # Analyze clusters
    games_processed['Cluster'] = clusters
    cluster_analysis = games_processed.groupby('Cluster')[features].mean()
    
    print("\nCluster Characteristics:")
    print(cluster_analysis)
    
    # Visualize cluster sizes
    plt.figure(figsize=(10, 6))
    games_processed['Cluster'].value_counts().plot(kind='bar')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Games')
    plt.savefig('cluster_distribution.png')
    plt.close()
    
    return kmodes, cluster_analysis

def main():
    # Load data
    games, games_details, players, teams = load_data()
    
    # Preprocess data
    games_processed, features = preprocess_data(games, games_details, players, teams)
    
    # Perform analyses
    rf_model, feature_importance = random_forest_classification(games_processed, features)
    lr_model = logistic_regression(games_processed, features)
    kmodes_model, cluster_analysis = kmodes_clustering(games_processed, features)
    
    # Save results
    print("\nAnalysis complete! Check the generated visualization files:")
    print("1. rf_confusion_matrix.png - Random Forest performance visualization")
    print("2. kmodes_elbow.png - K-Modes clustering elbow curve")
    print("3. cluster_distribution.png - Cluster size distribution")

if __name__ == "__main__":
    main()