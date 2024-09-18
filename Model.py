import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, HillClimbSearch

# Load the dataset from CSV file
data_frame = pd.read_csv('ann-test.data')

# Data preprocessing
# Creating a target variable from binary indicators
data_frame['target_variable'] = data_frame[['query_hypothyroid', 'query_hyperthyroid', 'TSH_measured']].idxmax(axis=1)

# Map target labels to numeric values
data_frame['target_variable'] = data_frame['target_variable'].map({
    'query_hypothyroid': 0,
    'query_hyperthyroid': 1,
    'TSH_measured': 2
})

# Define feature columns and target column
feature_columns = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
target_column = 'target_variable'

# Split the data into training and testing sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(data_frame[feature_columns], data_frame[target_column], test_size=0.2, random_state=42)

# Create a DataFrame for training data
training_data = X_train_set.copy()
training_data[target_column] = y_train_set

# Initialize the Bayesian Network model
bayes_net_model = BayesianNetwork()

# Apply Hill Climb Search to determine the best structure
search_algorithm = HillClimbSearch(training_data, scoring_method=K2Score(training_data))
best_structure = search_algorithm.estimate()

# Set the model structure and fit the parameters
bayes_net_model.add_edges_from(best_structure.edges())
bayes_net_model.fit(training_data, estimator=MaximumLikelihoodEstimator)

# Make predictions on the test set
predictions = []
for index in range(len(X_test_set)):
    prediction = bayes_net_model.predict(X_test_set.iloc[index:index+1])
    predictions.append(prediction[target_column].values[0])

# Evaluate the model accuracy
model_accuracy = accuracy_score(y_test_set, predictions)
print(f'Model Accuracy: {model_accuracy * 100:.2f}%')

# Check if the accuracy meets the expected threshold
if model_accuracy >= 0.85:
    print("Expected accuracy is achieved.")
else:
    print("Expected accuracy is not achieved.")