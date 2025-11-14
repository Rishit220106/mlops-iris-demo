from pycaret.classification import setup, compare_models
from pycaret.datasets import get_data
import joblib

# Load Iris dataset
iris = get_data('iris')
print(iris.head())

# Setup the PyCaret environment
exp = setup(data=iris, target='species', session_id=42, verbose=False, html=False, log_experiment=False)

# Compare and select the best model
best_model = compare_models()
print(best_model)

# Save the best model using joblib
joblib.dump(best_model, 'iris_model.pkl')
print("Saved iris_model.pkl")
