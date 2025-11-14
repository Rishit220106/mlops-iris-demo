from pycaret.classification import *
from pycaret.datasets import get_data

# Load dataset
iris = get_data('iris')

# Setup PyCaret
exp = setup(iris, target='species', session_id=42, silent=True, html=False)

# Train & finalize model
model = compare_models()
final = finalize_model(model)

# Save model
save_model(final, 'iris_model')  # creates iris_model.pkl
