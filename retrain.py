from pycaret.classification import *
from pycaret.datasets import get_data

# Load dataset
iris = get_data('iris')

# Setup PyCaret (minimal arguments to support older versions)
exp = setup(data=iris, target='species', session_id=42)

# Train & finalize best model
model = compare_models()
final = finalize_model(model)

# Save model
save_model(final, 'iris_model')
