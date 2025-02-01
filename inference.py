import pickle
import numpy as np

# Load the saved model
with open('temperature_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a new grid (or load a different one)
new_grid = np.random.random((5, 5, 5)) * 100  # Example new grid with random temperatures

# Flatten the grid and make a prediction
new_input = new_grid.flatten().reshape(1, -1)  # Flatten and reshape the input
prediction = model.predict(new_input)

print(f"Predicted average temperature: {prediction[0]:.2f}°C")
