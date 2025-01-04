from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Define the Random Model
class RandomModel(nn.Module):
    def __init__(self):
        super(RandomModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # Input size: 10, Output size: 64
        self.fc2 = nn.Linear(64, 32)  # Input size: 64, Output size: 32
        self.fc3 = nn.Linear(32, 1)   # Input size: 32, Output size: 1
        self.sigmoid = nn.Sigmoid()   # Sigmoid activation for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Sigmoid output layer
        return x

# Initialize the model and load trained weights
model = RandomModel()
model.load_state_dict(torch.load('random_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    input_data = np.array(data['input']).astype(np.float32)
    input_tensor = torch.tensor(input_data).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
