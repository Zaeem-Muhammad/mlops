import torch
import torch.nn as nn

class RandomModel(nn.Module):
    def __init__(self):
        super(RandomModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize and save the model
model = RandomModel()
torch.save(model.state_dict(), 'random_model.pth')
