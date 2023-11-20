import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm.auto import trange

# Load the training data
train_data = pd.read_csv('tests/train.csv')

# Function to preprocess text data
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Preprocess the questions
train_data['question1'] = train_data['question1'].astype(str).apply(preprocess)
train_data['question2'] = train_data['question2'].astype(str).apply(preprocess)

# Vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X1 = vectorizer.fit_transform(train_data['question1'])
X2 = vectorizer.transform(train_data['question2'])

# Use the difference in TF-IDF vectors as features
X_diff = X1 - X2
y = train_data['is_duplicate'].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_diff.toarray(), dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)
print(X_tensor.shape)

# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, h1=20, h2=20):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze(-1)

# Instantiate the model
input_size = X_tensor.shape[1]
model = NeuralNetwork(input_size)

# Define the loss function and optimizer
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create a DataLoader instance to handle batching
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

num_epochs = 10  # Define the number of epochs for which the model will be trained

# Training loop
for epoch in trange(num_epochs):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)/10
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss = {total_loss :.3f}")

torch.save(model.state_dict(), 'models/torch_model.pth')

print("The model's state dictionary has been saved to torch_model.pth.")

test_data = pd.read_csv('tests/test.csv')

# Preprocess and vectorize the test data
test_data['question1'] = test_data['question1'].astype(str).apply(preprocess)
test_data['question2'] = test_data['question2'].astype(str).apply(preprocess)
X1_test = vectorizer.transform(test_data['question1'])
X2_test = vectorizer.transform(test_data['question2'])
X_diff_test = X1_test - X2_test
X_tensor_test = torch.tensor(X_diff_test.toarray(), dtype=torch.float)

# Predict the labels
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor_test)
    y_pred_binary = (y_pred > 0.5).int().numpy()

predictions_df = pd.DataFrame({
    'question1': test_data['question1'],
    'question2': test_data['question2'],
    'predictions': y_pred_binary
})

# Save the DataFrame to a CSV file
predictions_df.to_csv('results/pytorch_NN_predictions.csv', index=False)

print("Predictions have been saved to NN_predictions.csv.")
