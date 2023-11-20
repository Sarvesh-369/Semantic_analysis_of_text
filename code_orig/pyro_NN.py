import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from tqdm.auto import trange
from pyro.infer import Predictive

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
# Define the Bayesian Neural Network
class Model(PyroModule):
    def __init__(self, input_size, h1=20, h2=20):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_size, h1)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([h1, input_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([h1]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([h2, h1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([h2]).to_event(1))
        self.fc3 = PyroModule[nn.Linear](h2, 1)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([1, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc3(x).squeeze(-1)
        # sigma = pyro.sample("sigma", dist.Uniform(0., 10.))  # Changed to a larger range
        # with pyro.plate("data", x.shape[0]):
        #     obs = pyro.sample("obs", dist.Bernoulli(logits=mu), obs=y)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(logits=mu), obs=y)
        return mu

# Instantiate the model with the correct input size
input_size = X_tensor.shape[1]
model = Model(input_size)

# Setup the guide and optimizer
guide = AutoDiagonalNormal(model)
adam = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

# The training loop
pyro.clear_param_store()
num_epochs = 10  

for epoch in trange(num_epochs):
    loss = svi.step(X_tensor, y_tensor)
    if epoch % 1 == 0:
        print(f"Epoch {epoch}: loss = {loss / (X_tensor.shape[0]*1000):.3f}")


param_store_path = 'models/model_params.pyro'  # Replace with your path
pyro.get_param_store().save(param_store_path)

# Load the test data
test_data = pd.read_csv('tests/test.csv')

# Assuming that the 'preprocess' function and 'vectorizer' are already defined and fitted on the training data
test_data['question1'] = test_data['question1'].astype(str).apply(preprocess)
test_data['question2'] = test_data['question2'].astype(str).apply(preprocess)

# # Vectorize the questions
X1_test = vectorizer.transform(test_data['question1'])
X2_test = vectorizer.transform(test_data['question2'])

# Use the difference in TF-IDF vectors as features
X_diff_test = X1_test - X2_test

# Convert to PyTorch tensors
X_tensor_test = torch.tensor(X_diff_test.toarray(), dtype=torch.float)

print(X_tensor_test.shape)
print(f"Input tensor shape: {X_tensor_test.shape}")

# Assuming that 'model' and 'guide' are already defined and trained
try:
    predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("obs", "_RETURN"))
    samples = predictive(X_tensor_test)
    yhat = samples["obs"].mean(0)  # Take the mean over all samples

    # Convert predictions to binary labels
    y_pred = (yhat > 0.5).int().numpy()
except RuntimeError as e:
    print("A runtime error occurred:")
    print(e)
    # Inspect the shape of the parameters
    for name, param in model.named_parameters():
        print(f"Shape of {name}: {param.shape}")
    raise
predictions_df = pd.DataFrame({
    'question1': test_data['question1'],
    'question2': test_data['question2'],
    'predictions': y_pred
})

# Save the DataFrame to a CSV file
predictions_df.to_csv('results/Pyro_NN_predictions.csv', index=False)

print("Predictions have been saved to Bayesian_NN_predictions.csv.")