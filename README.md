# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks are computational models inspired by the human brain, designed to recognize patterns and relationships in data. In this experiment, we develop a neural network regression model to predict output values based on given inputs. The model consists of one input neuron, two hidden layers with 10 neurons each, and one output neuron. The hidden layers use the ReLU activation function, introducing non-linearity to capture complex patterns. The final layer outputs a continuous value, making the model suitable for regression tasks.

The model is trained using the Mean Squared Error (MSE) loss function, which measures the difference between predicted and actual values. The RMSprop optimizer updates the weights through backpropagation, minimizing the loss over multiple epochs. The dataset is preprocessed using Min-Max Scaling to improve training efficiency.

## Neural Network Model
![Screenshot 2025-02-24 114031](https://github.com/user-attachments/assets/5b97064f-a475-41aa-a636-ce850657bc1e)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:DHARANYA.N
### Register Number:212223230044
```python
class NeuralNet(nn.Module):
  def __init__ (self):
        super().__init__()  # Changed _init_ to __init__
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 12)
        self.fc3 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x= self.relu(self.fc2(x))
        x= self.fc3(x) # No activation here since it's a regression task
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim. RMSprop(ai_brain.parameters(), lr=0.001)

def train_model (ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion (ai_brain (X_train), y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

![Screenshot 2025-02-24 114016](https://github.com/user-attachments/assets/efaa95e5-3bf2-44d5-9e44-3de42421c502)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-03 111131](https://github.com/user-attachments/assets/401bb0cc-04a6-43d6-8870-3434a2d43f29)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/59bb4840-3d67-44ce-8d30-6c6c3f681581)


## RESULT
The trained neural network regression model showed effective learning with a decreasing loss trend and accurate predictions.
