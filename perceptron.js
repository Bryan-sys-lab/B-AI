// Perceptron class for binary classification
// This implementation uses the standard perceptron learning algorithm
class Perceptron {
  // Constructor initializes the model parameters
  // learningRate: controls how much weights are updated (default 0.01)
  // epochs: maximum number of training iterations (default 1000)
  constructor(learningRate = 0.01, epochs = 1000) {
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.weights = []; // weights for each feature
    this.bias = 0; // bias term
  }

  // Activation function: step function for binary output
  // Returns 1 if input >= 0, otherwise 0
  activate(x) {
    return x >= 0 ? 1 : 0;
  }

  // Predict method: computes the output for given inputs
  // inputs: array of feature values
  // Returns the predicted class (0 or 1)
  predict(inputs) {
    let sum = this.bias;
    for (let i = 0; i < inputs.length; i++) {
      sum += this.weights[i] * inputs[i];
    }
    return this.activate(sum);
  }

  // Train method: implements the perceptron learning rule
  // X: array of training samples (each sample is an array of features)
  // y: array of target labels (0 or 1)
  train(X, y) {
    // Initialize weights to zero
    this.weights = new Array(X[0].length).fill(0);
    this.bias = 0;

    // Iterate over epochs
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let errors = 0;

      // Iterate over each training sample
      for (let i = 0; i < X.length; i++) {
        const prediction = this.predict(X[i]);
        const error = y[i] - prediction;

        // Update weights and bias if prediction is wrong
        if (error !== 0) {
          errors++;
          this.bias += this.learningRate * error;
          for (let j = 0; j < this.weights.length; j++) {
            this.weights[j] += this.learningRate * error * X[i][j];
          }
        }
      }

      // Early stopping if converged (no errors)
      if (errors === 0) {
        console.log(`Training converged after ${epoch + 1} epochs`);
        break;
      }
    }
  }
}

// Export the class for use in other modules
module.exports = Perceptron;