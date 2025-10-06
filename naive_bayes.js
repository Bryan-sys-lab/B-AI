// NaiveBayesClassifier class for prediction using Gaussian Naive Bayes
// This implementation assumes continuous features with Gaussian distribution
class NaiveBayesClassifier {
  // Constructor initializes the model data structures
  constructor() {
    this.classes = []; // list of unique class labels
    this.data = {}; // object to store training data grouped by class
  }

  // Fit method: stores the training data
  // X: array of feature vectors
  // y: array of corresponding class labels
  fit(X, y) {
    for (let i = 0; i < X.length; i++) {
      const features = X[i];
      const label = y[i];

      // Initialize data structure for new class
      if (!this.data[label]) {
        this.data[label] = [];
        this.classes.push(label);
      }

      // Store the feature vector for this class
      this.data[label].push(features);
    }
  }

  // Calculate probability density for a single feature value
  // Assumes Gaussian (normal) distribution
  // x: feature value
  // mean: mean of the distribution
  // variance: variance of the distribution
  // Returns the probability density
  calculateProbability(x, mean, variance) {
    const exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
    return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
  }

  // Calculate the probability of features given a class
  // Assumes feature independence (Naive Bayes assumption)
  // features: array of feature values
  // label: class label
  // Returns the likelihood probability
  calculateClassProbability(features, label) {
    let probability = 1;
    const classData = this.data[label];
    const numInstances = classData.length;

    // Calculate mean and variance for each feature in this class
    for (let i = 0; i < features.length; i++) {
      let sum = 0;
      for (let j = 0; j < numInstances; j++) {
        sum += classData[j][i];
      }
      const mean = sum / numInstances;

      let variance = 0;
      for (let j = 0; j < numInstances; j++) {
        variance += Math.pow(classData[j][i] - mean, 2);
      }
      variance /= numInstances;

      // Multiply probabilities (independence assumption)
      probability *= this.calculateProbability(features[i], mean, variance);
    }

    return probability;
  }

  // Predict method: returns the most likely class for given features
  // features: array of feature values
  // Returns the predicted class label
  predict(features) {
    let maxProbability = -1;
    let bestLabel = null;

    // Evaluate probability for each class
    for (let i = 0; i < this.classes.length; i++) {
      const label = this.classes[i];
      const probability = this.calculateClassProbability(features, label);

      if (probability > maxProbability) {
        maxProbability = probability;
        bestLabel = label;
      }
    }

    return bestLabel;
  }
}

// Export the class for use in other modules
module.exports = NaiveBayesClassifier;