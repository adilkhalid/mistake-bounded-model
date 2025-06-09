# train.py

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from perceptron_mistake_bounded.model import PerceptronMistakeBounded
from perceptron_mistake_bounded.utils import evaluate_model, plot_decision_boundary
import numpy as np

# Step 1: Generate data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           class_sep=2.0, random_state=42)

# Convert labels from 0/1 → -1/1
y = np.where(y == 0, -1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the model
model = PerceptronMistakeBounded(learning_rate=1.0, max_epochs=100)
model.fit(X_train, y_train)

# Step 3: Evaluate
evaluate_model(model, X_test, y_test)

# Step 4: Visualize
plot_decision_boundary(model, X_test, y_test)
