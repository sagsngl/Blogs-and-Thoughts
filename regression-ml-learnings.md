# From Control Systems to Autonomous Vehicles: My First Steps into Regression Models

## Introduction

My journey into machine learning began with a clear destination in mind: the autonomous vehicle industry. As a diagnostics engineer with a background in control systems, I spent months wondering how to bridge my engineering expertise into the AI/ML space. Three weeks ago, I decided to stop wondering and start learning. Today, I'm sharing what I've discovered about regression models—the fundamental building blocks of predictive machine learning—through my eyes as someone who thinks more in differential equations than neural networks.

The path I'm following isn't linear. I'm using what I call a "pincer movement": learning foundational concepts through Google's ML Crash Course while simultaneously diving into advanced control method publications. Each reinforces the other. The more I practice, the more the papers make sense. The papers reveal what I need to learn next.

This post captures my early findings about regression models and the lessons I'm learning as an engineer entering the ML world.

## Understanding the Regression Landscape

### Linear Regression: Where It All Begins

Linear regression is the foundation—a straight line fit through noisy data. Simple in concept, profound in its implications.

The basic formula:

```
ŷ = mx + b
```

Where we're solving for the best `m` (slope) and `b` (intercept) that minimize the distance between our predicted line and actual data points.

In practice, you'll encounter **multivariate linear regression** with many features:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example: predicting rice grain classification features
X = np.array([[area_1, perimeter_1, major_axis_1, ...],
              [area_2, perimeter_2, major_axis_2, ...],
              ...])
y = np.array([0, 1, 0, ...])  # 0 for Osmancik, 1 for Cammeo

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
```

### Logistic Regression: Crossing the Classification Threshold

Despite its name, logistic regression is actually for **classification**, not regression. This confused me initially—it's not predicting continuous values but rather probabilities bounded between 0 and 1.

The magic lies in the sigmoid function that squashes any input into a probability:

```
P(y=1) = 1 / (1 + e^(-z))
```

Where `z` is the linear combination of features.

```python
from sklearn.linear_model import LogisticRegression

# Same rice dataset, now with proper classification
model = LogisticRegression()
model.fit(X, y)
probabilities = model.predict_proba(X)  # Returns probabilities for each class
predictions = model.predict(X)  # Returns 0 or 1
```

### Polynomial Regression: When Lines Aren't Enough

Not all relationships are linear. Polynomial regression fits curves through data by adding polynomial features:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features (squares, interactions, etc.)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
```

The key insight: polynomial regression is still linear regression under the hood—just with transformed features. The algorithm doesn't know it's fitting a curve; it's solving a linear problem in a higher-dimensional space.

## The Real Lesson: Performance Metrics Are Your Truth-Tellers

Here's what surprised me most during my first week: **accuracy alone is dangerously misleading**.

I built a model that achieved 95% accuracy on the rice classification task and felt confident. Then I learned something uncomfortable: if I simply predicted "Osmancik" for every single grain, I'd get roughly 50% accuracy with zero intelligence. The metric didn't reveal this because I wasn't looking at the right ones.

### Precision and Recall: The Trade-off Nobody Tells You About

**Precision** answers: "Of the grains I labeled as Cammeo, how many actually were Cammeo?"

**Recall** answers: "Of all the actual Cammeo grains, how many did I find?"

These metrics expose a fundamental truth in machine learning: you can't optimize for everything simultaneously. You can build a model that finds every Cammeo grain but mislabels many Osmancik grains as Cammeo (high recall, low precision). Conversely, you can build one that's extremely conservative, only labeling something as Cammeo when it's absolutely sure (high precision, low recall).

```python
from sklearn.metrics import precision_score, recall_score, accuracy_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.3f}")   # Overall correctness
print(f"Precision: {precision:.3f}")  # Reliability of positive predictions
print(f"Recall: {recall:.3f}")        # Coverage of actual positives
```

### F1 Score: Harmonizing the Conflict

This is where F1 score became my friend. It's the harmonic mean of precision and recall—a single number that refuses to let you cheat at just one metric.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Unlike a simple average, the harmonic mean heavily penalizes extreme imbalance. A model with 99% precision and 1% recall gets an F1 score around 0.02, not 0.50. This forces a genuine balance.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.3f}")
```

**My personal insight**: F1 makes mathematical sense immediately, but intuition comes with use. I'm still in that phase where I calculate it and think "yes, that's the right answer" without *feeling* the number. I expect another week or two of practice will change that.

### ROC and AUC: Seeing Across Thresholds

This was the breakthrough concept for me. Most classification models don't output just 0 or 1; they output probabilities. We then choose a threshold—typically 0.5—to convert probabilities to class predictions.

But what if 0.5 isn't optimal for your problem?

The ROC curve shows your model's performance across *all possible thresholds*. The AUC (Area Under the Curve) is a single number representing the overall quality—essentially the probability that your model ranks a random positive example higher than a random negative example.

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

y_pred_proba = model.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
auc_score = auc(fpr, tpr)

print(f"AUC Score: {auc_score:.3f}")
```

For the rice dataset, an AUC of 0.95+ indicates excellent class separation. An AUC of 0.50 means your model is no better than random guessing.

## The Three Key Findings

### 1. Hyperparameters Are Landmines You Need to Navigate

Learning rates, regularization strengths, polynomial degrees—these parameters control whether your model converges to a good solution or gets stuck in a local minimum, or diverges entirely.

My first attempt at logistic regression used a learning rate of 0.1 for 1,000 iterations. The model converged, but not to a good solution. When I reduced it to 0.001 over 5,000 iterations, performance jumped significantly.

**The Problem**: There are no universal rules. Every dataset is different.

**The Question That Drove Me**: Isn't there a way to automatically find the best hyperparameters?

Yes—and this opened a whole new rabbit hole of research into **hyperparameter optimization**.

### 2. Automated Hyperparameter Tuning: Closing the Loop

The short answer to my question is yes, but it's computationally expensive. The main approaches are:

- **Grid Search**: Test every combination in a defined grid. Thorough but slow.
- **Random Search**: Test random combinations. Faster but less systematic.
- **Bayesian Optimization**: Build a probabilistic model of performance and intelligently choose what to test next. The sweet spot for many problems.
- **Hyperband/ASHA**: Modern algorithms that prune low-performing configurations early, using computational resources more efficiently.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'max_iter': [100, 500, 1000]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validated score: {grid_search.best_score_:.3f}")
```

This fascinated me as a control systems engineer. We've essentially created a feedback loop where the machine learns not just *how* to classify rice, but *how to learn* to classify rice. It's meta-learning at its simplest form.

### 3. Multidimensional Data Reveals Patterns You Can't See

The rice dataset has seven features per grain: area, perimeter, major axis length, minor axis length, eccentricity, convex area, and extent. My mechanical engineering background trained me to think in 2D or 3D visualizations. Seven dimensions is impossible to visualize directly, yet patterns exist in that space that perfectly separate the two rice varieties.

This was humbling and enlightening. Linear regression, logistic regression, and their variants find these patterns computationally by optimizing loss functions. The patterns don't exist on a graph I can draw; they exist as numerical solutions to optimization problems.

This reframes what "understanding" a problem means in machine learning. It's not visual intuition—it's mathematical proof through convergence.

## Practical Example: Classifying Rice with Logistic Regression

Here's the complete workflow from the Google Colab notebook I used, condensed and annotated:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load rice dataset (Cammeo and Osmancik)
df = pd.read_csv('rice.csv')
X = df.drop('Class', axis=1)
y = df['Class'].map({'Cammeo': 1, 'Osmancik': 0})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (crucial for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Comprehensive evaluation
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

The results on the rice dataset typically show F1 scores around 0.95+, meaning the model reliably identifies both varieties with few mistakes.

## What I Still Don't Fully Understand (Yet)

**Performance metrics feel mathematical but not intuitive.** I can calculate F1 score and understand why it's the harmonic mean, but I don't yet *feel* what a 0.87 F1 score means in practice. This is a timing issue—with more practice, these numbers will develop intuitive meaning. This is analogous to how a control engineer develops intuition for system stability over time; it's not instant.

**Feature engineering remains mysterious.** The rice dataset gives us morphological features directly. In real-world problems, raw data (images, text, sensor streams) requires transformation into meaningful features. This is where domain knowledge meets machine learning, and it's my next frontier.

## Key Takeaways for Someone Starting This Journey

1. **Start with linear and logistic regression.** They're not "simple"—they're foundational. Understanding them deeply will accelerate your learning of everything that comes after.

2. **Embrace multiple metrics.** Accuracy alone will lie to you. Learn precision, recall, F1, ROC/AUC. Different problems require different trade-offs, and these metrics help you navigate them.

3. **Hyperparameters matter tremendously.** Don't just use defaults. Experiment systematically (grid search, random search) to understand how parameters affect your model.

4. **Your domain expertise is an asset.** As an engineer, my understanding of systems, convergence, and optimization isn't a liability in ML—it's a foundation. Your background matters.

5. **Learning happens at the intersection of theory and practice.** Reading papers without coding leaves you with abstractions. Coding without understanding papers leaves you copying without learning. Do both simultaneously, even if it feels inefficient.

## What's Next

I'm already looking ahead. The obvious next steps are:

- **Feature engineering and selection**: How do you transform raw data into features that matter?
- **Regularization and overfitting**: How do you prevent your model from memorizing instead of learning?
- **Advanced algorithms**: Decision trees, random forests, SVMs, neural networks. Where does regression fit in this landscape?
- **Real datasets**: The rice classification is a beautiful, clean example. What happens with messy, real-world data?

And beyond these technical questions, the deeper one: How do these techniques apply to autonomous vehicle perception and control? That's where this journey is headed.

## Final Thoughts

Machine learning is becoming my second language in engineering. Like learning any language, there's an awkward phase where you're thinking in formulas and precision/recall before understanding feels natural. But I can already see the power of these tools—the ability to find patterns in multidimensional spaces that visual intuition can never reach.

For anyone with a technical background considering this transition: your foundation in mathematics, systems thinking, and optimization will serve you well. Machine learning is applied mathematics with feedback loops. We do that in control systems; this is just a different domain.

The next time you see a sensor reading, a dataset, or a pile of measurements, remember: hidden in that multidimensional space are patterns waiting to be discovered by regression models. Sometimes the patterns are simple lines. Sometimes they're complex curves. But they're there, and now we know how to find them.

---

**Have you started learning ML from an engineering background? I'd love to hear about your approach in the comments or via discussion.**