import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Random data generate
np.random.seed(0)  # Same result every time
X = 2 * np.random.rand(100, 1)  # 100 random numbers between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Step 2: Train model
model = LinearRegression()
model.fit(X, y)

# Step 3: Predict
X_new = np.array([[0], [2]])  # Predict for x=0 to x=2
y_pred = model.predict(X_new)

# Step 4: Plot
plt.scatter(X, y, color='blue', label='Random Data')
plt.plot(X_new, y_pred, color='red', linewidth=2, label='Prediction Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression on Random Data')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Output coefficients
print(f"Predicted model: y = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f}x")
