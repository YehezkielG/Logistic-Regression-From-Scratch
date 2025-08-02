import numpy as np

# y = wx_1 + wx_2 + ... wx_n + w0 
class Model():
    def __init__(self):
        """
        Initializes the model parameters.
        - intercept_ (w₀): The bias or y-intercept.
        - coef_ (w₁...wₙ): The weights for each feature.
        """
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X_features, y_target):
        """
        Trains the model with the given data.
        
        Parameters:
        X_features (np.array): The feature matrix (training data).
        y_target (np.array): The target vector (answers).
        """
        # --- Using the Normal Equation formula: w = (Xᵀ * X)⁻¹ * Xᵀ * y ---
        X_with_bias = np.insert(X_features, 0, 1, axis=1)
        X_transpose = X_with_bias.T
        weights = np.linalg.inv(X_transpose @ X_with_bias) @ X_transpose @ y_target
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        # Returning all weights can be useful for debugging
    
    def predict(self, X_new):
        """
        Makes predictions for new data.

        Parameters:
        X_new (np.array): New data to be predicted.
        
        Returns:
        np.array: The prediction results.
        """
        return np.dot(X_new, self.coef_) + self.intercept_
    
    def show_equation(self):
        """
        Displays the trained linear regression equation.
        """
        if self.coef_ is None:
            print("Model has not been trained yet. Run .fit() first.")
            return

        equation_str = f"y ≈ {self.intercept_:.2f}"
        for i, weight in enumerate(self.coef_):
            equation_str += f" + ({weight:.2f} * x_{i+1})"
        
        return equation_str