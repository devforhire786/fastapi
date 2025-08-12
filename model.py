# model.py
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from config import settings

class ModelHandler:
    """Handles loading the ML model and making predictions."""
    model = None

    def load_model(self):
        """Loads the model from the path specified in the config."""
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            print(f"❌ Model file not found at: {model_path}")
            # Create a dummy model if none exists, for demonstration purposes
            self.create_dummy_model(model_path)
            return

        print(f"Loading model from: {model_path}...")
        if model_path.suffix == ".joblib":
            self.model = joblib.load(model_path)
        elif model_path.suffix == ".h5":
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model file type: {model_path.suffix}")

        print("✅ Model loaded successfully.")

    def predict(self, input_data: np.ndarray) -> float:
        """Makes a prediction using the loaded model."""
        if self.model is None:
            print("⚠️ Model not loaded. Returning a default value.")
            return 0.0 # Default prediction if model isn't loaded

        # The model's predict method might return a complex structure,
        # so we extract the first element of the first prediction.
        prediction = self.model.predict(input_data)
        return float(prediction[0])

    def create_dummy_model(self, path: Path):
        """Creates and saves a dummy scikit-learn model."""
        print("Creating a dummy model for demonstration...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        path.parent.mkdir(parents=True, exist_ok=True)

        X, y = make_classification(n_samples=10, n_features=7, random_state=42)
        dummy_clf = RandomForestClassifier(random_state=42)
        dummy_clf.fit(X, y)

        joblib.dump(dummy_clf, path)
        print(f"✅ Dummy model saved to {path}. Please restart the server.")
        self.model = dummy_clf


# Create a single instance
model_handler = ModelHandler()
