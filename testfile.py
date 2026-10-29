"""Demo script for the model registry and training API.

Author: Leonardo M. Ferrisi
Date: 2026-10-29
"""

import numpy as np

from unrbl_ml_pipeline import models
from unrbl_ml_pipeline.training import train

# --- Explore the registry ---
print("Available models:", models.list_models())

info = models.get("examplemodel")
print("Name:", info.name)
print("Description:", info.description)

# --- Classification: 3-class problem on random data ---
rng = np.random.RandomState(42)
x_train = rng.randn(200, 32).astype(np.float32)
y_train = rng.randint(0, 3, size=200)

print("\n--- Classification ---")
model, history = train(
    models.ExampleModel,
    x_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
)
print("Final accuracy:", history.history["accuracy"][-1])
print("Output shape:", model.output_shape)

# --- Regression: predict 2 continuous outputs ---
y_reg = rng.randn(200, 2).astype(np.float32)

print("\n--- Regression ---")
model_reg, hist_reg = train(
    "examplemodel",
    x_train,
    y_reg,
    task="regression",
    epochs=10,
    validation_split=0.2,
)
print("Final MAE:", hist_reg.history["mae"][-1])
print("Output shape:", model_reg.output_shape)

# --- Different input shape: 2D time-series (samples, timesteps, channels) ---
x_ts = rng.randn(200, 64, 8).astype(np.float32)
y_ts = rng.randint(0, 5, size=200)

print("\n--- 2D input (time-series) ---")
model_ts, _ = train("examplemodel", x_ts, y_ts, epochs=5, validation_split=0.0)
print("Output shape:", model_ts.output_shape)
