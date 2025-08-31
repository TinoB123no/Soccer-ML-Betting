# Soccer Match Outcome Prediction (ML Betting Algorithm)
Designed experiments to evaluate accuracy, profit-based metrics, and interpretability across models.


This project explores different machine learning approaches to predicting soccer match outcomes and simulating betting strategies.

## Overview
- **Goal**: Predict outcomes of professional soccer matches (win/loss/draw) and explore profitability of different betting strategies.
- **Data**: Historical match data engineered into 700+ features (team form, expected goals (xG), table position, head-to-head results).
- **Methods**:
  - **XGBoost**: Strong baseline, ~75% accuracy.
  - **Neural Networks**: Deep feedforward models (TensorFlow) with dropout, nonlinear activations.
  - **Reinforcement Learning**: Policy gradient approach optimizing betting returns directly.
- **Outcome**:
  - XGBoost → best raw prediction accuracy.
  - Neural Nets → explored nonlinear structure, moderate performance (~65%).
  - RL Agent → maximized profitability, even when accuracy was lower.

## Repository Structure
src/
xgboost_model.py # Baseline tree model
neural_net.py # Deep learning model
reinforcement_learning.py# RL betting agent
images/ # Screenshots of code experiments

## Lessons Learned
- Feature engineering matters as much as model selection.
- High accuracy ≠ high profit; betting requires different evaluation metrics.
- Comparing model families gave deeper insight than optimizing one.

## Next Steps
- Add visuals (ROC, profit curves) when data is rebuilt.
- Package models into a simple API or Streamlit app.
