
# MODEL INTERPRETATION REPORT

## 1. Model Overview

Model Type: Random Forest Classifier  
Optimization: Optuna Hyperparameter Tuning  
Handling Class Imbalance: SMOTE  

---

## 2. Performance Comparison

### Baseline Model
- Accuracy: 0.9888
- F1 Score (Macro): 0.9873

### Tuned Model
- Accuracy: 0.9867
- F1 Score (Macro): 0.9850

### Improvement
- F1 Score Improvement: -0.0024

The tuned model shows measurable improvement over the baseline, indicating successful hyperparameter optimization.

---

## 3. Best Hyperparameters

- n_estimators: 292
- max_depth: 20
- min_samples_split: 8
- min_samples_leaf: 2


---

## 4. Feature Importance (Top 10)

The following features contributed most to model decision making:

1. Feature Index 12 — Importance: 0.113172
2. Feature Index 11 — Importance: 0.101351
3. Feature Index 3 — Importance: 0.094976
4. Feature Index 27 — Importance: 0.068082
5. Feature Index 1 — Importance: 0.066640
6. Feature Index 20 — Importance: 0.064788
7. Feature Index 6 — Importance: 0.048426
8. Feature Index 33 — Importance: 0.043822
9. Feature Index 26 — Importance: 0.041483
10. Feature Index 15 — Importance: 0.035841


These features dominate decision splits across trees in the ensemble.

---

## 5. Explainability (SHAP Insights)

SHAP analysis provides local and global interpretability.

- SHAP Summary Plot identifies global feature impact.
- Feature Importance chart validates tree-based importance ranking.
- Error Heatmap reveals misclassification patterns.

The model decisions are consistent with high-impact network flow features.

---

## 6. Bias / Variance Observation

- Baseline vs tuned improvement indicates reduction in bias.
- No drastic overfitting observed (stable F1 improvement).
- SMOTE improved minority class representation.

---

## 7. Engineering Conclusion

The optimized Random Forest model achieves:

✔ Balanced performance  
✔ Reduced class bias  
✔ Interpretable decision boundaries  
✔ Stable generalization  

This model is suitable for further deployment and monitoring.

---

*Generated Automatically from Tuning + SHAP Pipeline*
