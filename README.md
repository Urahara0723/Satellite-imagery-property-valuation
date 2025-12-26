# Multimodal Property Valuation: Selective Fusion of Satellite Imagery and Tabular Data

**Author:** Mohit Trivedi, IIT Roorkee  
**Project:** Integration of structured housing attributes with high-resolution satellite imagery via Relevance-Gated Fusion.

## 1. Overview
Traditional hedonic pricing models often fail to quantify **"curb appeal"** and environmental context. This project bridges the **"modality gap"** by using a Relevance-Gated Convolutional Neural Network (CNN) architecture. 

Rather than using naive concatenation, visual data is selectively fused only when it provides a complementary signal (e.g., waterfronts, green belts). This approach effectively suppresses visual noise in ambiguous contexts, ensuring the visual branch provides **Conditional Dominance** only where relevant.

---

## 2. Key Features
* **Selective Fusion:** Employs a sigmoid-based gating mechanism to weight visual features dynamically.
* **Frozen Backbone:** Utilizes a pretrained **EfficientNet-B0** as a fixed feature extractor to prevent overfitting.
* **Explainability:** Integrated **Grad-CAM** analysis to visualize environmental factors (vegetation, water, roads) influencing the price prediction.
* **High-Performance Baseline:** Established a near-optimal XGBoost baseline ($R^2 \approx 0.90$) prior to visual augmentation.

---

## 3. Methodology
The pipeline processes data through two parallel streams before merging them based on semantic relevance:

### Tabular Stream
* **Preprocessing:** Cleaning of structured attributes (sqft, grade, lat/long).
* **Engineering:** Calculation of house age, renovation flags, and structural utility ratios.

### Visual Stream
* **Acquisition:** Programmatic retrieval of **ESRI World Imagery** (Zoom Level 19) via `data_fetcher.py`.
* **Feature Extraction:** Compression of EfficientNet-B0 activations into compact descriptors (Mean, Std, Max, Min).
* **Semantic Indicators:** Computer vision-based extraction of Green Ratio, Water Ratio, and Edge Density.

### Relevance Gating
To address noise in "average" images, a visual gate modulates the visual features:
$$Gate = \sigma(W \cdot [F_{semantic}] + b)$$
$$Prediction = f_{XGBoost}(F_{tabular}, Gate \cdot F_{visual})$$

---

## 4. Results
The multimodal model demonstrates significant error reduction in visually distinctive segments, such as high-greenery areas ("privacy premium") and waterfront properties.

| Model Architecture | RMSE (Log) | $R^2$ Score |
| :--- | :--- | :--- |
| Tabular Baseline | 0.163368 | 0.899251 |
| **Multimodal (Gated)** | **0.162990** | **0.899717** |

---

## 5. Repository Structure
| File | Description |
| :--- | :--- |
| `tabular_baseline.ipynb` | Evaluation of traditional models (Linear Reg, KNN, RF, XGBoost). |
| `data_fetcher.py` | Script for programmatic acquisition of ESRI satellite imagery. |
| `enlarged_cnn_feature_extraction.ipynb` | CNN activation compression and semantic feature engineering. |
| `enlarged_final_multimodal_model.ipynb` | The core pipeline implementing Relevance Gating and final fusion. |
| `explainability_gradcam.ipynb` | Qualitative validation of focus regions using Grad-CAM. |
| `Property_Valuation_Report.pdf` | Formal research paper detailing the methodology and results. |

---
