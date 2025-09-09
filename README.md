# Hen-Network Group Project: Graph Regression for Productivity Prediction
Group Project for DL in Biomedical Applications Course @ FU Berlin

## Overview

The goal of this project is to predict the **productivity of chickens** within an Australian barn based on their **daily interactions**. We use a **Graph Neural Network (GNN)**, specifically a **Graph Attention Network (GAT)**, to model these interactions, where:

- **Nodes** represent chickens
- **Edges** represent interactions between them

The model is trained on one year of interaction data and used to predict productivity on new days.
- **Chicken Interaction Dataset** is kept on Kaggle (in private): https://kaggle.com/datasets/3ce959666cb1ea82c98d1138ce5a38bfac837982428764e93564d89f50029fdf
- **Productivity (Eggs) Dataset** is kept on Kaggle (in private): https://kaggle.com/datasets/6bc3b5a51ff4cb55720151132503e868668dfc8e32d6e2caf1ddd18032d2292c

---

## Key Concepts

### Target
- The target is the **productivity of chickens on a specific day** (e.g., egg-laying rate).
- This is a **continuous value** predicted by the model.

### Graph Representation
- **Nodes:** Individual chickens
- **Edges:** Daily interactions between chickens
- **Edge Features:** Time, location, and duration of each interaction

### Graph Regression
- **Input:** A graph built from interactions on one day
- **Output:** A single predicted value for that day's productivity

---

## Model Details

### Model Type
- **Graph Attention Network (GAT)**: Uses attention to focus on significant interactions in the graph

### Graph Construction
- Each **day = 1 graph**
- **Nodes = chickens**, **Edges = interactions (with features)**

### Feature Aggregation
- Node features are aggregated using the **GAT attention mechanism**
- Resulting graph-level representation is passed to **fully connected layers**
- Output is the **predicted productivity** for that day

### Training
- Trained on **1 year** of data
- Each graph has a corresponding **target productivity value**

### Prediction
- Input any new day's interaction graph → Output = predicted productivity

---

## 9-Step Project Plan

### 1. Upload Chicken Data Through Apache Spark
- Load the large dataset using pandas. We considered Apache Spark but it slowed down the daily regrouping later on.
- Verify schema and preview rows to understand data structure.

### 2. Rename Columns to Be Meaningful
- Rename columns to clear, consistent names such as:
  - `chicken1_id`, `chicken2_id`, `interaction_time`, `duration`, `location`, `date`, `productivity`

### 3. Clean the Data
- Handle missing values (drop or impute)
- Fix incorrect data types (e.g., convert timestamps, durations)
- Remove duplicates and invalid rows (e.g., self-loops or negative durations)
- Sanity check: Ensure each row represents one valid interaction

### 4. Group Data by Day to Build One Graph per Day
- For each unique date:
  - Collect all interactions from that day
  - Identify all unique chickens (nodes)
  - This forms the basis for one daily graph

### 5. Convert Each Day's Info into Graph Format
- For each day:
  - Create `edge_index` from interactions (source → target)
  - Encode edge features: `time`, `duration`, `location`
  - Create node features (use dummy features if none exist)
  - Assign the day’s productivity as the graph label (`y`)
  - Package into a graph object (e.g., using PyTorch Geometric's `Data`)

### 6. Create GAT Model for Graph Regression
- Build a Graph Attention Network (GAT) architecture:
  - Input: Daily graphs
  - Apply attention layers to learn from interactions
  - Aggregate node embeddings to form a graph-level embedding
  - Output: Predicted productivity value

### 7. Train the Model
- Split the data into training and validation sets
- Use Mean Squared Error (MSE) loss for regression
- Train the model on the daily graphs with productivity labels

### 8. Evaluate and Fine-Tune
- Evaluate model performance (e.g., MSE, MAE)
- Tune hyperparameters (e.g., number of GAT heads, hidden layers)
- Try different pooling strategies for graph-level representation

### 9. Build Deployment Function
- Write a function that:
  - Takes new daily interaction data
  - Builds a graph from it
  - Runs it through the trained model
  - Outputs predicted productivity for that day


---

## Type of Problem

- **Graph Regression**
- More specifically: **Graph-level Regression**, where each graph represents a day, and the output is one continuous value

---
