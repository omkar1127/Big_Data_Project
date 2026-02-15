# 🚖 Cab Fare Prediction System

An end-to-end Machine Learning pipeline for predicting cab fares using batch training and real-time streaming with Kafka.

---

## 📌 Project Overview

This project builds a **Cab Fare Prediction System** that:

- Cleans and preprocesses raw cab ride data
- Trains a Linear Regression model
- Evaluates performance (RMSE, MAE, R²)
- Serves predictions via:
  - User Interface (batch prediction)
  - Kafka Streaming (real-time prediction)

---

# 🏗️ Architecture Overview

The system consists of the following components:

INPUT → PREPROCESSING → ML MODEL → SAVED MODEL
↓
USER INTERFACE & KAFKA PIPELINE


---

# 📂 Project Pipeline

## 1️⃣ Input Layer

- Raw dataset: `cab_rides.csv`

---

## 2️⃣ Data Preprocessing

### 🔹 Data Cleaning
- Remove null values
- Filter invalid data

### 🔹 Feature Engineering
- Datetime extraction
- Derived features
- Encoding categorical variables

### 🔹 Pipeline Creation
Includes:
- Indexer
- Encoder
- Vector Assembler
- Scaler

---

## 3️⃣ Machine Learning Model

### 🔹 Model Training
- Algorithm: **Linear Regression**

### 🔹 Model Evaluation
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

### 🔹 Model Saving
- Saved pipeline
- Saved trained model

---

# 🚀 Deployment & Serving

## 🖥️ User Interface Flow

User Input Form → Load Models → Process Input → Predict Fare → Display Result


Allows users to manually input ride details and receive predicted fare.

---

## 🔄 Kafka Streaming Pipeline

### 🔹 Kafka Producer
- Reads: `ride_features.csv`
- Sends features to Kafka topic: `cab_price_features`

### 🔹 Kafka Broker
- Manages topic and streaming data

### 🔹 Kafka Consumer
1. Read Kafka stream
2. Parse JSON
3. Load saved model
4. Process stream
5. Predict prices
6. Output results

Enables real-time fare prediction.

---

# 🧠 Tech Stack

- Python
- Apache Spark (ML Pipeline)
- Linear Regression
- Apache Kafka
- JSON Streaming
- Scikit-learn / Spark ML (depending on implementation)

---

# 📊 Model Metrics

| Metric | Description |
|--------|-------------|
| RMSE   | Measures prediction error magnitude |
| MAE    | Average absolute error |
| R²     | Variance explained by the model |

---

📌 Features
✅ End-to-end ML pipeline
✅ Feature engineering automation
✅ Real-time streaming prediction
✅ Batch and streaming support
✅ Modular architecture

# 🛠️ How to Run

## 1️⃣ Train Model
```bash
python train_model.py
2️⃣ Start Kafka
zookeeper-server-start.sh config/zookeeper.properties
kafka-server-start.sh config/server.properties
3️⃣ Start Producer
python kafka_producer.py
4️⃣ Start Consumer
python kafka_consumer.py
5️⃣ Run UI
python app.py
