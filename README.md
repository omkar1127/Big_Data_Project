# 🚗 Real-Time Uber Fare Prediction System

# Cab Fare Prediction System

A real-time machine learning system for predicting cab fares using Apache Kafka streaming and a user-friendly interface.

## 🏗️ Architecture Overview

This project implements an end-to-end ML pipeline with three main components:
- **Batch Processing Pipeline**: Data preprocessing and model training
- **Streaming Pipeline**: Real-time predictions using Kafka
- **User Interface**: Interactive fare prediction interface

## 📊 System Components

### 1. Data Preprocessing

#### Input
- **Raw Data**: `cab_rides.csv`

#### Processing Steps
- **Data Cleaning**
  - Remove null values
  - Handle duplicates
  
- **Feature Engineering**
  - DateTime extraction
  - Derived features creation
  - Encoding categorical variables

- **Pipeline Creation**
  - Indexer
  - Encoder
  - VectorAssembler
  - Scaler

### 2. ML Model

#### Model Training
- **Algorithm**: Linear Regression
- **Input**: Processed feature vectors
- **Output**: Trained model pipeline

#### Model Evaluation
Metrics used:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-Squared)

#### Saved Artifacts
- Pipeline model
- Linear Regression model

### 3. Kafka Streaming Pipeline

#### Kafka Producer
- Reads from: `ride_features.csv`
- Sends data to Kafka topic

#### Kafka Broker
- **Topic**: `cab_price_features`
- Manages message distribution

#### Kafka Consumer
- Reads from Kafka stream
- Parses JSON messages
- Loads trained models
- Processes streaming data
- Predicts prices in real-time
- Outputs results

### 4. User Interface

Interactive web interface for fare predictions:
1. **User Input Form** - Collects ride details
2. **Load Models** - Loads pre-trained models
3. **Process Input** - Transforms user input
4. **Predict Fare** - Generates fare prediction
5. **Display Result** - Shows predicted fare

