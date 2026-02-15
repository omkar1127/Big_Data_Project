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

## 🚀 Getting Started

### Prerequisites
```bash
- Python 3.8+
- Apache Kafka
- PySpark
- Flask/Streamlit (for UI)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cab-fare-prediction.git
cd cab-fare-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kafka:
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka Server
bin/kafka-server-start.sh config/server.properties
```

## 📁 Project Structure

```
cab-fare-prediction/
├── data/
│   ├── cab_rides.csv
│   └── ride_features.csv
├── preprocessing/
│   ├── data_cleaning.py
│   └── feature_engineering.py
├── models/
│   ├── train_model.py
│   └── saved_models/
├── kafka/
│   ├── producer.py
│   └── consumer.py
├── ui/
│   └── app.py
└── README.md
```

## 🔄 Workflow

### Training Pipeline
1. Load raw data from `cab_rides.csv`
2. Clean and preprocess data
3. Engineer features
4. Build ML pipeline (Indexer → Encoder → VectorAssembler → Scaler)
5. Train Linear Regression model
6. Evaluate model performance
7. Save pipeline and model

### Streaming Pipeline
1. Producer reads `ride_features.csv`
2. Sends data to `cab_price_features` Kafka topic
3. Consumer subscribes to topic
4. Loads trained models
5. Processes incoming streams
6. Generates real-time predictions
7. Outputs results

### User Interface
1. User enters ride details in form
2. Input is processed and transformed
3. Loaded models predict fare
4. Result is displayed to user

## 📈 Model Performance

The model is evaluated using:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

## 🛠️ Technologies Used

- **Apache Spark**: Data processing and ML pipeline
- **Apache Kafka**: Real-time streaming
- **Python**: Core programming language
- **Scikit-learn/PySpark MLlib**: Machine learning
- **Flask/Streamlit**: User interface

## 📝 Usage

### Run Training Pipeline
```bash
python models/train_model.py
```

### Start Kafka Producer
```bash
python kafka/producer.py
```

### Start Kafka Consumer
```bash
python kafka/consumer.py
```

### Launch User Interface
```bash
python ui/app.py
```

## 🔮 Future Enhancements

- [ ] Add support for multiple ML algorithms
- [ ] Implement model versioning
- [ ] Add real-time monitoring dashboard
- [ ] Integrate with cloud services (AWS/GCP/Azure)
- [ ] Implement A/B testing framework
- [ ] Add data validation and drift detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

Your Name - [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Apache Kafka documentation
- PySpark ML documentation
- Open source community
