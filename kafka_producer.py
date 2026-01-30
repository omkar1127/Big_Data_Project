from kafka import KafkaProducer
import pandas as pd
import json
import time

# ----------------------------------
# Kafka Producer
# ----------------------------------
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

topic = "cab_price_features"

# ----------------------------------
# Read file with multiple rows
# ----------------------------------
df = pd.read_csv("ride_features.csv")

print(f"📂 Loaded {len(df)} rows from file")

# ----------------------------------
# Send each row as one Kafka message
# ----------------------------------
for idx, row in df.iterrows():
    message = row.to_dict()

    producer.send(topic, value=message)
    print(f"📤 Sent row {idx+1}: {message}")

    time.sleep(1)  # simulate streaming delay

producer.flush()
print("✅ All rows sent successfully")
