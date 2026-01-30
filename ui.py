# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

import plotly.graph_objects as go
import plotly.express as px

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel


# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(
    page_title="🚕 Cab Price Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# SPARK SESSION
# =============================================================================
@st.cache_resource
def create_spark_session():
    """Create and cache Spark session"""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName("CabPricePredictionUI")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


# =============================================================================
# LOAD MODELS
# =============================================================================
@st.cache_resource
def load_models(_spark):
    """Load preprocessing pipeline and trained model"""
    try:
        preprocessing_pipeline = PipelineModel.load("models/preprocessing_pipeline")
        lr_model = LinearRegressionModel.load("models/linear_regression_model")
        return preprocessing_pipeline, lr_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure models are trained and saved in 'models/' directory")
        return None, None


# =============================================================================
# LOAD ANALYTICS DATA
# =============================================================================
@st.cache_data
def load_analytics_data():
    """Load and process data for analytics dashboard"""
    try:
        spark_temp = create_spark_session()
        
        # Load CSV
        df = spark_temp.read.csv("cab_rides.csv", header=True, inferSchema=True)
        
        # Clean data
        df = df.filter(
            (F.col("price").isNotNull()) & 
            (F.col("price") > 0) & 
            (F.col("distance") > 0)
        )
        
        # Extract datetime features
        df = df.withColumn(
            "timestamp_converted",
            (F.col("time_stamp") / 1000).cast("timestamp")
        )
        df = df.withColumn("hour", F.hour("timestamp_converted"))
        df = df.withColumn("day_of_week", F.dayofweek("timestamp_converted"))
        
        # Hourly statistics
        hourly_stats = df.groupBy("hour").agg(
            F.avg("price").alias("avg_price"),
            F.count("*").alias("ride_count")
        ).orderBy("hour").toPandas()
        
        # Daily statistics
        daily_stats = df.groupBy("day_of_week").agg(
            F.avg("price").alias("avg_price")
        ).orderBy("day_of_week").toPandas()
        
        # Map day numbers to names
        day_mapping = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
        daily_stats["day_name"] = daily_stats["day_of_week"].map(day_mapping)
        
        # Service-wise statistics
        service_stats = df.groupBy("cab_type", "name").agg(
            F.avg("price").alias("avg_price")
        ).toPandas()
        
        # Distance vs Price (sample for performance)
        distance_price = df.select("distance", "price").sample(
            fraction=0.1, seed=42
        ).limit(1000).toPandas()
        
        # Surge statistics
        surge_stats = df.groupBy("surge_multiplier").agg(
            F.avg("price").alias("avg_price"),
            F.count("*").alias("count")
        ).orderBy("surge_multiplier").toPandas()
        
        return {
            "hourly": hourly_stats,
            "daily": daily_stats,
            "service": service_stats,
            "distance_price": distance_price,
            "surge": surge_stats
        }
    
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        return None


# =============================================================================
# FEATURE ENGINEERING (SAME AS TRAINING)
# =============================================================================
def feature_engineering(df):
    """Apply feature engineering to match training pipeline"""
    df = df.withColumn(
        "time_of_day",
        F.when(F.col("hour").between(6, 11), "morning")
         .when(F.col("hour").between(12, 16), "afternoon")
         .when(F.col("hour").between(17, 20), "evening")
         .otherwise("night")
    )

    df = df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
    )

    df = df.withColumn(
        "is_surge",
        F.when(F.col("surge_multiplier") > 1.0, 1).otherwise(0)
    )

    df = df.withColumn(
        "distance_category",
        F.when(F.col("distance") < 1, "short")
         .when(F.col("distance") < 3, "medium")
         .otherwise("long")
    )

    return df


# =============================================================================
# INIT
# =============================================================================
spark = create_spark_session()
pipeline_model, lr_model = load_models(spark)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚕 Cab Ride Price Prediction System</div>', 
            unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🚕 Price Prediction",
    "📊 Analytics Dashboard",
    "💡 Pricing Insights"
])


# =============================================================================
# TAB 1: PRICE PREDICTION
# =============================================================================
with tab1:
    st.markdown("### 🚕 Predict Cab Fare Using ML Model")
    
    if pipeline_model is None or lr_model is None:
        st.warning("⚠️ Models not loaded. Please train the model first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🚗 Ride Details")
            cab_type = st.selectbox("Cab Type", ["Uber", "Lyft"])
            
            # Service types based on cab type
            if cab_type == "Uber":
                service_options = ["UberX", "UberXL", "UberPool", "Black", "Black SUV"]
            else:
                service_options = ["Lyft", "Lyft XL", "Lux", "Lux Black", "Shared"]
            
            name = st.selectbox("Service Type", service_options)
            source = st.text_input("Source Location", "Boston University")
            destination = st.text_input("Destination Location", "Fenway")

        with col2:
            st.markdown("#### 📍 Trip Parameters")
            distance = st.number_input(
                "Distance (miles)", 
                min_value=0.1, 
                max_value=50.0,
                value=5.0,
                step=0.1
            )
            surge_multiplier = st.slider(
                "Surge Multiplier", 
                1.0, 3.0, 1.0, 0.25,
                help="Higher during peak hours or high demand"
            )
            
            st.markdown("#### 🕐 Time Details")
            col2a, col2b = st.columns(2)
            with col2a:
                hour = st.slider("Hour of Day", 0, 23, 10)
                month = st.slider("Month", 1, 12, 6)
            with col2b:
                day_of_week = st.slider("Day of Week", 1, 7, 3, 
                                       help="1=Sunday, 7=Saturday")

        st.markdown("---")
        
        if st.button("🚀 Predict Price", type="primary", use_container_width=True):
            with st.spinner("Calculating price..."):
                try:
                    # Prepare input data
                    input_data = [{
                        "cab_type": cab_type,
                        "name": name,
                        "source": source,
                        "destination": destination,
                        "distance": float(distance),
                        "surge_multiplier": float(surge_multiplier),
                        "hour": int(hour),
                        "day_of_week": int(day_of_week),
                        "month": int(month)
                    }]

                    pdf = pd.DataFrame(input_data)

                    # Method 1: Use to_dict() - Most compatible
                    sdf = spark.createDataFrame(pdf.to_dict(orient='records'))

                    # Feature engineering
                    sdf = feature_engineering(sdf)

                    # Transform and predict
                    transformed = pipeline_model.transform(sdf)
                    prediction = (
                        lr_model
                        .transform(transformed)
                        .select("prediction")
                        .collect()[0][0]
                    )

                    # Display result
                    st.success(f"### 💰 Estimated Cab Price: **${prediction:.2f}**")
                    
                    # Additional insights
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Base Price", f"${prediction/surge_multiplier:.2f}")
                    with col2:
                        st.metric("Surge Multiplier", f"{surge_multiplier}x")
                    with col3:
                        price_per_mile = prediction / distance
                        st.metric("Price per Mile", f"${price_per_mile:.2f}")
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")


# =============================================================================
# TAB 2: ANALYTICS DASHBOARD
# =============================================================================
with tab2:
    st.markdown("### 📊 Real-time Analytics Dashboard")
    
    # Load analytics data
    analytics_data = load_analytics_data()
    
    if analytics_data is None:
        st.warning("⚠️ Unable to load analytics data. Please ensure 'cab_rides.csv' exists.")
    else:
        # Display summary metrics
        st.markdown("#### 📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_rides = analytics_data["hourly"]["ride_count"].sum()
            st.metric("Total Rides", f"{total_rides:,}")
        
        with col2:
            avg_price = analytics_data["hourly"]["avg_price"].mean()
            st.metric("Avg Price", f"${avg_price:.2f}")
        
        with col3:
            max_price = analytics_data["hourly"]["avg_price"].max()
            st.metric("Peak Hour Price", f"${max_price:.2f}")
        
        with col4:
            peak_hour = analytics_data["hourly"].loc[
                analytics_data["hourly"]["avg_price"].idxmax(), "hour"
            ]
            st.metric("Peak Hour", f"{int(peak_hour)}:00")
        
        st.markdown("---")
        
        # Row 1: Hourly trends
        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=analytics_data["hourly"]["hour"],
                y=analytics_data["hourly"]["avg_price"],
                mode="lines+markers",
                fill="tozeroy",
                name="Avg Price",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8)
            ))
            fig1.update_layout(
                title="Average Price by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Price ($)",
                height=400,
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=analytics_data["hourly"]["hour"],
                y=analytics_data["hourly"]["ride_count"],
                marker_color="#ff7f0e"
            ))
            fig2.update_layout(
                title="Ride Volume by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Rides",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # Row 2: Daily and Service comparison
        col1, col2 = st.columns(2)

        with col1:
            fig3 = px.bar(
                analytics_data["daily"],
                x="day_name",
                y="avg_price",
                title="Average Price by Day of Week",
                labels={"day_name": "Day", "avg_price": "Avg Price ($)"},
                color="avg_price",
                color_continuous_scale="Blues"
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            # Top 10 services by price
            top_services = analytics_data["service"].nlargest(10, "avg_price")
            
            fig4 = px.bar(
                top_services,
                x="name",
                y="avg_price",
                color="cab_type",
                barmode="group",
                title="Top 10 Services by Average Price",
                labels={"name": "Service", "avg_price": "Avg Price ($)", "cab_type": "Provider"}
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # Row 3: Distance correlation and Surge analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📏 Distance vs Price Correlation")
            fig5 = px.scatter(
                analytics_data["distance_price"],
                x="distance",
                y="price",
                trendline="ols",
                title="Distance vs Price Relationship",
                labels={"distance": "Distance (miles)", "price": "Price ($)"},
                opacity=0.6
            )
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚡ Surge Pricing Impact")
            fig6 = px.bar(
                analytics_data["surge"],
                x="surge_multiplier",
                y="avg_price",
                title="Average Price by Surge Multiplier",
                labels={"surge_multiplier": "Surge Multiplier", "avg_price": "Avg Price ($)"},
                text="count",
                color="avg_price",
                color_continuous_scale="Reds"
            )
            fig6.update_traces(texttemplate='%{text} rides', textposition='outside')
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)


# =============================================================================
# TAB 3: PRICING INSIGHTS
# =============================================================================
with tab3:
    st.markdown("### 💡 Smart Pricing Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        #### ✅ Best Times to Ride (Lower Prices)
        - **Early Morning** (2 AM - 6 AM)
        - **Mid-day** (10 AM - 3 PM)
        - **Mid-week** (Tuesday - Thursday)
        - **Off-peak hours** (Avoid 7-9 AM, 5-7 PM)
        """)
        
        st.info("""
        #### 💡 Money-Saving Tips
        - Book rides 10-15 minutes in advance
        - Compare Uber vs Lyft prices
        - Use standard services instead of premium
        - Share rides when possible
        - Avoid weekend nights
        """)

    with col2:
        st.warning("""
        #### ⚠️ High Price Periods
        - **Morning Rush** (7 AM - 9 AM)
        - **Evening Rush** (5 PM - 7 PM)
        - **Weekend Nights** (Fri-Sat 9 PM - 2 AM)
        - **Bad Weather** (Rain, Snow)
        - **Events & Concerts**
        """)
        
        st.error("""
        #### 🚨 Surge Pricing Factors
        - High demand, low supply
        - Major events (sports, concerts)
        - Weather conditions
        - Holidays and special occasions
        - Airport rush hours
        """)

    st.markdown("---")

    # Interactive Scenario Comparison
    st.markdown("### 🧮 Scenario Price Comparison")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Set Base Parameters")
        scenario_distance = st.slider(
            "Distance (miles)", 
            1, 30, 10,
            key="scenario_dist"
        )
        base_rate = st.number_input(
            "Base Rate ($/mile)",
            1.0, 5.0, 2.5,
            step=0.1,
            key="base_rate"
        )
        base_fee = st.number_input(
            "Base Fee ($)",
            0.0, 10.0, 5.0,
            step=0.5,
            key="base_fee"
        )
    
    with col2:
        # Calculate prices for different scenarios
        scenarios = {
            "Early Morning (3 AM)": 0.8,
            "Morning Rush (8 AM)": 1.8,
            "Midday (12 PM)": 1.0,
            "Afternoon (3 PM)": 1.1,
            "Evening Rush (6 PM)": 2.0,
            "Night (10 PM)": 1.3,
            "Weekend Morning": 0.9,
            "Weekend Night": 1.7
        }

        base_price = base_fee + (scenario_distance * base_rate)

        scenario_df = pd.DataFrame([
            {
                "Scenario": scenario,
                "Multiplier": f"{multiplier}x",
                "Estimated Price": base_price * multiplier
            }
            for scenario, multiplier in scenarios.items()
        ])

        # Sort by price
        scenario_df = scenario_df.sort_values("Estimated Price")

        fig_scenario = px.bar(
            scenario_df,
            x="Scenario",
            y="Estimated Price",
            text="Estimated Price",
            title=f"Price Comparison for {scenario_distance} mile Trip",
            color="Estimated Price",
            color_continuous_scale="RdYlGn_r"
        )
        fig_scenario.update_traces(
            texttemplate="$%{text:.2f}",
            textposition="outside"
        )
        fig_scenario.update_layout(
            xaxis_tickangle=-45,
            height=450
        )
        st.plotly_chart(fig_scenario, use_container_width=True)

    # Display detailed table
    st.markdown("#### 📋 Detailed Scenario Breakdown")
    scenario_df_display = scenario_df.copy()
    scenario_df_display["Estimated Price"] = scenario_df_display["Estimated Price"].apply(
        lambda x: f"${x:.2f}"
    )
    
    # Add savings column
    max_price = scenario_df["Estimated Price"].max()
    scenario_df["Savings"] = max_price - scenario_df["Estimated Price"]
    scenario_df_display["Savings vs Peak"] = scenario_df["Savings"].apply(
        lambda x: f"${x:.2f}" if x > 0 else "-"
    )
    
    st.dataframe(
        scenario_df_display,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    
    # Additional insights
    st.markdown("### 🎯 Personalized Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🏢 For Commuters
        - Set up recurring rides during off-peak
        - Use ride-share options
        - Consider monthly passes
        - Plan 30 min before rush hour
        """)
    
    with col2:
        st.markdown("""
        #### ✈️ For Airport Trips
        - Book 24 hours in advance
        - Avoid early morning flights
        - Compare airport shuttles
        - Check public transit options
        """)
    
    with col3:
        st.markdown("""
        #### 🎉 For Events
        - Arrive/leave before crowd
        - Walk 2-3 blocks away
        - Compare prices frequently
        - Consider ride-sharing with friends
        """)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🚕 Cab Price Prediction System | Built with PySpark & Streamlit</p>
        <p>Data-driven insights for smarter ride decisions</p>
    </div>
""", unsafe_allow_html=True)