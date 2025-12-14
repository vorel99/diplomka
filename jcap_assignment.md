

# GeoScore Framework for Predicting Municipal Risks in Germany

Geographic location is a strong predictor of many social, economic, and environmental outcomes. In Germany, a wide range of open data is available at the municipality level (approximately 11,000 municipalities), covering aspects such as demographics, economic structure, infrastructure, and environment. However, these datasets are often heterogeneous, incomplete, and vary in quality.

The concept of a GeoScore is to transform such heterogeneous, location-based open data into a unified feature representation that can be used by machine learning models to predict negative target variables, for example unemployment rate, crime rate, school dropout rate, or business insolvency.

This diploma assignment focuses on building a generalizable framework that ingests open geospatial and statistical data for German municipalities and predicts an arbitrary negative target variable defined at the municipal level.

The goals of this diploma assignment are the following:
1.	Identify and collect relevant open data for German municipalities from reliable public sources.
2.	Preprocess data, including spatial aggregation, smoothing, normalization, and handling of missing values.
3.	Design a reusable GeoScore framework that transforms location-based open data into a structured feature representation.
4.	Train and evaluate machine learning models using a clear train/validation/test data split.
5.	Demonstrate the framework on at least one negative target variable at the municipality level.
6.	Analyze and interpret model results, including spatial patterns and feature importance.

