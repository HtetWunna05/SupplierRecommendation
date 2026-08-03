# Feature Engineering Report

## 1. Cost Efficiency
- **Calculation:** [Quantity] * [Unit_Price]
- **Value:** Allows the system to identify high-spend areas and compare supplier pricing fairly.

## 2. Quality Impact Score
- **Calculation:** 100 - (Quality_Severity * 5)
- **Value:** Converts "Incidents" (text) into a "Score" (number). This makes it possible to mathematically rank suppliers.

## 3. Consistency Index
- **Calculation:** [Reliability_Score] / [Avg_Lead_Time]
- **Value:** This is the most important feature. It rewards suppliers who are both fast (low lead time) AND reliable. High consistency means lower risk for the supply chain.