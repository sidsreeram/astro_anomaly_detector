import pandas as pd
import numpy as np

df = pd.read_csv('outputs/reports/anomaly_scores.csv')

print("=== DATASET ===")
print("Total tiles:", len(df))
print("Anomalies:", int(df['Is_Anomaly'].sum()))
print("Anomaly rate:", round(100*df['Is_Anomaly'].mean(), 2), "%")

print("\n=== ANOMALY SCORE ===")
print("Min:", round(df['Anomaly_Score'].min(), 6))
print("Max:", round(df['Anomaly_Score'].max(), 6))
print("Mean:", round(df['Anomaly_Score'].mean(), 6))
print("Std:", round(df['Anomaly_Score'].std(), 6))
print("Threshold 95pct:", round(np.percentile(df['Anomaly_Score'], 95), 6))

print("\n=== MSE ===")
print("Mean:", round(df['MSE'].mean(), 6))
print("Max:", round(df['MSE'].max(), 6))
print("Std:", round(df['MSE'].std(), 6))

print("\n=== SSIM DELTA ===")
print("Mean:", round(df['SSIM_Delta'].mean(), 6))
print("Max:", round(df['SSIM_Delta'].max(), 6))

print("\n=== TOP 10 ANOMALIES ===")
top10 = df.head(10)
for _, r in top10.iterrows():
    print(f"  {r['Filename']:<30} Score={r['Anomaly_Score']:.5f}  MSE={r['MSE']:.5f}  SSIM={r['SSIM_Delta']:.5f}")
