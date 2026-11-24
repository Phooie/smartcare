import csv
import numpy as np

# Load your CSV
with open('evaluation_results_20260629_095802.csv', 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# Count by label
labels = {}
for r in results:
    label = r['label']
    labels[label] = labels.get(label, 0) + 1

print("📊 Label Distribution:")
for label, count in labels.items():
    print(f"   {label}: {count}")

# Count hand/mouth detection
hand_count = sum(1 for r in results if r['hand_detected'] == 'True')
mouth_count = sum(1 for r in results if r['mouth_detected'] == 'True')

print(f"\n📈 Detection:")
print(f"   Hand detected: {hand_count}/{len(results)}")
print(f"   Mouth detected: {mouth_count}/{len(results)}")