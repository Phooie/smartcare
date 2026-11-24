#!/usr/bin/env python
"""
Metrics Calculator - Reads existing CSV and calculates metrics
Output format matches the original accuracy test results
"""

import csv
import numpy as np
import os
import ast

LABELS = ['hand_only', 'mouth_only', 'both_no_pill', 'pill_intake', 'nothing']

def calculate_metrics_from_csv(csv_file):
    """Calculate all metrics from a CSV file"""
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    # Read CSV
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to proper types
            row['frame'] = int(row['frame'])
            row['hand_detected'] = row['hand_detected'].lower() == 'true' if row['hand_detected'] else False
            row['mouth_detected'] = row['mouth_detected'].lower() == 'true' if row['mouth_detected'] else False
            row['distance'] = float(row['distance']) if row['distance'] and row['distance'] != 'None' else None
            row['correct'] = row['correct']
            results.append(row)
    
    total = len(results)
    print(f"\n📁 Loaded {total} labeled frames from {csv_file}")
    
    # ===== CONFUSION MATRIX =====
    confusion = {label: {label2: 0 for label2 in LABELS} for label in LABELS}
    
    for r in results:
        user = r['label']
        system = r['system_label']
        confusion[user][system] = confusion[user].get(system, 0) + 1
    
    # ===== ACCURACY =====
    correct_count = sum(1 for r in results if r['correct'] == 'T')
    accuracy = correct_count / total * 100
    
    # ===== PER-LABEL METRICS =====
    metrics = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[l][label] for l in LABELS if l != label)
        fn = sum(confusion[label][l] for l in LABELS if l != label)
        tn = total - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # ===== DETECTION RATES =====
    total_frames = max([r['frame'] for r in results]) if results else 0
    hand_count = sum(1 for r in results if r['hand_detected'])
    mouth_count = sum(1 for r in results if r['mouth_detected'])
    hand_rate = hand_count / total * 100 if total > 0 else 0
    mouth_rate = mouth_count / total * 100 if total > 0 else 0
    
    # ===== DISTANCE STATS =====
    distances = [r['distance'] for r in results if r['distance'] is not None]
    avg_distance = sum(distances) / len(distances) if distances else 0
    min_distance = min(distances) if distances else 0
    max_distance = max(distances) if distances else 0
    
    # ===== PRINT RESULTS (EXACT FORMAT) =====
    print("\n" + "=" * 70)
    print("📊 ACCURACY TEST RESULTS")
    print("=" * 70)
    
    print(f"\n📈 OVERALL METRICS:")
    print(f"   Total Frames: {total}")
    print(f"   Labeled Frames: {total}")
    print(f"   Correct Predictions: {correct_count}/{total}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    print(f"\n📊 PER-LABEL METRICS:")
    for label in LABELS:
        m = metrics[label]
        print(f"\n   {label}:")
        print(f"      Precision: {m['precision']:.3f}")
        print(f"      Recall:    {m['recall']:.3f}")
        print(f"      F1-Score:  {m['f1']:.3f}")
    
    print(f"\n📊 CONFUSION MATRIX:")
    print("                   Actual (User)")
    print("                  H    M    B    P    N")
    print("   Predicted H   ", end="")
    for l in LABELS:
        print(f"{confusion['hand_only'][l]:>4}", end="")
    print()
    print("   (System)  M   ", end="")
    for l in LABELS:
        print(f"{confusion['mouth_only'][l]:>4}", end="")
    print()
    print("              B   ", end="")
    for l in LABELS:
        print(f"{confusion['both_no_pill'][l]:>4}", end="")
    print()
    print("              P   ", end="")
    for l in LABELS:
        print(f"{confusion['pill_intake'][l]:>4}", end="")
    print()
    print("              N   ", end="")
    for l in LABELS:
        print(f"{confusion['nothing'][l]:>4}", end="")
    print()
    print("   H=hand_only, M=mouth_only, B=both_no_pill, P=pill_intake, N=nothing")
    
    print(f"\n📊 DETECTION RATES:")
    print(f"   Hand Detected: {hand_count}/{total} ({hand_rate:.1f}%)")
    print(f"   Mouth Detected: {mouth_count}/{total} ({mouth_rate:.1f}%)")
    
    print(f"\n📊 DISTANCE STATS:")
    print(f"   Average Distance: {avg_distance:.1f} px")
    print(f"   Min Distance: {min_distance:.1f} px")
    print(f"   Max Distance: {max_distance:.1f} px")
    
    # ===== INTERPRETATION =====
    print(f"\n💡 INTERPRETATION:")
    if accuracy >= 90:
        print("   ✅ Excellent! System is highly accurate.")
    elif accuracy >= 75:
        print("   👍 Good. System is reliable with room for improvement.")
    elif accuracy >= 60:
        print("   ⚠️ Moderate. Consider improving lighting or camera position.")
    else:
        print("   ❌ Needs improvement. Check detection thresholds and lighting.")
    
    print("=" * 70)


def main():
    """Main function"""
    print("=" * 70)
    print("📊 METRICS CALCULATOR")
    print("=" * 70)
    print("This script calculates metrics from an existing CSV file.")
    print("=" * 70)
    
    # Ask for CSV file
    csv_file = input("\n📁 Enter CSV filename (e.g., evaluation_results_20260723_145514.csv): ")
    
    if not csv_file:
        # Try to find the most recent CSV
        csv_files = [f for f in os.listdir() if f.startswith('evaluation_results_') and f.endswith('.csv')]
        if csv_files:
            csv_files.sort(reverse=True)
            print(f"\n📁 Found CSV files:")
            for i, f in enumerate(csv_files[:5], 1):
                print(f"   {i}. {f}")
            choice = input(f"\nSelect a file (1-{min(5, len(csv_files))}) or press Enter to use '{csv_files[0]}': ")
            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    csv_file = csv_files[idx]
                else:
                    csv_file = csv_files[0]
            else:
                csv_file = csv_files[0]
        else:
            print("❌ No CSV files found in current directory")
            return
    
    calculate_metrics_from_csv(csv_file)


if __name__ == "__main__":
    main()