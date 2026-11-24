#!/usr/bin/env python
"""
Patient Overall Accuracy Visualization
Generates bar charts showing only overall accuracy for each patient
with large, readable labels for presentation/paper use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set global font sizes for better readability
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

LABELS = ['hand_only', 'mouth_only', 'both_no_pill', 'pill_intake', 'nothing']

# ============================================================
# HARDCODE YOUR CSV FILES HERE
# ============================================================
CSV_FILES = [
    'phoo.csv',
    'ndaone.csv',
    'cso.csv',
    'cst.csv',
    'aeint.csv',
]

# Custom names for participants
PARTICIPANT_NAMES = {
    'phoo.csv': 'Patient 1',
    'ndaone.csv': 'Patient 2',
    'cso.csv': 'Patient 3',
    'cst.csv': 'Patient 4',
    'aeint.csv': 'Patient 5',
}
# ============================================================

def load_and_parse_csv(csv_file):
    """Load CSV and parse data"""
    if not os.path.exists(csv_file):
        print(f"⚠️ File not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Clean data
    df['label'] = df['label'].astype(str).str.strip().replace('nan', np.nan)
    df['system_label'] = df['system_label'].astype(str).str.strip().replace('nan', np.nan)
    
    # Filter labeled frames
    df['is_labeled'] = df['system_label'].notna()
    labeled_df = df[df['is_labeled'] == True].copy()
    
    # Calculate correctness
    labeled_df['correct'] = labeled_df['label'] == labeled_df['system_label']
    
    return labeled_df

def calculate_metrics_for_participant(df, participant_name):
    """Calculate metrics for a specific participant/CSV"""
    
    total = len(df)
    if total == 0:
        return None
    
    # Overall accuracy
    correct_count = df['correct'].sum()
    accuracy = correct_count / total * 100
    
    # Confusion matrix
    confusion = {label: {label2: 0 for label2 in LABELS} for label in LABELS}
    
    for _, row in df.iterrows():
        user = row['label']
        system = row['system_label']
        if user in confusion and system in confusion[user]:
            confusion[user][system] = confusion[user].get(system, 0) + 1
    
    # Per-label metrics (just for reference)
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
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
        }
    
    return {
        'name': participant_name,
        'total_frames': total,
        'correct': correct_count,
        'accuracy': accuracy,
        'metrics': metrics
    }

def generate_accuracy_graphs(results):
    """Generate accuracy bar charts only"""
    
    if not results:
        print("❌ No results to plot")
        return
    
    # Create output directory
    output_dir = "accuracy_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    participants = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    total_frames = [r['total_frames'] for r in results]
    correct_counts = [r['correct'] for r in results]
    
    # Color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FF9FF3', '#54A0FF', '#5F27CD']
    
    # ============================================================
    # GRAPH 1: Simple Accuracy Bar Chart (Large Labels)
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(participants, accuracies, color=colors[:len(participants)], 
                  edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels on bars (large and bold)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=18, color='black')
    
    # Add total frames info below patient names
    for i, (name, frames, correct) in enumerate(zip(participants, total_frames, correct_counts)):
        ax.text(i, -5, f'n={frames}\n({correct}/{frames})', 
                ha='center', va='top', fontsize=11, color='gray')
    
    # Customize axes
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=18, fontweight='bold')
    ax.set_title('Overall Accuracy by Patient', fontsize=22, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.set_xticklabels(participants, fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    
    # Grid and reference lines
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y=90, color='blue', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Add average accuracy line
    avg_accuracy = np.mean(accuracies)
    ax.axhline(y=avg_accuracy, color='purple', linestyle='-', alpha=0.7, 
               linewidth=2, label=f'Average: {avg_accuracy:.1f}%')
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_patient.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/accuracy_by_patient.png")
    
    # ============================================================
    # GRAPH 2: Accuracy with Pill Intake F1 Overlay
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract pill intake F1 scores
    pill_f1 = []
    for result in results:
        if 'pill_intake' in result['metrics']:
            pill_f1.append(result['metrics']['pill_intake']['f1'])
        else:
            pill_f1.append(0)
    
    x = np.arange(len(participants))
    width = 0.35
    
    # Accuracy bars
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
                   color='#4472C4', edgecolor='black', linewidth=2)
    
    # Pill Intake F1 bars
    bars2 = ax.bar(x + width/2, pill_f1, width, label='Pill Intake F1-Score',
                   color='#FF6B6B', edgecolor='black', linewidth=2)
    
    # Value labels
    for bar, val in zip(bars1, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    for bar, val in zip(bars2, pill_f1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Score (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=18, fontweight='bold')
    ax.set_title('Accuracy vs Pill Intake F1-Score by Patient', fontsize=22, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_pill_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/accuracy_vs_pill_f1.png")
    
    # ============================================================
    # GRAPH 3: Horizontal Bar Chart (Good for papers)
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort by accuracy
    sorted_data = sorted(zip(participants, accuracies), key=lambda x: x[1])
    sorted_participants = [d[0] for d in sorted_data]
    sorted_accuracies = [d[1] for d in sorted_data]
    
    bars = ax.barh(sorted_participants, sorted_accuracies, 
                   color=colors[:len(sorted_participants)], 
                   edgecolor='black', linewidth=2, height=0.6)
    
    # Add value labels
    for bar, acc in zip(bars, sorted_accuracies):
        width_val = bar.get_width()
        ax.text(width_val + 1, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1f}%', ha='left', va='center', 
                fontweight='bold', fontsize=16)
    
    ax.set_xlabel('Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Patients', fontsize=18, fontweight='bold')
    ax.set_title('Overall Accuracy by Patient (Sorted)', fontsize=22, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add vertical reference lines
    ax.axvline(x=100, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=90, color='blue', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(x=80, color='orange', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(x=70, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Add average accuracy line
    avg_accuracy = np.mean(sorted_accuracies)
    ax.axvline(x=avg_accuracy, color='purple', linestyle='-', alpha=0.7, 
               linewidth=2, label=f'Average: {avg_accuracy:.1f}%')
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_patient_horizontal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/accuracy_by_patient_horizontal.png")
    
    # ============================================================
    # GRAPH 4: Grouped Bar Chart with all Metrics (Compact)
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract all metrics
    precision_values = []
    recall_values = []
    f1_values = []
    
    for result in results:
        metrics = result['metrics']
        # Average across all labels
        avg_precision = np.mean([metrics[label]['precision'] for label in LABELS])
        avg_recall = np.mean([metrics[label]['recall'] for label in LABELS])
        avg_f1 = np.mean([metrics[label]['f1'] for label in LABELS])
        precision_values.append(avg_precision)
        recall_values.append(avg_recall)
        f1_values.append(avg_f1)
    
    x = np.arange(len(participants))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_values, width, label='Avg Precision', 
                   color='#4ECDC4', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, recall_values, width, label='Avg Recall', 
                   color='#45B7D1', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, f1_values, width, label='Avg F1-Score', 
                   color='#96CEB4', edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x + width*1.5, accuracies, width, label='Accuracy', 
                   color='#FF6B6B', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=18, fontweight='bold')
    ax.set_title('Average Performance Metrics by Patient', fontsize=22, fontweight='bold', pad=20)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(participants, fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/all_metrics_comparison.png")
    
    # ============================================================
    # Print Summary
    # ============================================================
    print(f"\n📊 Accuracy Summary:")
    print("-" * 50)
    for name, acc, frames, correct in zip(participants, accuracies, total_frames, correct_counts):
        print(f"   {name}: {acc:.1f}% ({correct}/{frames} frames)")
    print("-" * 50)
    print(f"   Average Accuracy: {np.mean(accuracies):.1f}%")
    print(f"   Max Accuracy: {max(accuracies):.1f}% ({participants[np.argmax(accuracies)]})")
    print(f"   Min Accuracy: {min(accuracies):.1f}% ({participants[np.argmin(accuracies)]})")
    print("-" * 50)
    
    print(f"\n📊 All graphs saved to: {output_dir}/")
    print("   - accuracy_by_patient.png (Vertical bar chart)")
    print("   - accuracy_vs_pill_f1.png (Accuracy vs Pill Intake F1)")
    print("   - accuracy_by_patient_horizontal.png (Horizontal bar chart)")
    print("   - all_metrics_comparison.png (All metrics grouped)")

def main():
    """Main function to process hardcoded CSV files and generate graphs"""
    
    print("=" * 70)
    print("📊 PATIENT ACCURACY VISUALIZATION")
    print("=" * 70)
    print("Processing hardcoded CSV files...")
    print("=" * 70)
    
    # Process each file
    results = []
    for file in CSV_FILES:
        df = load_and_parse_csv(file)
        if df is not None and len(df) > 0:
            participant_name = PARTICIPANT_NAMES.get(file, file.replace('.csv', ''))
            result = calculate_metrics_for_participant(df, participant_name)
            if result:
                results.append(result)
                print(f"   ✅ {participant_name}: {len(df)} frames, {result['accuracy']:.1f}% accuracy")
        else:
            print(f"   ⚠️ Skipping {file} (no data)")
    
    if not results:
        print("❌ No valid results found!")
        return
    
    print(f"\n📈 Generating accuracy graphs for {len(results)} patients...")
    generate_accuracy_graphs(results)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()