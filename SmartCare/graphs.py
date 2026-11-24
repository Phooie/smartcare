#!/usr/bin/env python
"""
Participant Performance Visualization
Generates separate graphs for Precision, Recall, Accuracy, and F1-Score
for each participant based on CSV evaluation data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import ast

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

LABELS = ['hand_only', 'mouth_only', 'both_no_pill', 'pill_intake', 'nothing']

# ============================================================
# HARDCODE YOUR CSV FILES HERE
# ============================================================
CSV_FILES = [
    # 'evaluation_results_20260723_171445.csv',
    # 'evaluation_results_20260723_174629.csv',
    # 'ndaone.csv',
    # 'ndatwo.csv',
    # 'evaluation_results_20260723_145514.csv',
    # 'evaluation_results_20260723_182345.csv',
    # 'evaluation_results_20260723_183156.csv',
    'phoo.csv',  # Your aggregate data
    'ndaone.csv',  # Participant 1
    'cso.csv',  # Participant 2
    'cst.csv',  # Participant 3
    'aeint.csv',  # Participant 4
]

# Custom names for participants (optional)
PARTICIPANT_NAMES = {
    'phoo.csv': 'Participant 1',
    'ndaone.csv': 'Participant 2',
    'cso.csv': 'Participant 3',
    'cst.csv': 'Participant 4',
    'aeint.csv': 'Participant 5',
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
    
    # Per-label metrics
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
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    return {
        'name': participant_name,
        'total_frames': total,
        'correct': correct_count,
        'accuracy': accuracy,
        'metrics': metrics
    }

def generate_participant_graphs(results):
    """Generate separate graphs for each metric across participants"""
    
    if not results:
        print("❌ No results to plot")
        return
    
    # Create output directory
    output_dir = "participant_metrics_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    participants = [r['name'] for r in results]
    
    # Per-label metrics for each participant
    metrics_data = {label: {'precision': [], 'recall': [], 'f1': []} for label in LABELS}
    
    for result in results:
        for label in LABELS:
            if label in result['metrics']:
                metrics_data[label]['precision'].append(result['metrics'][label]['precision'])
                metrics_data[label]['recall'].append(result['metrics'][label]['recall'])
                metrics_data[label]['f1'].append(result['metrics'][label]['f1'])
            else:
                metrics_data[label]['precision'].append(0)
                metrics_data[label]['recall'].append(0)
                metrics_data[label]['f1'].append(0)
    
    # Accuracy data
    accuracies = [r['accuracy'] for r in results]
    
    # Colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    label_display = [l.replace('_', ' ').title() for l in LABELS]
    
    # 1. PRECISION GRAPH
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(participants))
    width = 0.15
    multiplier = 0
    
    for i, label in enumerate(LABELS):
        offset = width * multiplier
        ax.bar(x + offset, metrics_data[label]['precision'], width, 
               label=label_display[i], color=colors[i])
        multiplier += 1
    
    ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Participants', fontsize=12, fontweight='bold')
    ax.set_title('Precision by Participant and Label', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_by_participant.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/precision_by_participant.png")
    
    # 2. RECALL GRAPH
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(participants))
    multiplier = 0
    
    for i, label in enumerate(LABELS):
        offset = width * multiplier
        ax.bar(x + offset, metrics_data[label]['recall'], width, 
               label=label_display[i], color=colors[i])
        multiplier += 1
    
    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Participants', fontsize=12, fontweight='bold')
    ax.set_title('Recall by Participant and Label', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recall_by_participant.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/recall_by_participant.png")
    
    # 3. F1-SCORE GRAPH
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(participants))
    multiplier = 0
    
    for i, label in enumerate(LABELS):
        offset = width * multiplier
        ax.bar(x + offset, metrics_data[label]['f1'], width, 
               label=label_display[i], color=colors[i])
        multiplier += 1
    
    ax.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Participants', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score by Participant and Label', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_by_participant.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/f1_by_participant.png")
    
    # 4. ACCURACY GRAPH
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(participants, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                                                    '#FF9FF3', '#54A0FF', '#5F27CD'])
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Participants', fontsize=12, fontweight='bold')
    ax.set_title('Overall Accuracy by Participant', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add horizontal line for average accuracy
    avg_accuracy = np.mean(accuracies)
    ax.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7, 
               label=f'Average Accuracy: {avg_accuracy:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_participant.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/accuracy_by_participant.png")
    
    # 5. HEATMAP
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    precision_data = []
    recall_data = []
    f1_data = []
    
    for result in results:
        precision_data.append([result['metrics'][label]['precision'] for label in LABELS])
        recall_data.append([result['metrics'][label]['recall'] for label in LABELS])
        f1_data.append([result['metrics'][label]['f1'] for label in LABELS])
    
    # Precision heatmap
    sns.heatmap(precision_data, ax=axes[0, 0], annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=label_display, yticklabels=participants, 
                cbar_kws={'label': 'Precision (%)'}, vmin=0, vmax=100)
    axes[0, 0].set_title('Precision by Participant and Label', fontsize=12, fontweight='bold')
    
    # Recall heatmap
    sns.heatmap(recall_data, ax=axes[0, 1], annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=label_display, yticklabels=participants, 
                cbar_kws={'label': 'Recall (%)'}, vmin=0, vmax=100)
    axes[0, 1].set_title('Recall by Participant and Label', fontsize=12, fontweight='bold')
    
    # F1 heatmap
    sns.heatmap(f1_data, ax=axes[1, 0], annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=label_display, yticklabels=participants, 
                cbar_kws={'label': 'F1-Score (%)'}, vmin=0, vmax=100)
    axes[1, 0].set_title('F1-Score by Participant and Label', fontsize=12, fontweight='bold')
    
    # Accuracy bar chart
    axes[1, 1].bar(participants, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=10)
    axes[1, 1].set_title('Overall Accuracy by Participant', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(axes[1, 1].patches, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/all_metrics_heatmap.png")
    
    # 6. SUMMARY TABLE
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Participant', 'Total Frames', 'Accuracy (%)', 
               'Pill Intake\nPrecision (%)', 'Pill Intake\nRecall (%)', 'Pill Intake\nF1 (%)']
    
    for result in results:
        pill_metrics = result['metrics'].get('pill_intake', {})
        table_data.append([
            result['name'],
            result['total_frames'],
            f"{result['accuracy']:.1f}",
            f"{pill_metrics.get('precision', 0):.1f}",
            f"{pill_metrics.get('recall', 0):.1f}",
            f"{pill_metrics.get('f1', 0):.1f}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colColours=['#4472C4']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            if j == 0:
                continue
            if j >= 2:
                try:
                    val_float = float(val)
                    if val_float >= 90:
                        color = '#C6EFCE'
                    elif val_float >= 70:
                        color = '#FFEB9C'
                    else:
                        color = '#FFC7CE'
                    table[(i+1, j)].set_facecolor(color)
                except:
                    pass
    
    ax.set_title('Participant Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/performance_summary_table.png")
    
    print(f"\n📊 All graphs saved to: {output_dir}/")
    print("   - precision_by_participant.png")
    print("   - recall_by_participant.png")
    print("   - f1_by_participant.png")
    print("   - accuracy_by_participant.png")
    print("   - all_metrics_heatmap.png")
    print("   - performance_summary_table.png")

def main():
    """Main function to process hardcoded CSV files and generate graphs"""
    
    print("=" * 70)
    print("📊 PARTICIPANT PERFORMANCE VISUALIZATION")
    print("=" * 70)
    print("Processing hardcoded CSV files...")
    print("=" * 70)
    
    # Process each file
    results = []
    for file in CSV_FILES:
        df = load_and_parse_csv(file)
        if df is not None and len(df) > 0:
            # Use custom name if available, otherwise use filename
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
    
    print(f"\n📈 Generating graphs for {len(results)} participants...")
    generate_participant_graphs(results)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()