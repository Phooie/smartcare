#!/usr/bin/env python
"""
Participant Performance Visualization
Generates separate bar charts for each participant showing all 5 labels
for Precision, Recall, F1-Score, and Accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

LABELS = ['hand_only', 'mouth_only', 'both_no_pill', 'pill_intake', 'nothing']
LABEL_DISPLAY = ['Hand Only', 'Mouth Only', 'Both No Pill', 'Pill Intake', 'Nothing']
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

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
    """Generate separate bar charts for each participant"""
    
    if not results:
        print("❌ No results to plot")
        return
    
    # Create output directory
    output_dir = "participant_metrics_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate individual graphs for each participant
    for result in results:
        participant_name = result['name']
        metrics = result['metrics']
        accuracy = result['accuracy']
        total_frames = result['total_frames']
        
        # Prepare data
        precision_values = [metrics[label]['precision'] for label in LABELS]
        recall_values = [metrics[label]['recall'] for label in LABELS]
        f1_values = [metrics[label]['f1'] for label in LABELS]
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{participant_name} - Performance Metrics\n(Total Frames: {total_frames})', 
                     fontsize=16, fontweight='bold')
        
        x = np.arange(len(LABEL_DISPLAY))
        width = 0.6
        
        # 1. PRECISION BAR CHART
        bars1 = axes[0, 0].bar(x, precision_values, width, color=COLORS, edgecolor='black', linewidth=1)
        axes[0, 0].set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Precision by Label', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(LABEL_DISPLAY, rotation=45, ha='right', fontsize=10)
        axes[0, 0].set_ylim(0, 105)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars1, precision_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. RECALL BAR CHART
        bars2 = axes[0, 1].bar(x, recall_values, width, color=COLORS, edgecolor='black', linewidth=1)
        axes[0, 1].set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Recall by Label', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(LABEL_DISPLAY, rotation=45, ha='right', fontsize=10)
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        
        for bar, val in zip(bars2, recall_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. F1-SCORE BAR CHART
        bars3 = axes[1, 0].bar(x, f1_values, width, color=COLORS, edgecolor='black', linewidth=1)
        axes[1, 0].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('F1-Score by Label', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(LABEL_DISPLAY, rotation=45, ha='right', fontsize=10)
        axes[1, 0].set_ylim(0, 105)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        
        for bar, val in zip(bars3, f1_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. OVERALL ACCURACY
        axes[1, 1].bar(['Accuracy'], [accuracy], width=0.4, color='#4472C4', edgecolor='black', linewidth=1)
        axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Overall Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim(0, 105)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Add value label
        axes[1, 1].text(0, accuracy + 1, f'{accuracy:.1f}%', 
                       ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Add total frames info
        axes[1, 1].text(0, 5, f'Total Frames: {total_frames}\nCorrect: {result["correct"]}/{total_frames}', 
                       ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        # Save individual participant graph
        safe_name = participant_name.replace(' ', '_')
        plt.savefig(f'{output_dir}/{safe_name}_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/{safe_name}_metrics.png")
    
    # Also create combined comparison graphs
    
    # Extract data for comparison
    participants = [r['name'] for r in results]
    precision_data = []
    recall_data = []
    f1_data = []
    accuracies = []
    
    for result in results:
        precision_data.append([result['metrics'][label]['precision'] for label in LABELS])
        recall_data.append([result['metrics'][label]['recall'] for label in LABELS])
        f1_data.append([result['metrics'][label]['f1'] for label in LABELS])
        accuracies.append(result['accuracy'])
    
    # Combined Precision Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(participants))
    width = 0.15
    multiplier = 0
    
    for i, label in enumerate(LABELS):
        offset = width * multiplier
        values = [p[i] for p in precision_data]
        ax.bar(x + offset, values, width, label=LABEL_DISPLAY[i], color=COLORS[i])
        multiplier += 1
    
    ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=12, fontweight='bold')
    ax.set_title('Precision Comparison Across Patients', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/combined_precision.png")
    
    # Combined Recall Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(participants))
    multiplier = 0
    
    for i, label in enumerate(LABELS):
        offset = width * multiplier
        values = [r[i] for r in recall_data]
        ax.bar(x + offset, values, width, label=LABEL_DISPLAY[i], color=COLORS[i])
        multiplier += 1
    
    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=12, fontweight='bold')
    ax.set_title('Recall Comparison Across Patients', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/combined_recall.png")
    
    # Combined F1 Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(participants))
    multiplier = 0
    
    for i, label in enumerate(LABELS):
        offset = width * multiplier
        values = [r[i] for r in f1_data]
        ax.bar(x + offset, values, width, label=LABEL_DISPLAY[i], color=COLORS[i])
        multiplier += 1
    
    ax.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score Comparison Across Patients', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/combined_f1.png")
    
    # Accuracy Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(participants, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FF9FF3', '#54A0FF', '#5F27CD'])
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Patients', fontsize=12, fontweight='bold')
    ax.set_title('Overall Accuracy Comparison Across Patients', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    avg_accuracy = np.mean(accuracies)
    ax.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7, 
               label=f'Average Accuracy: {avg_accuracy:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/combined_accuracy.png")
    
    print(f"\n📊 All graphs saved to: {output_dir}/")
    print("   Individual patient graphs:")
    for result in results:
        safe_name = result['name'].replace(' ', '_')
        print(f"   - {safe_name}_metrics.png")
    print("   Combined comparison graphs:")
    print("   - combined_precision.png")
    print("   - combined_recall.png")
    print("   - combined_f1.png")
    print("   - combined_accuracy.png")

def main():
    """Main function to process hardcoded CSV files and generate graphs"""
    
    print("=" * 70)
    print("📊 PATIENT PERFORMANCE VISUALIZATION")
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
    
    print(f"\n📈 Generating graphs for {len(results)} patients...")
    generate_participant_graphs(results)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()