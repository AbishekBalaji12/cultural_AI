#!/usr/bin/env python3
"""
Simple Cultural Visualization Script
Creates a quick radar chart showing cultural dimensions.

Authors: NITHIES S (22MIA1025), ABISHEK B (22MIA1075)
Course: SWE1017, Slot: F2
"""

import matplotlib.pyplot as plt
import numpy as np

def create_simple_cultural_radar():
    """Create a simple cultural radar chart"""
    
    # Cultural data (based on research)
    cultures = {
        'American': [0.4, 0.91, 0.8, 0.3, 0.4],
        'Japanese': [0.54, 0.46, 0.2, 0.9, 0.85],
        'German': [0.35, 0.67, 0.9, 0.7, 0.5],
        'Indian': [0.77, 0.48, 0.3, 0.8, 0.9],
        'British': [0.35, 0.89, 0.4, 0.6, 0.4]
    }
    
    # Dimension labels
    dimensions = [
        'Power\nDistance',
        'Individualism',
        'Directness\nTolerance', 
        'Formality\nPreference',
        'Hierarchy\nRespect'
    ]
    
    # Calculate angles for radar chart
    angles = [n / float(len(dimensions)) * 2 * np.pi for n in range(len(dimensions))]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for each culture
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Plot each culture
    for i, (culture, values) in enumerate(cultures.items()):
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2.5, 
               label=culture, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 1)
    ax.set_title('Cultural Communication Dimensions\nCross-Cultural Miscommunication Identifier', 
                size=16, fontweight='bold', pad=30)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Add concentric circles with labels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cultural_radar_chart.png', dpi=300, bbox_inches='tight')
    print("✅ Cultural radar chart saved as 'cultural_radar_chart.png'")
    
    # Show the plot
    plt.show()

def create_simple_heatmap():
    """Create a simple heatmap of cultural dimensions"""
    
    # Cultural data
    cultures = ['American', 'Japanese', 'German', 'Indian', 'British']
    dimensions = ['Power Distance', 'Individualism', 'Directness', 'Formality', 'Hierarchy']
    
    data = [
        [0.4, 0.91, 0.8, 0.3, 0.4],   # American
        [0.54, 0.46, 0.2, 0.9, 0.85], # Japanese  
        [0.35, 0.67, 0.9, 0.7, 0.5],  # German
        [0.77, 0.48, 0.3, 0.8, 0.9],  # Indian
        [0.35, 0.89, 0.4, 0.6, 0.4]   # British
    ]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_yticks(np.arange(len(cultures)))
    ax.set_xticklabels(dimensions)
    ax.set_yticklabels(cultures)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Dimension Score (0-1)', rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(len(cultures)):
        for j in range(len(dimensions)):
            text = ax.text(j, i, f'{data[i][j]:.2f}', ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title("Cultural Dimensions Heatmap\nCross-Cultural Communication Analysis", 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cultural_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Cultural heatmap saved as 'cultural_heatmap.png'")
    
    # Show the plot
    plt.show()

def create_simple_bar_chart():
    """Create a simple bar chart comparing cultures"""
    
    cultures = ['American', 'Japanese', 'German', 'Indian', 'British']
    directness = [0.8, 0.2, 0.9, 0.3, 0.4]
    formality = [0.3, 0.9, 0.7, 0.8, 0.6]
    hierarchy = [0.4, 0.85, 0.5, 0.9, 0.4]
    
    x = np.arange(len(cultures))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width, directness, width, label='Directness Tolerance', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, formality, width, label='Formality Preference', color='#E74C3C', alpha=0.8)
    bars3 = ax.bar(x + width, hierarchy, width, label='Hierarchy Respect', color='#2ECC71', alpha=0.8)
    
    ax.set_xlabel('Cultures', fontweight='bold')
    ax.set_ylabel('Dimension Score (0-1)', fontweight='bold')
    ax.set_title('Key Cultural Dimensions Comparison\nCross-Cultural Communication Styles', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cultures)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cultural_bar_chart.png', dpi=300, bbox_inches='tight')
    print("✅ Cultural bar chart saved as 'cultural_bar_chart.png'")
    
    # Show the plot
    plt.show()

def main():
    """Main function to create all simple visualizations"""
    print("Cross-Cultural Miscommunication Identifier - Simple Visualizations")
    print("=" * 70)
    
    try:
        print("\n1. Creating Cultural Radar Chart...")
        create_simple_cultural_radar()
        
        print("\n2. Creating Cultural Heatmap...")
        create_simple_heatmap()
        
        print("\n3. Creating Cultural Bar Chart...")
        create_simple_bar_chart()
        
        print("\n✅ All visualizations created successfully!")
        print("Files saved:")
        print("  - cultural_radar_chart.png")
        print("  - cultural_heatmap.png") 
        print("  - cultural_bar_chart.png")
        
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "radar":
            create_simple_cultural_radar()
        elif sys.argv[1] == "heatmap":
            create_simple_heatmap()
        elif sys.argv[1] == "bar":
            create_simple_bar_chart()
        else:
            print("Available options:")
            print("  python simple_cultural_graph.py radar    - Create radar chart only")
            print("  python simple_cultural_graph.py heatmap  - Create heatmap only")
            print("  python simple_cultural_graph.py bar      - Create bar chart only")
            print("  python simple_cultural_graph.py          - Create all charts")
    else:
        main()