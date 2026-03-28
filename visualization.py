#!/usr/bin/env python3
"""
Comprehensive Visualization Module for Cross-Cultural Miscommunication Identifier
Creates various graphs and charts to visualize cultural dimensions and system performance.

Authors: NITHIES S (22MIA1025), ABISHEK B (22MIA1075)
Course: SWE1017, Slot: F2
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our main system
try:
    from main_system import CrossCulturalMiscommunicationIdentifier
except ImportError:
    print("Please ensure main_system.py is in the same directory")
    exit(1)

class CulturalVisualization:
    """Main visualization class for cultural analysis"""
    
    def __init__(self):
        self.system = CrossCulturalMiscommunicationIdentifier()
        self.cultures = list(self.system.cultural_graph.cultural_profiles.keys())
        self.dimensions = ['power_distance', 'individualism', 'directness_tolerance', 
                          'formality_preference', 'hierarchy_respect']
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_cultural_radar_chart(self, save_path: str = None):
        """Create radar chart comparing all cultures"""
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each dimension
        angles = [n / float(len(self.dimensions)) * 2 * np.pi for n in range(len(self.dimensions))]
        angles += angles[:1]  # Complete the circle
        
        # Colors for different cultures
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Plot each culture
        for i, culture in enumerate(self.cultures):
            profile = self.system.cultural_graph.cultural_profiles[culture]
            values = [
                profile.power_distance,
                profile.individualism,
                profile.directness_tolerance,
                profile.formality_preference,
                profile.hierarchy_respect
            ]
            values += values[:1]  # Complete the circle
            
            # Plot the culture
            ax.plot(angles, values, 'o-', linewidth=2.5, 
                   label=culture.title(), color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.2, color=colors[i % len(colors)])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.replace('_', ' ').title() for d in self.dimensions])
        ax.set_ylim(0, 1)
        ax.set_title('Cultural Dimensions Comparison\n(Cross-Cultural Communication Styles)', 
                    size=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for angle, dim in zip(angles[:-1], self.dimensions):
            ax.text(angle, 1.1, dim.replace('_', ' ').title(), 
                   horizontalalignment='center', size=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Radar chart saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_heatmap_cultural_matrix(self, save_path: str = None):
        """Create heatmap showing cultural dimension values"""
        # Prepare data for heatmap
        data = []
        for culture in self.cultures:
            profile = self.system.cultural_graph.cultural_profiles[culture]
            row = [
                profile.power_distance,
                profile.individualism,
                profile.directness_tolerance,
                profile.formality_preference,
                profile.hierarchy_respect
            ]
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, 
                         index=[c.title() for c in self.cultures],
                         columns=[d.replace('_', ' ').title() for d in self.dimensions])
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0.5,
                   square=True, fmt='.2f', cbar_kws={'label': 'Dimension Score (0-1)'})
        
        plt.title('Cultural Dimensions Heatmap\n(Higher values = More pronounced trait)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Cultural Dimensions', fontsize=12, fontweight='bold')
        plt.ylabel('Cultures', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()
        return plt.gcf()
    
    def create_risk_level_distribution(self, test_messages: List[Dict] = None, save_path: str = None):
        """Create bar chart showing risk level distribution across cultural pairs"""
        
        if not test_messages:
            # Default test messages
            test_messages = [
                "Submit the report by 5 PM.",
                "Could you please help me?",
                "This is wrong. Fix it.",
                "Great job on this project!",
                "We need to discuss this urgently.",
                "Thank you for your assistance.",
                "Please review the attached document.",
                "I disagree with this approach."
            ]
        
        # Test all cultural pairs
        results = []
        for sender in self.cultures:
            for receiver in self.cultures:
                if sender != receiver:
                    for message in test_messages:
                        alert = self.system.analyze_message(message, sender, receiver)
                        results.append({
                            'sender': sender.title(),
                            'receiver': receiver.title(),
                            'pair': f"{sender.title()} → {receiver.title()}",
                            'message': message,
                            'risk_level': alert.risk_level.value,
                            'confidence': alert.confidence_score
                        })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Count risk levels by cultural pair
        risk_counts = df.groupby(['pair', 'risk_level']).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(15, 8))
        risk_counts.plot(kind='bar', stacked=True, ax=ax, 
                        color=['#2ECC71', '#F39C12', '#E74C3C'])  # Green, Orange, Red
        
        plt.title('Risk Level Distribution Across Cultural Pairs\n(Based on Sample Messages)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Cultural Communication Pairs', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Messages', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Risk Level', title_fontsize=12, fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk distribution chart saved to {save_path}")
        
        plt.show()
        return fig, df
    
    def create_dimension_comparison_chart(self, cultures_to_compare: List[str] = None, save_path: str = None):
        """Create grouped bar chart comparing specific cultures"""
        
        if not cultures_to_compare:
            cultures_to_compare = ['american', 'japanese', 'german', 'indian']
        
        # Prepare data
        data = {}
        for culture in cultures_to_compare:
            if culture in self.system.cultural_graph.cultural_profiles:
                profile = self.system.cultural_graph.cultural_profiles[culture]
                data[culture.title()] = [
                    profile.power_distance,
                    profile.individualism,
                    profile.directness_tolerance,
                    profile.formality_preference,
                    profile.hierarchy_respect
                ]
        
        # Create DataFrame
        df = pd.DataFrame(data, index=[d.replace('_', ' ').title() for d in self.dimensions])
        
        # Create grouped bar chart
        ax = df.plot(kind='bar', figsize=(12, 8), width=0.8)
        
        plt.title('Cultural Dimensions Comparison\n(Selected Cultures)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Cultural Dimensions', fontsize=12, fontweight='bold')
        plt.ylabel('Dimension Score (0-1)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Cultures', title_fontsize=12, fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to {save_path}")
        
        plt.show()
        return plt.gcf()
    
    def create_interactive_3d_plot(self, save_path: str = None):
        """Create interactive 3D scatter plot using plotly"""
        
        # Prepare data
        cultures_data = []
        for culture in self.cultures:
            profile = self.system.cultural_graph.cultural_profiles[culture]
            cultures_data.append({
                'culture': culture.title(),
                'power_distance': profile.power_distance,
                'directness_tolerance': profile.directness_tolerance,
                'formality_preference': profile.formality_preference,
                'individualism': profile.individualism,
                'hierarchy_respect': profile.hierarchy_respect
            })
        
        df = pd.DataFrame(cultures_data)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(df, 
                           x='power_distance', 
                           y='directness_tolerance', 
                           z='formality_preference',
                           color='culture',
                           size='hierarchy_respect',
                           hover_data=['individualism'],
                           title='3D Cultural Dimensions Space<br>Size = Hierarchy Respect, Hover for Individualism')
        
        # Customize layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Power Distance',
                yaxis_title='Directness Tolerance', 
                zaxis_title='Formality Preference'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(f"{save_path}.html")
            print(f"Interactive 3D plot saved to {save_path}.html")
        
        fig.show()
        return fig
    
    def create_miscommunication_risk_matrix(self, save_path: str = None):
        """Create matrix showing miscommunication risk between all cultural pairs"""
        
        # Test message for consistency
        test_message = "Please complete this task by tomorrow."
        
        # Create risk matrix
        risk_matrix = np.zeros((len(self.cultures), len(self.cultures)))
        culture_labels = [c.title() for c in self.cultures]
        
        for i, sender in enumerate(self.cultures):
            for j, receiver in enumerate(self.cultures):
                if sender != receiver:
                    alert = self.system.analyze_message(test_message, sender, receiver)
                    # Convert risk level to numeric value
                    risk_value = {'low': 0.33, 'medium': 0.66, 'high': 1.0}[alert.risk_level.value]
                    risk_matrix[i, j] = risk_value
                else:
                    risk_matrix[i, j] = 0  # No risk communicating within same culture
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(risk_matrix, 
                   xticklabels=culture_labels,
                   yticklabels=culture_labels,
                   annot=True, 
                   cmap='RdYlGn_r',
                   center=0.5,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Miscommunication Risk\n(0 = Low, 1 = High)'})
        
        plt.title('Cross-Cultural Miscommunication Risk Matrix\n(Test Message: "Please complete this task by tomorrow.")', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Receiver Culture', fontsize=12, fontweight='bold')
        plt.ylabel('Sender Culture', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk matrix saved to {save_path}")
        
        plt.show()
        return plt.gcf(), risk_matrix
    
    def create_dashboard_subplot(self, save_path: str = None):
        """Create a comprehensive dashboard with multiple visualizations"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Cultural Dimensions Radar (top left)
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        angles = [n / float(len(self.dimensions)) * 2 * np.pi for n in range(len(self.dimensions))]
        angles += angles[:1]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for i, culture in enumerate(self.cultures):
            profile = self.system.cultural_graph.cultural_profiles[culture]
            values = [profile.power_distance, profile.individualism, 
                     profile.directness_tolerance, profile.formality_preference, 
                     profile.hierarchy_respect] + [profile.power_distance]
            ax1.plot(angles, values, 'o-', linewidth=2, label=culture.title(), color=colors[i])
            ax1.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([d.replace('_', '\n').title() for d in self.dimensions], fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.set_title('Cultural Dimensions', fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=8)
        
        # 2. Directness vs Formality Scatter (top middle)
        ax2 = plt.subplot(2, 3, 2)
        for i, culture in enumerate(self.cultures):
            profile = self.system.cultural_graph.cultural_profiles[culture]
            ax2.scatter(profile.directness_tolerance, profile.formality_preference, 
                       s=200, alpha=0.7, color=colors[i], label=culture.title())
            ax2.annotate(culture.title(), 
                        (profile.directness_tolerance, profile.formality_preference),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Directness Tolerance')
        ax2.set_ylabel('Formality Preference')
        ax2.set_title('Communication Style Matrix', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # 3. Power Distance vs Hierarchy (top right)
        ax3 = plt.subplot(2, 3, 3)
        for i, culture in enumerate(self.cultures):
            profile = self.system.cultural_graph.cultural_profiles[culture]
            ax3.scatter(profile.power_distance, profile.hierarchy_respect,
                       s=200, alpha=0.7, color=colors[i], label=culture.title())
            ax3.annotate(culture.title(),
                        (profile.power_distance, profile.hierarchy_respect),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Power Distance')
        ax3.set_ylabel('Hierarchy Respect')
        ax3.set_title('Authority Relationship Matrix', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # 4. Risk Level Distribution (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        test_messages = ["Send report now.", "Please help.", "This is wrong."]
        risk_data = {'low': 0, 'medium': 0, 'high': 0}
        
        for sender in self.cultures:
            for receiver in self.cultures:
                if sender != receiver:
                    for msg in test_messages:
                        alert = self.system.analyze_message(msg, sender, receiver)
                        risk_data[alert.risk_level.value] += 1
        
        ax4.bar(risk_data.keys(), risk_data.values(), color=['#2ECC71', '#F39C12', '#E74C3C'])
        ax4.set_title('Risk Level Distribution', fontweight='bold')
        ax4.set_ylabel('Number of Cases')
        
        # 5. Cultural Dimensions Bar Chart (bottom middle and right)
        ax5 = plt.subplot(2, 3, (5, 6))
        
        # Prepare data for stacked bar chart
        dimension_data = {dim: [] for dim in self.dimensions}
        for culture in self.cultures:
            profile = self.system.cultural_graph.cultural_profiles[culture]
            dimension_data['power_distance'].append(profile.power_distance)
            dimension_data['individualism'].append(profile.individualism)
            dimension_data['directness_tolerance'].append(profile.directness_tolerance)
            dimension_data['formality_preference'].append(profile.formality_preference)
            dimension_data['hierarchy_respect'].append(profile.hierarchy_respect)
        
        x = np.arange(len(self.cultures))
        width = 0.15
        
        for i, dim in enumerate(self.dimensions):
            ax5.bar(x + i * width, dimension_data[dim], width, 
                   label=dim.replace('_', ' ').title(), alpha=0.8)
        
        ax5.set_xlabel('Cultures')
        ax5.set_ylabel('Dimension Score')
        ax5.set_title('Cultural Dimensions Comparison', fontweight='bold')
        ax5.set_xticks(x + width * 2)
        ax5.set_xticklabels([c.title() for c in self.cultures])
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Cross-Cultural Communication Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        plt.show()
        return fig

def demonstrate_all_visualizations():
    """Run all visualization demos"""
    print("Cross-Cultural Miscommunication Identifier - Visualization Demo")
    print("=" * 70)
    
    viz = CulturalVisualization()
    
    print("\n1. Creating Cultural Radar Chart...")
    viz.create_cultural_radar_chart("cultural_radar_chart.png")
    
    print("\n2. Creating Cultural Dimensions Heatmap...")
    viz.create_heatmap_cultural_matrix("cultural_heatmap.png")
    
    print("\n3. Creating Risk Level Distribution...")
    viz.create_risk_level_distribution(save_path="risk_distribution.png")
    
    print("\n4. Creating Dimension Comparison Chart...")
    viz.create_dimension_comparison_chart(save_path="dimension_comparison.png")
    
    print("\n5. Creating Miscommunication Risk Matrix...")
    viz.create_miscommunication_risk_matrix("risk_matrix.png")
    
    print("\n6. Creating Comprehensive Dashboard...")
    viz.create_dashboard_subplot("cultural_dashboard.png")
    
    try:
        print("\n7. Creating Interactive 3D Plot...")
        viz.create_interactive_3d_plot("interactive_3d_plot")
    except Exception as e:
        print(f"Interactive plot failed (requires plotly): {e}")
    
    print("\n✅ All visualizations created successfully!")
    print("Check the generated PNG files in your project directory.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        viz = CulturalVisualization()
        
        if sys.argv[1] == "radar":
            viz.create_cultural_radar_chart()
        elif sys.argv[1] == "heatmap":
            viz.create_heatmap_cultural_matrix()
        elif sys.argv[1] == "risk":
            viz.create_risk_level_distribution()
        elif sys.argv[1] == "comparison":
            viz.create_dimension_comparison_chart()
        elif sys.argv[1] == "matrix":
            viz.create_miscommunication_risk_matrix()
        elif sys.argv[1] == "dashboard":
            viz.create_dashboard_subplot()
        elif sys.argv[1] == "interactive":
            viz.create_interactive_3d_plot()
        else:
            print("Available options: radar, heatmap, risk, comparison, matrix, dashboard, interactive")
    else:
        demonstrate_all_visualizations()