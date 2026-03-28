"""
Evaluation and Testing Module for Cross-Cultural Miscommunication Identifier
This module provides comprehensive testing and evaluation capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from datetime import datetime
import os

# Import our main system (assuming it's in the same directory)
try:
    from main_system import CrossCulturalMiscommunicationIdentifier, RiskLevel, MiscommunicationAlert
except ImportError:
    print("Please ensure main_system.py is in the same directory")
    exit(1)

class EvaluationFramework:
    """Comprehensive evaluation framework for the system"""
    
    def __init__(self):
        self.system = CrossCulturalMiscommunicationIdentifier()
        self.test_results = []
        
    def create_test_dataset(self) -> List[Dict]:
        """Create comprehensive test dataset"""
        test_cases = [
            # American to Japanese (High formality mismatch)
            {"text": "Send me the report now.", "sender": "american", "receiver": "japanese", 
             "expected_risk": "high", "category": "directness"},
            {"text": "Could you please send the report when convenient?", "sender": "american", 
             "receiver": "japanese", "expected_risk": "low", "category": "polite_request"},
            
            # German to Indian (Different directness preferences)
            {"text": "This approach is wrong. Fix it immediately.", "sender": "german", 
             "receiver": "indian", "expected_risk": "high", "category": "criticism"},
            {"text": "Perhaps we could consider an alternative approach to this task.", 
             "sender": "german", "receiver": "indian", "expected_risk": "low", "category": "suggestion"},
            
            # Japanese to American (Over-politeness)
            {"text": "I humbly request that you might possibly consider reviewing this document if it's not too much trouble.", 
             "sender": "japanese", "receiver": "american", "expected_risk": "medium", "category": "over_polite"},
            {"text": "Please review this document by Friday.", "sender": "japanese", 
             "receiver": "american", "expected_risk": "low", "category": "appropriate"},
            
            # British to German (Indirectness vs Directness)
            {"text": "I was wondering if you might possibly be able to help with this rather challenging task when you have a spare moment.", 
             "sender": "british", "receiver": "german", "expected_risk": "medium", "category": "too_indirect"},
            {"text": "Please help with this task.", "sender": "british", "receiver": "german", 
             "expected_risk": "low", "category": "direct_request"},
            
            # Indian to American (Hierarchy awareness)
            {"text": "Sir, I most respectfully request your kind consideration for my humble proposal.", 
             "sender": "indian", "receiver": "american", "expected_risk": "medium", "category": "over_formal"},
            {"text": "I'd like to propose this idea for your consideration.", "sender": "indian", 
             "receiver": "american", "expected_risk": "low", "category": "professional"},
            
            # Cross-cultural compliments
            {"text": "Your work is acceptable.", "sender": "german", "receiver": "american", 
             "expected_risk": "medium", "category": "understated_praise"},
            {"text": "Excellent job on this project!", "sender": "american", "receiver": "german", 
             "expected_risk": "low", "category": "direct_praise"},
            
            # Time-sensitive requests
            {"text": "I need this ASAP.", "sender": "american", "receiver": "japanese", 
             "expected_risk": "high", "category": "urgent_direct"},
            {"text": "When you have time, could you prioritize this task?", "sender": "american", 
             "receiver": "japanese", "expected_risk": "low", "category": "urgent_polite"},
            
            # Criticism and feedback
            {"text": "This is completely wrong.", "sender": "american", "receiver": "indian", 
             "expected_risk": "high", "category": "harsh_criticism"},
            {"text": "There might be room for improvement in this approach.", "sender": "american", 
             "receiver": "indian", "expected_risk": "low", "category": "gentle_feedback"},
        ]
        
        return test_cases
    
    def run_evaluation(self) -> Dict:
        """Run comprehensive evaluation"""
        test_cases = self.create_test_dataset()
        results = {
            'total_cases': len(test_cases),
            'correct_predictions': 0,
            'risk_level_accuracy': {'low': 0, 'medium': 0, 'high': 0},
            'category_performance': {},
            'detailed_results': []
        }
        
        print(f"Running evaluation on {len(test_cases)} test cases...")
        
        for i, case in enumerate(test_cases):
            # Analyze the message
            alert = self.system.analyze_message(
                case['text'], case['sender'], case['receiver']
            )
            
            # Check if prediction matches expected
            predicted_risk = alert.risk_level.value
            expected_risk = case['expected_risk']
            correct = predicted_risk == expected_risk
            
            if correct:
                results['correct_predictions'] += 1
                results['risk_level_accuracy'][expected_risk] += 1
            
            # Track category performance
            category = case['category']
            if category not in results['category_performance']:
                results['category_performance'][category] = {'correct': 0, 'total': 0}
            
            results['category_performance'][category]['total'] += 1
            if correct:
                results['category_performance'][category]['correct'] += 1
            
            # Store detailed result
            detail = {
                'case_id': i + 1,
                'text': case['text'],
                'sender': case['sender'],
                'receiver': case['receiver'],
                'expected_risk': expected_risk,
                'predicted_risk': predicted_risk,
                'correct': correct,
                'category': category,
                'confidence': alert.confidence_score,
                'explanation': alert.explanation,
                'rewrite': alert.suggested_rewrite
            }
            results['detailed_results'].append(detail)
            
            # Print progress
            status = "✓" if correct else "✗"
            print(f"{status} Case {i+1}/{len(test_cases)}: {case['category']} - "
                  f"Expected: {expected_risk}, Got: {predicted_risk}")
        
        # Calculate overall accuracy
        results['overall_accuracy'] = results['correct_predictions'] / results['total_cases']
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate evaluation report"""
        report = f"""
CROSS-CULTURAL MISCOMMUNICATION IDENTIFIER - EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

OVERALL PERFORMANCE:
- Total Test Cases: {results['total_cases']}
- Correct Predictions: {results['correct_predictions']}
- Overall Accuracy: {results['overall_accuracy']:.2%}

RISK LEVEL ACCURACY:
"""
        
        for risk_level, correct_count in results['risk_level_accuracy'].items():
            total_of_level = sum(1 for case in results['detailed_results'] 
                               if case['expected_risk'] == risk_level)
            accuracy = correct_count / total_of_level if total_of_level > 0 else 0
            report += f"- {risk_level.upper()}: {correct_count}/{total_of_level} ({accuracy:.2%})\n"
        
        report += "\nCATEGORY PERFORMANCE:\n"
        for category, stats in results['category_performance'].items():
            accuracy = stats['correct'] / stats['total']
            report += f"- {category.replace('_', ' ').title()}: {stats['correct']}/{stats['total']} ({accuracy:.2%})\n"
        
        report += "\nDETAILED ANALYSIS:\n"
        report += "=" * 70 + "\n"
        
        for result in results['detailed_results']:
            status_icon = "✓" if result['correct'] else "✗"
            report += f"\n{status_icon} Case {result['case_id']}: {result['category'].replace('_', ' ').title()}\n"
            report += f"Text: \"{result['text']}\"\n"
            report += f"Route: {result['sender'].title()} → {result['receiver'].title()}\n"
            report += f"Expected Risk: {result['expected_risk'].upper()}\n"
            report += f"Predicted Risk: {result['predicted_risk'].upper()}\n"
            report += f"Confidence: {result['confidence']:.2f}\n"
            report += f"Explanation: {result['explanation']}\n"
            report += f"Suggested Rewrite: \"{result['rewrite']}\"\n"
            report += "-" * 50 + "\n"
        
        return report
    
    def visualize_results(self, results: Dict):
        """Create visualization of evaluation results"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Cultural Miscommunication Identifier - Evaluation Results', fontsize=16)
        
        # 1. Overall Accuracy
        ax1 = axes[0, 0]
        accuracy = results['overall_accuracy']
        colors = ['#2E8B57' if accuracy >= 0.8 else '#FF6B35' if accuracy >= 0.6 else '#DC143C']
        ax1.bar(['Overall Accuracy'], [accuracy], color=colors[0], alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy Score')
        ax1.set_title('Overall Performance')
        ax1.text(0, accuracy + 0.02, f'{accuracy:.1%}', ha='center', fontweight='bold')
        
        # 2. Risk Level Performance
        ax2 = axes[0, 1]
        risk_levels = list(results['risk_level_accuracy'].keys())
        accuracies = []
        for level in risk_levels:
            total_of_level = sum(1 for case in results['detailed_results'] 
                               if case['expected_risk'] == level)
            acc = results['risk_level_accuracy'][level] / total_of_level if total_of_level > 0 else 0
            accuracies.append(acc)
        
        bars = ax2.bar(risk_levels, accuracies, color=['#90EE90', '#FFD700', '#FFA07A'], alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Accuracy Score')
        ax2.set_title('Performance by Risk Level')
        ax2.set_xlabel('Risk Level')
        
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                    f'{acc:.1%}', ha='center', fontweight='bold')
        
        # 3. Category Performance
        ax3 = axes[1, 0]
        categories = list(results['category_performance'].keys())
        cat_accuracies = [results['category_performance'][cat]['correct'] / 
                         results['category_performance'][cat]['total'] 
                         for cat in categories]
        
        colors_cat = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        bars = ax3.barh(range(len(categories)), cat_accuracies, color=colors_cat, alpha=0.7)
        ax3.set_yticks(range(len(categories)))
        ax3.set_yticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax3.set_xlim(0, 1)
        ax3.set_xlabel('Accuracy Score')
        ax3.set_title('Performance by Category')
        
        for i, (bar, acc) in enumerate(zip(bars, cat_accuracies)):
            ax3.text(acc + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.1%}', va='center', fontweight='bold')
        
        # 4. Confusion Matrix
        ax4 = axes[1, 1]
        risk_levels_ordered = ['low', 'medium', 'high']
        confusion_matrix = np.zeros((3, 3))
        
        for result in results['detailed_results']:
            expected_idx = risk_levels_ordered.index(result['expected_risk'])
            predicted_idx = risk_levels_ordered.index(result['predicted_risk'])
            confusion_matrix[expected_idx][predicted_idx] += 1
        
        im = ax4.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        ax4.set_title('Confusion Matrix')
        ax4.set_xticks(range(3))
        ax4.set_yticks(range(3))
        ax4.set_xticklabels([level.title() for level in risk_levels_ordered])
        ax4.set_yticklabels([level.title() for level in risk_levels_ordered])
        ax4.set_xlabel('Predicted Risk Level')
        ax4.set_ylabel('Expected Risk Level')
        
        # Add text annotations to confusion matrix
        for i in range(3):
            for j in range(3):
                text = ax4.text(j, i, int(confusion_matrix[i, j]), 
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def save_results(self, results: Dict, filename_prefix: str = "evaluation"):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results as JSON
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save results as CSV
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(csv_filename, index=False)
        
        # Save report as text
        report_filename = f"{filename_prefix}_report_{timestamp}.txt"
        report = self.generate_report(results)
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nResults saved:")
        print(f"- JSON: {json_filename}")
        print(f"- CSV: {csv_filename}")
        print(f"- Report: {report_filename}")
        
        return json_filename, csv_filename, report_filename

class PerformanceAnalyzer:
    """Analyze system performance across different dimensions"""
    
    def __init__(self):
        self.system = CrossCulturalMiscommunicationIdentifier()
    
    def cultural_pair_analysis(self) -> Dict:
        """Analyze performance across different cultural pairs"""
        cultures = ['american', 'japanese', 'german', 'indian', 'british']
        test_message = "Please complete this task by tomorrow."
        
        results = {}
        for sender in cultures:
            for receiver in cultures:
                if sender != receiver:
                    pair_key = f"{sender}->{receiver}"
                    alert = self.system.analyze_message(test_message, sender, receiver)
                    results[pair_key] = {
                        'risk_level': alert.risk_level.value,
                        'confidence': alert.confidence_score,
                        'explanation': alert.explanation
                    }
        
        return results
    
    def speech_act_analysis(self) -> Dict:
        """Test different types of speech acts"""
        test_cases = {
            'direct_order': "Do this now.",
            'polite_request': "Could you please help me with this?",
            'suggestion': "Maybe we should consider this approach.",
            'question': "How should we handle this situation?",
            'compliment': "Great work on this project!",
            'criticism': "This approach is not working."
        }
        
        results = {}
        for speech_type, message in test_cases.items():
            # Test with high-context vs low-context cultures
            high_context_alert = self.system.analyze_message(
                message, "american", "japanese"
            )
            low_context_alert = self.system.analyze_message(
                message, "japanese", "american"
            )
            
            results[speech_type] = {
                'high_context': {
                    'risk': high_context_alert.risk_level.value,
                    'confidence': high_context_alert.confidence_score
                },
                'low_context': {
                    'risk': low_context_alert.risk_level.value,
                    'confidence': low_context_alert.confidence_score
                }
            }
        
        return results

def run_comprehensive_evaluation():
    """Run the complete evaluation suite"""
    print("Cross-Cultural Miscommunication Identifier - Comprehensive Evaluation")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = EvaluationFramework()
    
    # Run main evaluation
    print("\n1. Running main evaluation...")
    results = evaluator.run_evaluation()
    
    # Generate and display report
    print("\n2. Generating evaluation report...")
    report = evaluator.generate_report(results)
    print(report)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    fig = evaluator.visualize_results(results)
    plt.show()
    
    # Save results
    print("\n4. Saving results...")
    files = evaluator.save_results(results)
    
    # Additional analysis
    print("\n5. Running additional analysis...")
    analyzer = PerformanceAnalyzer()
    
    cultural_pairs = analyzer.cultural_pair_analysis()
    print(f"\nCultural Pair Analysis completed for {len(cultural_pairs)} pairs")
    
    speech_acts = analyzer.speech_act_analysis()
    print(f"Speech Act Analysis completed for {len(speech_acts)} types")
    
    # Save additional analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f"cultural_pairs_analysis_{timestamp}.json", 'w') as f:
        json.dump(cultural_pairs, f, indent=2)
    
    with open(f"speech_acts_analysis_{timestamp}.json", 'w') as f:
        json.dump(speech_acts, f, indent=2)
    
    print(f"\nEvaluation completed! Overall accuracy: {results['overall_accuracy']:.2%}")
    return results

def quick_test():
    """Quick test function for development"""
    system = CrossCulturalMiscommunicationIdentifier()
    
    test_cases = [
        ("Send me the report now.", "american", "japanese"),
        ("Could you please help when convenient?", "british", "german"),
        ("This is completely wrong.", "german", "indian")
    ]
    
    print("Quick Test Results:")
    print("=" * 50)
    
    for i, (text, sender, receiver) in enumerate(test_cases, 1):
        alert = system.analyze_message(text, sender, receiver)
        print(f"\nTest {i}:")
        print(f"Message: \"{text}\"")
        print(f"Route: {sender.title()} → {receiver.title()}")
        print(f"Risk: {alert.risk_level.value.upper()}")
        print(f"Confidence: {alert.confidence_score:.2f}")
        print(f"Rewrite: \"{alert.suggested_rewrite}\"")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        try:
            run_comprehensive_evaluation()
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()