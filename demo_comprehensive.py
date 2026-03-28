#!/usr/bin/env python3
"""
Comprehensive Demo Script for Cross-Cultural Miscommunication Identifier
This script demonstrates all features and capabilities of the system.

Authors: NITHIES S (22MIA1025), ABISHEK B (22MIA1075)
Course: SWE1017, Slot: F2
"""

import os
import sys
import json
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import our main system
try:
    from main_system import CrossCulturalMiscommunicationIdentifier, RiskLevel
    from evaluation_and_testing import EvaluationFramework, PerformanceAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all project files are in the same directory.")
    sys.exit(1)

class ComprehensiveDemo:
    """Comprehensive demonstration of the system capabilities"""
    
    def __init__(self):
        print("Initializing Cross-Cultural Miscommunication Identifier...")
        self.system = CrossCulturalMiscommunicationIdentifier()
        print("System initialized successfully!\n")
    
    def demo_basic_analysis(self):
        """Demonstrate basic message analysis"""
        print("="*60)
        print("DEMO 1: BASIC MESSAGE ANALYSIS")
        print("="*60)
        
        test_cases = [
            {
                "message": "Submit the report by 5 PM today.",
                "sender": "american",
                "receiver": "japanese",
                "scenario": "American manager to Japanese subordinate"
            },
            {
                "message": "Could you please help me with this task when you have a moment?",
                "sender": "british",
                "receiver": "german", 
                "scenario": "British colleague to German colleague"
            },
            {
                "message": "This approach is completely wrong. Fix it immediately.",
                "sender": "german",
                "receiver": "indian",
                "scenario": "German team lead giving feedback to Indian team member"
            },
            {
                "message": "I humbly request your kind consideration of my proposal.",
                "sender": "indian",
                "receiver": "american",
                "scenario": "Indian employee to American manager"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {case['scenario']}")
            print(f"Message: \"{case['message']}\"")
            print(f"Route: {case['sender'].title()} → {case['receiver'].title()}")
            
            # Analyze the message
            alert = self.system.analyze_message(
                case['message'], case['sender'], case['receiver']
            )
            
            # Display results with formatting
            risk