import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        pipeline, AutoModel
    )
    import torch
    import torch.nn as nn
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from textblob import TextBlob
except ImportError as e:
    print(f"Installing required packages... {e}")
    os.system("pip install transformers torch sklearn nltk textblob numpy pandas")
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        pipeline, AutoModel
    )
    import torch
    import torch.nn as nn
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpeechAct(Enum):
    REQUEST = "request"
    ORDER = "order"
    SUGGESTION = "suggestion"
    APOLOGY = "apology"
    COMPLIMENT = "compliment"
    QUESTION = "question"
    STATEMENT = "statement"

class CulturalDimension(Enum):
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM = "individualism"
    MASCULINITY = "masculinity"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE = "indulgence"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class CulturalProfile:
    """Represents cultural communication norms for a specific culture"""
    name: str
    power_distance: float  # 0-1 scale
    individualism: float
    directness_tolerance: float
    formality_preference: float
    hierarchy_respect: float

@dataclass
class MiscommunicationAlert:
    """Represents a detected miscommunication risk"""
    original_text: str
    risk_level: RiskLevel
    explanation: str
    suggested_rewrite: str
    confidence_score: float

class CulturalKnowledgeGraph:
    """Simplified cultural knowledge representation"""
    
    def __init__(self):
        self.cultural_profiles = self._initialize_profiles()
        
    def _initialize_profiles(self) -> Dict[str, CulturalProfile]:
        """Initialize cultural profiles based on research"""
        return {
            "american": CulturalProfile(
                name="American",
                power_distance=0.4,
                individualism=0.91,
                directness_tolerance=0.8,
                formality_preference=0.3,
                hierarchy_respect=0.4
            ),
            "japanese": CulturalProfile(
                name="Japanese",
                power_distance=0.54,
                individualism=0.46,
                directness_tolerance=0.2,
                formality_preference=0.9,
                hierarchy_respect=0.85
            ),
            "german": CulturalProfile(
                name="German",
                power_distance=0.35,
                individualism=0.67,
                directness_tolerance=0.9,
                formality_preference=0.7,
                hierarchy_respect=0.5
            ),
            "indian": CulturalProfile(
                name="Indian",
                power_distance=0.77,
                individualism=0.48,
                directness_tolerance=0.3,
                formality_preference=0.8,
                hierarchy_respect=0.9
            ),
            "british": CulturalProfile(
                name="British",
                power_distance=0.35,
                individualism=0.89,
                directness_tolerance=0.4,
                formality_preference=0.6,
                hierarchy_respect=0.4
            )
        }
    
    def get_profile(self, culture: str) -> Optional[CulturalProfile]:
        """Get cultural profile for a given culture"""
        return self.cultural_profiles.get(culture.lower())
    
    def calculate_mismatch_score(self, sender_culture: str, receiver_culture: str, 
                                speech_act: str, directness_score: float) -> float:
        """Calculate cultural mismatch score"""
        sender = self.get_profile(sender_culture)
        receiver = self.get_profile(receiver_culture)
        
        if not sender or not receiver:
            return 0.5  # Default moderate risk
        
        # Calculate mismatch based on directness tolerance
        directness_mismatch = abs(directness_score - receiver.directness_tolerance)
        
        # Factor in hierarchy respect for orders/requests
        hierarchy_factor = 1.0
        if speech_act in ["order", "request"]:
            hierarchy_factor = abs(sender.hierarchy_respect - receiver.hierarchy_respect)
        
        # Combine factors
        mismatch_score = (directness_mismatch + hierarchy_factor * 0.3) / 1.3
        return min(mismatch_score, 1.0)

class SpeechActClassifier:
    """Classifies speech acts in text"""
    
    def __init__(self):
        self.patterns = {
            SpeechAct.REQUEST: [
                r'\b(could you|would you|can you|please)\b',
                r'\b(i would appreciate|if possible)\b',
                r'\?(.*)(help|assist|send|provide)'
            ],
            SpeechAct.ORDER: [
                r'\b(must|need to|have to|should)\b',
                r'^(submit|send|complete|finish|do)',
                r'\b(immediately|asap|by \d+|deadline)\b'
            ],
            SpeechAct.SUGGESTION: [
                r'\b(suggest|recommend|might want to|consider)\b',
                r'\b(how about|what if|maybe|perhaps)\b'
            ],
            SpeechAct.APOLOGY: [
                r'\b(sorry|apologize|regret|my mistake)\b'
            ],
            SpeechAct.COMPLIMENT: [
                r'\b(great|excellent|well done|good job)\b',
                r'\b(impressive|outstanding|fantastic)\b'
            ],
            SpeechAct.QUESTION: [
                r'\?$',
                r'^(what|when|where|why|how|who)\b'
            ]
        }
    
    def classify(self, text: str) -> SpeechAct:
        """Classify speech act of given text"""
        text_lower = text.lower().strip()
        
        scores = {}
        for act, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[act] = score
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return SpeechAct.STATEMENT

class PolitenessAnalyzer:
    """Analyzes politeness and directness of text"""
    
    def __init__(self):
        self.polite_markers = [
            'please', 'thank you', 'could you', 'would you', 'if possible',
            'i would appreciate', 'when convenient', 'at your earliest'
        ]
        
        self.direct_markers = [
            'must', 'need to', 'have to', 'immediately', 'asap', 'now',
            'required', 'mandatory', 'do this', 'send me'
        ]
        
        self.hedge_words = [
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'would',
            'i think', 'i believe', 'it seems', 'kind of', 'sort of'
        ]
    
    def analyze_politeness(self, text: str) -> float:
        """Calculate politeness score (0-1, higher = more polite)"""
        text_lower = text.lower()
        
        polite_count = sum(1 for marker in self.polite_markers if marker in text_lower)
        hedge_count = sum(1 for hedge in self.hedge_words if hedge in text_lower)
        direct_count = sum(1 for marker in self.direct_markers if marker in text_lower)
        
        # Calculate politeness score
        politeness = (polite_count * 2 + hedge_count - direct_count) / max(len(text.split()), 1)
        return max(0, min(1, politeness + 0.5))  # Normalize to 0-1
    
    def analyze_directness(self, text: str) -> float:
        """Calculate directness score (0-1, higher = more direct)"""
        text_lower = text.lower()
        
        direct_count = sum(1 for marker in self.direct_markers if marker in text_lower)
        hedge_count = sum(1 for hedge in self.hedge_words if hedge in text_lower)
        
        # Imperative sentences are more direct
        is_imperative = text.strip().endswith('.') and not text.lower().startswith(('i', 'we', 'you'))
        imperative_score = 0.3 if is_imperative else 0
        
        directness = (direct_count * 2 - hedge_count) / max(len(text.split()), 1) + imperative_score
        return max(0, min(1, directness + 0.3))  # Normalize to 0-1

class MessageRewriter:
    """Generates culturally appropriate rewrites"""
    
    def __init__(self):
        self.rewrite_templates = {
            "make_polite": {
                "request": "Could you please {action} when convenient?",
                "order": "I would appreciate if you could {action}.",
                "suggestion": "You might want to consider {action}."
            },
            "make_direct": {
                "request": "Please {action}.",
                "order": "{action} by {deadline}.",
                "suggestion": "I recommend {action}."
            },
            "add_formality": {
                "request": "I would be grateful if you could {action}.",
                "order": "It would be appreciated if you could {action}.",
                "general": "I hope this message finds you well. {original}"
            }
        }
    
    def extract_action(self, text: str) -> str:
        """Extract the main action from text"""
        # Simple extraction - in a real system, this would be more sophisticated
        text = text.strip()
        if text.endswith('.'):
            text = text[:-1]
        
        # Remove common prefixes
        prefixes = ['could you', 'please', 'can you', 'would you', 'i need you to']
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text if text else "complete the task"
    
    def rewrite_for_culture(self, original_text: str, speech_act: SpeechAct, 
                           target_culture: str, cultural_graph: CulturalKnowledgeGraph) -> str:
        """Rewrite text to be appropriate for target culture"""
        profile = cultural_graph.get_profile(target_culture)
        if not profile:
            return original_text
        
        action = self.extract_action(original_text)
        
        # High formality cultures (Japanese, Indian)
        if profile.formality_preference > 0.7:
            if speech_act == SpeechAct.ORDER:
                return f"I would be most grateful if you could {action}. Thank you for your consideration."
            elif speech_act == SpeechAct.REQUEST:
                return f"Could you please {action} at your earliest convenience? I would greatly appreciate it."
        
        # Low directness tolerance cultures
        elif profile.directness_tolerance < 0.4:
            if speech_act == SpeechAct.ORDER:
                return f"When you have a moment, could you please {action}?"
            elif speech_act == SpeechAct.REQUEST:
                return f"I was wondering if you might be able to {action}."
        
        # High directness tolerance cultures
        elif profile.directness_tolerance > 0.7:
            return f"Please {action}."
        
        return original_text

class CrossCulturalMiscommunicationIdentifier:
    """Main system class"""
    
    def __init__(self):
        self.cultural_graph = CulturalKnowledgeGraph()
        self.speech_act_classifier = SpeechActClassifier()
        self.politeness_analyzer = PolitenessAnalyzer()
        self.message_rewriter = MessageRewriter()
        
        print("Cross-Cultural Miscommunication Identifier initialized successfully!")
    
    def analyze_message(self, text: str, sender_culture: str, receiver_culture: str,
                       sender_role: str = "colleague", receiver_role: str = "colleague") -> MiscommunicationAlert:
        """Analyze a message for cross-cultural miscommunication risks"""
        
        # Step 1: Classify speech act
        speech_act = self.speech_act_classifier.classify(text)
        
        # Step 2: Analyze politeness and directness
        politeness_score = self.politeness_analyzer.analyze_politeness(text)
        directness_score = self.politeness_analyzer.analyze_directness(text)
        
        # Step 3: Calculate cultural mismatch
        mismatch_score = self.cultural_graph.calculate_mismatch_score(
            sender_culture, receiver_culture, speech_act.value, directness_score
        )
        
        # Step 4: Determine risk level
        if mismatch_score < 0.3:
            risk_level = RiskLevel.LOW
        elif mismatch_score < 0.6:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.HIGH
        
        # Step 5: Generate explanation
        explanation = self._generate_explanation(
            speech_act, sender_culture, receiver_culture, 
            directness_score, politeness_score, mismatch_score
        )
        
        # Step 6: Generate rewrite
        suggested_rewrite = self.message_rewriter.rewrite_for_culture(
            text, speech_act, receiver_culture, self.cultural_graph
        )
        
        return MiscommunicationAlert(
            original_text=text,
            risk_level=risk_level,
            explanation=explanation,
            suggested_rewrite=suggested_rewrite,
            confidence_score=min(0.9, 0.6 + mismatch_score * 0.3)
        )
    
    def _generate_explanation(self, speech_act: SpeechAct, sender_culture: str,
                             receiver_culture: str, directness_score: float,
                             politeness_score: float, mismatch_score: float) -> str:
        """Generate human-readable explanation"""
        
        receiver_profile = self.cultural_graph.get_profile(receiver_culture)
        if not receiver_profile:
            return "Unable to analyze - receiver culture not recognized."
        
        explanations = []
        
        # Directness mismatch
        if directness_score > receiver_profile.directness_tolerance + 0.3:
            explanations.append(
                f"This message may appear too direct for {receiver_profile.name} culture, "
                f"where indirect communication is preferred."
            )
        
        # Formality mismatch
        if politeness_score < receiver_profile.formality_preference - 0.3:
            explanations.append(
                f"The tone may be too informal for {receiver_profile.name} culture, "
                f"which values formal communication."
            )
        
        # Hierarchy mismatch
        if speech_act in [SpeechAct.ORDER, SpeechAct.REQUEST] and receiver_profile.hierarchy_respect > 0.7:
            explanations.append(
                f"In {receiver_profile.name} culture, commands should be phrased more respectfully "
                f"to acknowledge hierarchy."
            )
        
        if not explanations:
            explanations.append("Minor cultural differences detected - message should be generally acceptable.")
        
        return " ".join(explanations)
    
    def batch_analyze(self, messages: List[Dict]) -> List[MiscommunicationAlert]:
        """Analyze multiple messages"""
        results = []
        for msg in messages:
            alert = self.analyze_message(
                msg['text'], msg['sender_culture'], msg['receiver_culture'],
                msg.get('sender_role', 'colleague'), msg.get('receiver_role', 'colleague')
            )
            results.append(alert)
        return results

def demo_system():
    """Demonstration of the system"""
    print("\n" + "="*60)
    print("CROSS-CULTURAL MISCOMMUNICATION IDENTIFIER DEMO")
    print("="*60)
    
    # Initialize system
    system = CrossCulturalMiscommunicationIdentifier()
    
    # Test messages
    test_messages = [
        {
            "text": "Submit the report by 5 PM today.",
            "sender_culture": "american",
            "receiver_culture": "japanese",
            "context": "Manager to subordinate"
        },
        {
            "text": "I was wondering if you might possibly be able to help me with this task when you have some free time.",
            "sender_culture": "british",
            "receiver_culture": "german",
            "context": "Colleague to colleague"
        },
        {
            "text": "Please send me the data immediately.",
            "sender_culture": "german",
            "receiver_culture": "indian",
            "context": "Team lead to team member"
        },
        {
            "text": "Could you please review this document at your earliest convenience? Thank you very much.",
            "sender_culture": "indian",
            "receiver_culture": "american",
            "context": "Subordinate to manager"
        }
    ]
    
    print(f"\nAnalyzing {len(test_messages)} test messages...\n")
    
    for i, msg in enumerate(test_messages, 1):
        print(f"Test Case {i}: {msg['context']}")
        print(f"From: {msg['sender_culture'].title()} → To: {msg['receiver_culture'].title()}")
        print(f"Original: \"{msg['text']}\"")
        
        # Analyze message
        alert = system.analyze_message(
            msg['text'], msg['sender_culture'], msg['receiver_culture']
        )
        
        # Display results
        print(f"Risk Level: {alert.risk_level.value.upper()}")
        print(f"Confidence: {alert.confidence_score:.2f}")
        print(f"Explanation: {alert.explanation}")
        print(f"Suggested Rewrite: \"{alert.suggested_rewrite}\"")
        print("-" * 60)

def interactive_mode():
    """Interactive mode for testing"""
    system = CrossCulturalMiscommunicationIdentifier()
    
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    
    cultures = list(system.cultural_graph.cultural_profiles.keys())
    print(f"Available cultures: {', '.join(cultures)}")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            # Get input
            text = input("Enter message to analyze: ").strip()
            if text.lower() == 'quit':
                break
            
            sender = input("Sender culture: ").strip().lower()
            receiver = input("Receiver culture: ").strip().lower()
            
            if sender not in cultures or receiver not in cultures:
                print(f"Please use cultures from: {', '.join(cultures)}\n")
                continue
            
            # Analyze
            alert = system.analyze_message(text, sender, receiver)
            
            # Display results
            print(f"\nRisk Level: {alert.risk_level.value.upper()}")
            print(f"Explanation: {alert.explanation}")
            print(f"Suggested Rewrite: \"{alert.suggested_rewrite}\"")
            print(f"Confidence: {alert.confidence_score:.2f}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    print("Cross-Cultural Miscommunication Identifier")
    print("Authors: NITHIES S (22MIA1025), ABISHEK B (22MIA1075)")
    print("Course: SWE1017, Slot: F2\n")
    
    # Run demo
    demo_system()
    
    # Ask if user wants interactive mode
    choice = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
    if choice == 'y':
        interactive_mode()
    
    print("\nThank you for using the Cross-Cultural Miscommunication Identifier!")