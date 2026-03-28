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
        # Profanity/slang normalization map
        self.profanity_map = {
            r'\bdamn\b': '', r'\bhell\b': '', r'\bcrap\b': '', r'\bstupid\b': '',
            r'\bidiot\b': '', r'\bshut up\b': 'please be quiet',
            r'\bscrewed\b': 'in a difficult situation', r'\bpissed\b': 'upset',
            r'\bwhat the\b': '', r'\bwtf\b': '', r'\basap\b': 'as soon as possible',
            r'\bfyi\b': 'for your information', r'\bbtw\b': 'by the way',
            r'\bOMG\b': '', r'\bomg\b': '',
        }

        # Common typo corrections
        self.typo_map = {
            r'\bthin\b': 'thing', r'\bthng\b': 'thing', r'\bteh\b': 'the',
            r'\brecieve\b': 'receive', r'\boccured\b': 'occurred',
            r'\bseperate\b': 'separate', r'\bdefinate\b': 'definite',
            r'\bwierd\b': 'weird', r'\bbelive\b': 'believe',
        }

        # Aggressive / urgent phrase softening
        self.aggressive_map = {
            r'\bright now\b': 'at your earliest convenience',
            r'\bimmediately\b': 'as soon as possible',
            r'\binstantly\b': 'promptly',
            r'\bthis instant\b': 'at your earliest convenience',
            r'\bdo it now\b': 'please attend to this',
            r'\bget it done\b': 'please complete this',
            r'\bhurry up\b': 'please prioritize this',
            r'\bI hope you do\b': 'I kindly request that you',
            r'\byou must\b': 'I would appreciate if you could',
            r'\byou need to\b': 'could you please',
            r'\bdo that\b': 'handle this matter',
            r'\bdo this\b': 'attend to this',
        }

    def _clean_text(self, text: str) -> str:
        """Remove profanity, fix typos, soften aggressive phrases."""
        cleaned = text.strip()

        # Fix typos first
        for pattern, replacement in self.typo_map.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Remove / replace profanity
        for pattern, replacement in self.profanity_map.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Soften aggressive language
        for pattern, replacement in self.aggressive_map.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Clean up multiple spaces left by removals
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

        # Ensure the sentence ends with a period
        if cleaned and cleaned[-1] not in '.!?':
            cleaned += '.'

        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _build_formal_rewrite(self, cleaned: str, speech_act: SpeechAct,
                               profile: 'CulturalProfile') -> str:
        """Wrap cleaned text in a culture-appropriate structure."""
        # Strip trailing punctuation to embed inside templates cleanly
        core = cleaned.rstrip('.!?').strip()

        # Very high formality (Japanese, Indian): full formal wrapper
        if profile.formality_preference > 0.7:
            intent = self._extract_core_intent(core)
            if speech_act in (SpeechAct.ORDER, SpeechAct.REQUEST):
                return (
                    f"I hope this message finds you well. "
                    f"I would be most grateful if you could {intent.lower()}. "
                    f"Thank you very much for your time and consideration.\n\nRespectfully,\n[Your Name]"
                )
            else:
                return (
                    f"I hope this message finds you well. "
                    f"{intent.capitalize()}. "
                    f"Thank you for your attention to this matter.\n\nRespectfully,\n[Your Name]"
                )

        # Low directness tolerance but not ultra-formal (British)
        elif profile.directness_tolerance < 0.5:
            intent = self._extract_core_intent(core)
            if speech_act in (SpeechAct.ORDER, SpeechAct.REQUEST):
                return f"When you have a moment, could you please {intent.lower()}? I would greatly appreciate it."
            else:
                return f"I was wondering if you might be able to {intent.lower()}."

        # High directness tolerance (German, American): keep it brief but polite
        elif profile.directness_tolerance > 0.7:
            intent = self._extract_core_intent(core)
            return f"Please {intent.lower()}."

        # Neutral fallback
        intent = self._extract_core_intent(core)
        return f"Could you please {intent.lower()}? Thank you."

    def _extract_core_intent(self, text: str) -> str:
        """Strip any politeness prefixes added during cleaning to get the bare action."""
        text = text.rstrip('.!?').strip()
        noise_prefixes = [
            'i kindly request that you', 'i would appreciate if you could',
            'could you please', 'please', 'i hope you', 'kindly',
        ]
        lower = text.lower()
        for prefix in noise_prefixes:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                lower = text.lower()
        return text if text else "complete this task"

    def rewrite_for_culture(self, original_text: str, speech_act: SpeechAct,
                            target_culture: str, cultural_graph: 'CulturalKnowledgeGraph') -> str:
        """Full pipeline: clean → rebuild → format for target culture."""
        profile = cultural_graph.get_profile(target_culture)
        if not profile:
            return original_text

        # Step 1: clean the raw input
        cleaned = self._clean_text(original_text)

        # Step 2: build a culturally appropriate sentence around it
        rewrite = self._build_formal_rewrite(cleaned, speech_act, profile)

        return rewrite

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
