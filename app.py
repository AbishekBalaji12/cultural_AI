"""
Cross-Cultural Miscommunication Identifier - Complete Integrated Web App

"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from flask import Flask, render_template_string, request, jsonify
import json
import os
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BACKEND SYSTEM - Enhanced Cross-Cultural Analyzer
# ============================================================================

class SpeechAct(Enum):
    REQUEST = "request"
    ORDER = "order"
    SUGGESTION = "suggestion"
    APOLOGY = "apology"
    COMPLIMENT = "compliment"
    QUESTION = "question"
    STATEMENT = "statement"
    CRITICISM = "criticism"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConfidenceBreakdown:
    overall_confidence: float
    speech_act_confidence: float
    cultural_match_confidence: float
    politeness_detection_confidence: float
    explanation_quality: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class CulturalProfile:
    name: str
    region: str
    languages: List[str]
    power_distance: float
    individualism: float
    directness_tolerance: float
    formality_preference: float
    hierarchy_respect: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class DetailedExplanation:
    summary: str
    cultural_context: str
    specific_issues: List[str]
    why_problematic: str
    impact_assessment: str
    recommendation: str
    examples: List[str]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class EnhancedMiscommunicationAlert:
    original_text: str
    risk_level: RiskLevel
    detailed_explanation: DetailedExplanation
    suggested_rewrite: str
    confidence_breakdown: ConfidenceBreakdown
    detected_speech_act: str
    cultural_dimensions_analysis: Dict
    tone_analysis: Dict
    formality_score: float
    directness_score: float
    politeness_score: float
    timestamp: str

class CulturalKnowledgeGraph:
    def __init__(self):
        self.cultural_profiles = {
            "american": CulturalProfile(
                name="American (US)", region="North America", languages=["English"],
                power_distance=0.40, individualism=0.91, directness_tolerance=0.80,
                formality_preference=0.30, hierarchy_respect=0.40
            ),
            "japanese": CulturalProfile(
                name="Japanese", region="East Asia", languages=["Japanese", "日本語"],
                power_distance=0.54, individualism=0.46, directness_tolerance=0.20,
                formality_preference=0.90, hierarchy_respect=0.85
            ),
            "german": CulturalProfile(
                name="German", region="Western Europe", languages=["German", "Deutsch"],
                power_distance=0.35, individualism=0.67, directness_tolerance=0.90,
                formality_preference=0.70, hierarchy_respect=0.50
            ),
            "indian": CulturalProfile(
                name="Indian", region="South Asia", languages=["Hindi", "English", "Tamil"],
                power_distance=0.77, individualism=0.48, directness_tolerance=0.30,
                formality_preference=0.80, hierarchy_respect=0.90
            ),
            "british": CulturalProfile(
                name="British (UK)", region="Western Europe", languages=["English"],
                power_distance=0.35, individualism=0.89, directness_tolerance=0.40,
                formality_preference=0.60, hierarchy_respect=0.40
            ),
            "chinese": CulturalProfile(
                name="Chinese", region="East Asia", languages=["Mandarin", "中文"],
                power_distance=0.80, individualism=0.20, directness_tolerance=0.25,
                formality_preference=0.85, hierarchy_respect=0.90
            ),
            "french": CulturalProfile(
                name="French", region="Western Europe", languages=["French", "Français"],
                power_distance=0.68, individualism=0.71, directness_tolerance=0.50,
                formality_preference=0.75, hierarchy_respect=0.60
            ),
            "brazilian": CulturalProfile(
                name="Brazilian", region="South America", languages=["Portuguese"],
                power_distance=0.69, individualism=0.38, directness_tolerance=0.45,
                formality_preference=0.40, hierarchy_respect=0.65
            ),
            "mexican": CulturalProfile(
                name="Mexican", region="North America", languages=["Spanish", "Español"],
                power_distance=0.81, individualism=0.30, directness_tolerance=0.35,
                formality_preference=0.70, hierarchy_respect=0.85
            ),
            "korean": CulturalProfile(
                name="South Korean", region="East Asia", languages=["Korean", "한국어"],
                power_distance=0.60, individualism=0.18, directness_tolerance=0.30,
                formality_preference=0.88, hierarchy_respect=0.88
            )
        }
    
    def get_profile(self, culture: str) -> Optional[CulturalProfile]:
        return self.cultural_profiles.get(culture.lower())
    
    def calculate_detailed_mismatch(self, sender_culture: str, receiver_culture: str,
                                   speech_act: str, directness_score: float,
                                   formality_score: float) -> Dict:
        sender = self.get_profile(sender_culture)
        receiver = self.get_profile(receiver_culture)
        
        if not sender or not receiver:
            return {"overall_mismatch": 0.5, "dimension_mismatches": {}, 
                   "sender_profile": {}, "receiver_profile": {}}
        
        mismatches = {
            "directness": abs(directness_score - receiver.directness_tolerance),
            "formality": abs(formality_score - receiver.formality_preference),
            "hierarchy": abs(sender.hierarchy_respect - receiver.hierarchy_respect)
        }
        
        overall_mismatch = (
            mismatches["directness"] * 0.35 +
            mismatches["formality"] * 0.35 +
            mismatches["hierarchy"] * 0.3
        )
        
        return {
            "overall_mismatch": min(overall_mismatch, 1.0),
            "dimension_mismatches": mismatches,
            "sender_profile": sender.to_dict(),
            "receiver_profile": receiver.to_dict()
        }

class SpeechActClassifier:
    def __init__(self):
        self.patterns = {
            SpeechAct.REQUEST: {
                "patterns": [
                    r'\b(could you|would you|can you|please|kindly)\b',
                    r'\b(i would appreciate|if possible)\b'
                ],
                "weight": 1.0
            },
            SpeechAct.ORDER: {
                "patterns": [
                    r'\b(must|need to|have to|should|required)\b',
                    r'^(submit|send|complete|finish|do|get|make)\b',
                    r'\b(immediately|asap|now|urgent)\b'
                ],
                "weight": 1.2
            },
            SpeechAct.CRITICISM: {
                "patterns": [
                    r'\b(wrong|incorrect|mistake|error|problem)\b',
                    r'\b(bad|poor|terrible|unacceptable)\b'
                ],
                "weight": 1.1
            },
            SpeechAct.SUGGESTION: {
                "patterns": [
                    r'\b(suggest|recommend|consider)\b',
                    r'\b(maybe|perhaps)\b'
                ],
                "weight": 0.9
            }
        }
    
    def classify_with_confidence(self, text: str) -> Tuple[SpeechAct, float]:
        text_lower = text.lower().strip()
        scores = {}
        max_score = 0
        
        for act, config in self.patterns.items():
            score = sum(1 for pattern in config["patterns"] if re.search(pattern, text_lower))
            weighted_score = score * config["weight"]
            scores[act] = weighted_score
            max_score = max(max_score, weighted_score)
        
        if max_score > 0:
            detected_act = max(scores, key=scores.get)
            confidence = min(0.95, (scores[detected_act] / (max_score + 1)) * 0.8 + 0.2)
            return detected_act, confidence
        return SpeechAct.STATEMENT, 0.5

class PolitenessAnalyzer:
    def __init__(self):
        self.polite_markers = {
            "please": 2.0, "thank you": 1.5, "could you": 1.8, "would you": 1.8,
            "if possible": 1.5, "i would appreciate": 2.0, "i'd appreciate": 2.0,
            "appreciate": 1.5, "grateful": 1.6, "kindly": 1.7,
            "hope": 1.0, "wanted to ask": 1.5, "if you": 1.2, "let me know": 1.2,
            "prefer": 1.0, "opinion": 1.2, "means a lot": 1.5, "support": 1.3,
            "kindness": 1.5, "suggestions": 1.2
        }
        
        self.direct_markers = {
            "must": 2.0, "need to": 1.5, "immediately": 2.0, "asap": 2.0,
            "now": 1.5, "urgent": 1.8, "give me": 1.5, "send me": 1.5
        }
        
        self.hedge_words = {
            "maybe": 1.0, "perhaps": 1.2, "possibly": 1.0, "might": 0.8,
            "could": 1.0, "would": 1.0, "wanted": 0.8, "wondering": 1.2,
            "if you have time": 1.5, "some time": 1.0
        }
    
    def analyze_comprehensive(self, text: str) -> Dict:
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        
        polite_score = sum(weight for marker, weight in self.polite_markers.items() if marker in text_lower)
        direct_score = sum(weight for marker, weight in self.direct_markers.items() if marker in text_lower)
        hedge_score = sum(weight for hedge, weight in self.hedge_words.items() if hedge in text_lower)
        
        # Better normalization
        politeness = min(1.0, (polite_score + hedge_score * 0.5) / max(word_count * 0.1, 1))
        directness = max(0, min(1, (direct_score - hedge_score * 0.3) / max(word_count * 0.15, 1)))
        
        formal_markers = ["dear", "sincerely", "regards", "respectfully", "hope this finds you", "thank you"]
        formality = min(1.0, sum(1 for m in formal_markers if m in text_lower) * 0.15 + 0.3)
        
        confidence = min(0.9, 0.5 + (polite_score + direct_score) * 0.05)
        
        return {
            "politeness_score": politeness,
            "directness_score": directness,
            "formality_score": formality,
            "polite_markers_found": [m for m in self.polite_markers if m in text_lower],
            "direct_markers_found": [m for m in self.direct_markers if m in text_lower],
            "confidence": confidence
        }

class MessageRewriter:
    def rewrite_for_culture(self, text: str, receiver_profile: CulturalProfile, 
                           speech_act: SpeechAct) -> str:
        
        # Clean the text
        text = text.strip()
        
        # Check politeness level of original
        polite_words = ['please', 'thank you', 'appreciate', 'hope', 'kindly', 'grateful', 
                       'wondering', 'could you', 'would you', 'if possible']
        polite_count = sum(1 for word in polite_words if word in text.lower())
        
        # Check if already has greeting/closing
        has_formal_greeting = text.lower().startswith(('dear', 'respected', 'hello'))
        has_polite_opening = 'hope' in text.lower()[:100]  # First 100 chars
        has_closing = any(word in text.lower()[-200:] for word in ['regards', 'sincerely', 'thank you', 'thanks'])
        
        # Check for harsh/direct language that NEEDS rewriting
        harsh_patterns = [
            r'^(send|give|complete|submit|do)\s+me',  # "Send me", "Give me"
            r'\b(immediately|asap|now|urgent)\b',      # Urgent words
            r'^(send|do|make|complete)',               # Commands at start
        ]
        needs_rewriting = any(re.search(pattern, text.lower()) for pattern in harsh_patterns)
        
        # If message is already polite AND well-structured, minimal changes
        if polite_count >= 3 and (has_formal_greeting or has_polite_opening) and has_closing and not needs_rewriting:
            
            # For Japanese->American: Message is TOO polite, make it more direct
            if receiver_profile.name == "American (US)" and polite_count > 5:
                # Simplify excessive politeness
                rewritten = text
                rewritten = re.sub(r"I hope this message finds you well\.\s*", "", rewritten)
                rewritten = re.sub(r"I really appreciate it\.", "Thanks!", rewritten)
                rewritten = re.sub(r"Your opinion means a lot to me,?\s*and\s*", "I'd appreciate ", rewritten)
                rewritten = re.sub(r"thank you so much", "thanks", rewritten, flags=re.IGNORECASE)
                rewritten = re.sub(r"I was wondering if you might possibly", "Could you", rewritten)
                
                # If it's already good, just add a note
                if rewritten == text:
                    return f"{text}\n\n[Note: Message is well-written. For American culture, you could be even more direct and concise if needed.]"
                
                return rewritten.strip()
            
            # For other high formality cultures - message is good
            elif receiver_profile.formality_preference > 0.7:
                return f"{text}\n\n[Note: Message is already well-formatted for {receiver_profile.name} culture.]"
            
            # For moderate cultures - message is good
            else:
                return f"{text}\n\n[Note: Message is appropriately polite for {receiver_profile.name} culture.]"
        
        # ONLY IF message needs real rewriting (harsh/too direct)
        lines = text.split('\n')
        clean_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith(('subject:', 'sub:', 're:'))]
        
        rewritten_parts = []
        
        # Add greeting ONLY if missing
        if not has_formal_greeting and not has_polite_opening:
            if receiver_profile.formality_preference > 0.7:
                rewritten_parts.append("Dear [Recipient Name],\n")
                rewritten_parts.append("I hope this message finds you well.\n")
            else:
                rewritten_parts.append("Hello [Recipient Name],\n")
        
        # Rewrite each line to be more polite
        for line in clean_lines:
            original_line = line
            
            # Fix direct commands
            line = re.sub(r'^(send|give|complete|submit|do)\s+me\s+', r'Could you please \1 ', line, flags=re.IGNORECASE)
            line = re.sub(r'^(send|give|complete|submit|do)\s+', r'Could you please \1 ', line, flags=re.IGNORECASE)
            
            # Replace harsh urgency
            line = re.sub(r'\b(immediately|asap|now)\b', 'at your earliest convenience', line, flags=re.IGNORECASE)
            line = re.sub(r'\burgent\b', 'time-sensitive', line, flags=re.IGNORECASE)
            
            # If line was changed, it needed rewriting
            if line != original_line or needs_rewriting:
                rewritten_parts.append(line)
            else:
                rewritten_parts.append(original_line)
        
        # Join body
        body = '\n\n'.join(rewritten_parts)
        
        # Add closing ONLY if missing
        if not has_closing:
            if receiver_profile.formality_preference > 0.7:
                body += "\n\nThank you very much for your time and consideration.\n\nRespectfully,\n[Your Name]"
            else:
                body += "\n\nThank you for your time.\n\nBest regards,\n[Your Name]"
        
        return body.strip()

class EnhancedCrossCulturalSystem:
    def __init__(self):
        self.cultural_graph = CulturalKnowledgeGraph()
        self.speech_act_classifier = SpeechActClassifier()
        self.politeness_analyzer = PolitenessAnalyzer()
        self.message_rewriter = MessageRewriter()
        print("✅ Enhanced Cross-Cultural System Initialized!")
    
    def analyze_message_enhanced(self, text: str, sender_culture: str, 
                                receiver_culture: str) -> EnhancedMiscommunicationAlert:
        # Speech act
        speech_act, speech_confidence = self.speech_act_classifier.classify_with_confidence(text)
        
        # Tone analysis
        tone_analysis = self.politeness_analyzer.analyze_comprehensive(text)
        
        # Cultural mismatch
        cultural_analysis = self.cultural_graph.calculate_detailed_mismatch(
            sender_culture, receiver_culture, speech_act.value,
            tone_analysis['directness_score'], tone_analysis['formality_score']
        )
        
        # Risk level
        mismatch_score = cultural_analysis['overall_mismatch']
        if speech_act in [SpeechAct.ORDER, SpeechAct.CRITICISM]:
            if mismatch_score > 0.6:
                risk_level = RiskLevel.CRITICAL
            elif mismatch_score > 0.4:
                risk_level = RiskLevel.HIGH
            elif mismatch_score > 0.25:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
        else:
            if mismatch_score > 0.7:
                risk_level = RiskLevel.CRITICAL
            elif mismatch_score > 0.5:
                risk_level = RiskLevel.HIGH
            elif mismatch_score > 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
        
        # Explanation
        sender_prof = cultural_analysis['sender_profile']
        receiver_prof = cultural_analysis['receiver_profile']
        mismatches = cultural_analysis['dimension_mismatches']
        
        summary = f"This {speech_act.value} from {sender_prof['name']} to {receiver_prof['name']} has {risk_level.value.upper()} risk."
        
        cultural_context = (
            f"{sender_prof['name']} ({sender_prof['region']}) values "
            f"{'high' if sender_prof['directness_tolerance'] > 0.6 else 'low'} directness. "
            f"{receiver_prof['name']} ({receiver_prof['region']}) prefers "
            f"{'indirect' if receiver_prof['directness_tolerance'] < 0.4 else 'direct'} communication."
        )
        
        specific_issues = []
        if mismatches.get('directness', 0) > 0.3:
            specific_issues.append(
                f"Directness mismatch: {tone_analysis['directness_score']:.2f} vs {receiver_prof['directness_tolerance']:.2f}"
            )
        if mismatches.get('formality', 0) > 0.3:
            specific_issues.append(
                f"Formality mismatch: {tone_analysis['formality_score']:.2f} vs {receiver_prof['formality_preference']:.2f}"
            )
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            why_problematic = "This message violates key cultural norms and may cause offense or discomfort."
        elif risk_level == RiskLevel.MEDIUM:
            why_problematic = "Minor cultural differences may create confusion."
        else:
            why_problematic = "Message is generally appropriate."
        
        impact_emojis = {
            RiskLevel.CRITICAL: "🚨 CRITICAL IMPACT",
            RiskLevel.HIGH: "⚠️ HIGH IMPACT",
            RiskLevel.MEDIUM: "⚡ MODERATE IMPACT",
            RiskLevel.LOW: "✅ LOW IMPACT"
        }
        impact_assessment = impact_emojis[risk_level]
        
        recommendations = []
        if receiver_prof['directness_tolerance'] < 0.4 and tone_analysis['directness_score'] > 0.6:
            recommendations.append("Use indirect phrasing and softer language")
        if tone_analysis['formality_score'] < receiver_prof['formality_preference'] - 0.2:
            recommendations.append("Increase formality with proper greetings")
        if not recommendations:
            recommendations.append("Message is generally appropriate")
        
        recommendation = "\n".join(f"• {r}" for r in recommendations)
        
        examples = [f"Languages: {', '.join(receiver_prof['languages'])}"]
        
        detailed_explanation = DetailedExplanation(
            summary=summary,
            cultural_context=cultural_context,
            specific_issues=specific_issues,
            why_problematic=why_problematic,
            impact_assessment=impact_assessment,
            recommendation=recommendation,
            examples=examples
        )
        
        # Rewrite
        receiver_profile_obj = self.cultural_graph.get_profile(receiver_culture)
        suggested_rewrite = self.message_rewriter.rewrite_for_culture(
            text, receiver_profile_obj, speech_act
        )
        
        # Confidence
        confidence_breakdown = ConfidenceBreakdown(
            overall_confidence=min(0.92, 0.65 + (1 - mismatch_score) * 0.25),
            speech_act_confidence=speech_confidence,
            cultural_match_confidence=1 - mismatch_score * 0.7,
            politeness_detection_confidence=tone_analysis['confidence'],
            explanation_quality=0.85
        )
        
        return EnhancedMiscommunicationAlert(
            original_text=text,
            risk_level=risk_level,
            detailed_explanation=detailed_explanation,
            suggested_rewrite=suggested_rewrite,
            confidence_breakdown=confidence_breakdown,
            detected_speech_act=speech_act.value,
            cultural_dimensions_analysis=cultural_analysis,
            tone_analysis=tone_analysis,
            formality_score=tone_analysis['formality_score'],
            directness_score=tone_analysis['directness_score'],
            politeness_score=tone_analysis['politeness_score'],
            timestamp=datetime.now().isoformat()
        )

# ============================================================================
# FLASK WEB APPLICATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cross-cultural-2025'

# Initialize system
print("Initializing system...")
system = EnhancedCrossCulturalSystem()
print("System ready!")

# HTML Templates (inline)
BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Cultural Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .main-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin: 20px auto;
            max-width: 1200px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        .hero-section {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .hero-section h1 {
            font-size: 2.5rem;
            font-weight: 700;
        }
        .form-control, .form-select {
            border-radius: 10px;
            padding: 12px;
            border: 2px solid #e0e0e0;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: 600;
        }
        .risk-badge {
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .risk-low { background: #2ecc71; color: white; }
        .risk-medium { background: #f39c12; color: white; }
        .risk-high { background: #e74c3c; color: white; }
        .risk-critical { background: #c0392b; color: white; }
        .result-card {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .progress-bar-custom {
            height: 25px;
            border-radius: 15px;
            font-weight: 600;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
    </style>
</head>
<body>
    {{ content | safe }}
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        {{ script | safe }}
    </script>
</body>
</html>
'''

INDEX_CONTENT = '''
<div class="hero-section">
    <h1><i class="fas fa-comments"></i> Cross-Cultural Communication Analyzer</h1>
    <p>Detect and resolve cross-cultural miscommunication</p>
    <p><strong>10 Cultures | 20+ Languages | AI-Powered</strong></p>
</div>

<div class="main-container">
    <h2 class="mb-4"><i class="fas fa-edit"></i> Analyze Your Message</h2>
    
    <form id="analysisForm">
        <div class="mb-4">
            <label class="form-label"><strong>Your Message or Email</strong></label>
            <textarea class="form-control" id="messageText" rows="8" 
                      placeholder="Paste your email or message here..."></textarea>
        </div>
        
        <div class="row">
            <div class="col-md-6 mb-3">
                <label class="form-label"><strong>Your Culture</strong></label>
                <select class="form-select" id="senderCulture">
                    {% for culture, info in cultures.items() %}
                    <option value="{{ culture }}">{{ info.name }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="col-md-6 mb-3">
                <label class="form-label"><strong>Receiver's Culture</strong></label>
                <select class="form-select" id="receiverCulture">
                    {% for culture, info in cultures.items() %}
                    <option value="{{ culture }}" {% if culture == 'japanese' %}selected{% endif %}>{{ info.name }}</option>
                    {% endfor %}
                </select>
            </div>
        </div>
        
        <div class="text-center">
            <button type="submit" class="btn btn-primary btn-lg">
                <i class="fas fa-search"></i> Analyze Message
            </button>
        </div>
    </form>
    
    <div class="loading" id="loading">
        <div class="spinner-border text-primary"></div>
        <p class="mt-3">Analyzing...</p>
    </div>
    
    <div id="results" style="display: none;">
        <hr class="my-5">
        <h2 class="mb-4"><i class="fas fa-chart-pie"></i> Analysis Results</h2>
        
        <div class="result-card">
            <h4><i class="fas fa-exclamation-triangle"></i> Risk Assessment</h4>
            <div class="mb-3">
                <strong>Risk Level:</strong> <span id="riskBadge"></span>
            </div>
            <div class="mb-3">
                <strong>Confidence:</strong> <span id="overallConfidence"></span>
                <div class="progress mt-2">
                    <div id="confidenceBar" class="progress-bar bg-success progress-bar-custom"></div>
                </div>
            </div>
            <div>
                <strong>Speech Act:</strong> <span id="speechAct" class="badge bg-primary"></span>
            </div>
        </div>
        
        <div class="result-card">
            <h4><i class="fas fa-chart-bar"></i> Tone Metrics</h4>
            <div class="row">
                <div class="col-md-4 mb-3">
                    <strong>Politeness</strong>
                    <div class="progress mt-2">
                        <div id="politenessBar" class="progress-bar bg-success progress-bar-custom"></div>
                    </div>
                </div>
                <div class="col-md-4 mb-3">
                    <strong>Directness</strong>
                    <div class="progress mt-2">
                        <div id="directnessBar" class="progress-bar bg-info progress-bar-custom"></div>
                    </div>
                </div>
                <div class="col-md-4 mb-3">
                    <strong>Formality</strong>
                    <div class="progress mt-2">
                        <div id="formalityBar" class="progress-bar bg-primary progress-bar-custom"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="result-card">
            <h4><i class="fas fa-globe"></i> Cultural Analysis</h4>
            <div id="culturalAnalysis"></div>
        </div>
        
        <div class="result-card">
            <h4><i class="fas fa-lightbulb"></i> Recommendations</h4>
            <div id="recommendations"></div>
        </div>
        
        <div class="result-card">
            <h4><i class="fas fa-edit"></i> Improved Version</h4>
            <div class="alert alert-success">
                <pre id="improvedMessage" style="white-space: pre-wrap; font-family: inherit; margin: 0;"></pre>
            </div>
            <button class="btn btn-outline-primary mt-2" onclick="copyImproved()">
                <i class="fas fa-copy"></i> Copy to Clipboard
            </button>
        </div>
    </div>
</div>

<div class="text-center text-white mt-4">
    <p><strong>Cross-Cultural Analyzer V2.0</strong></p>
    <p>NITHIES S (22MIA1025) | ABISHEK B (22MIA1075)</p>
</div>
'''

INDEX_SCRIPT = '''
$('#analysisForm').on('submit', function(e) {
    e.preventDefault();
    
    const message = $('#messageText').val().trim();
    if (!message) {
        alert('Please enter a message');
        return;
    }
    
    $('#results').hide();
    $('#loading').show();
    
    $.ajax({
        url: '/api/analyze',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            message: message,
            sender_culture: $('#senderCulture').val(),
            receiver_culture: $('#receiverCulture').val()
        }),
        success: function(result) {
            console.log('Success:', result);
            displayResults(result);
            $('#loading').hide();
            $('#results').show();
        },
        error: function(xhr, status, error) {
            console.error('Error:', xhr, status, error);
            $('#loading').hide();
            alert('Error: ' + (xhr.responseJSON?.error || error));
        }
    });
});

function displayResults(result) {
    const riskColors = {
        'low': 'risk-low',
        'medium': 'risk-medium',
        'high': 'risk-high',
        'critical': 'risk-critical'
    };
    $('#riskBadge').html('<span class="risk-badge ' + riskColors[result.risk_level] + '">' + result.risk_level.toUpperCase() + '</span>');
    
    const confidence = result.confidence.overall;
    $('#overallConfidence').text((confidence * 100).toFixed(0) + '%');
    $('#confidenceBar').css('width', (confidence * 100) + '%').text((confidence * 100).toFixed(0) + '%');
    
    $('#speechAct').text(result.detected_speech_act.toUpperCase());
    
    $('#politenessBar').css('width', (result.politeness_score * 100) + '%')
        .text((result.politeness_score * 100).toFixed(0) + '%');
    $('#directnessBar').css('width', (result.directness_score * 100) + '%')
        .text((result.directness_score * 100).toFixed(0) + '%');
    $('#formalityBar').css('width', (result.formality_score * 100) + '%')
        .text((result.formality_score * 100).toFixed(0) + '%');
    
    let culturalHtml = '<p><strong>Summary:</strong> ' + result.explanation.summary + '</p>';
    culturalHtml += '<p><strong>Cultural Context:</strong> ' + result.explanation.cultural_context + '</p>';
    
    if (result.explanation.specific_issues.length > 0) {
        culturalHtml += '<p><strong>Specific Issues:</strong></p><ul>';
        result.explanation.specific_issues.forEach(issue => {
            culturalHtml += '<li>' + issue + '</li>';
        });
        culturalHtml += '</ul>';
    }
    
    culturalHtml += '<p><strong>Why This Matters:</strong> ' + result.explanation.why_problematic + '</p>';
    culturalHtml += '<p><strong>Impact:</strong> ' + result.explanation.impact_assessment + '</p>';
    
    $('#culturalAnalysis').html(culturalHtml);
    
    $('#recommendations').html('<div>' + result.explanation.recommendations.replace(/\\n/g, '<br>') + '</div>');
    
    $('#improvedMessage').text(result.improved_message);
}

function copyImproved() {
    const text = $('#improvedMessage').text();
    navigator.clipboard.writeText(text).then(() => {
        alert('✅ Copied to clipboard!');
    });
}
'''

@app.route('/')
def index():
    """Main page"""
    cultures = {}
    for key, profile in system.cultural_graph.cultural_profiles.items():
        cultures[key] = {
            'name': profile.name,
            'region': profile.region
        }
    
    from jinja2 import Template
    content_template = Template(INDEX_CONTENT)
    content = content_template.render(cultures=cultures)
    
    full_template = Template(BASE_TEMPLATE)
    return full_template.render(content=content, script=INDEX_SCRIPT)

@app.route('/api/analyze', methods=['POST'])
def analyze_message():
    """Analysis endpoint"""
    try:
        data = request.get_json()
        print("Received data:", data)
        
        message = data.get('message', '').strip()
        sender_culture = data.get('sender_culture', '').lower()
        receiver_culture = data.get('receiver_culture', '').lower()
        
        if not all([message, sender_culture, receiver_culture]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        print(f"Analyzing: {sender_culture} -> {receiver_culture}")
        
        # Analyze message
        alert = system.analyze_message_enhanced(message, sender_culture, receiver_culture)
        
        print("Analysis complete!")
        
        # Prepare response
        response = {
            'success': True,
            'original_message': alert.original_text,
            'risk_level': alert.risk_level.value,
            'detected_speech_act': alert.detected_speech_act,
            'improved_message': alert.suggested_rewrite,
            
            'politeness_score': round(alert.politeness_score, 2),
            'directness_score': round(alert.directness_score, 2),
            'formality_score': round(alert.formality_score, 2),
            
            'confidence': {
                'overall': round(alert.confidence_breakdown.overall_confidence, 2),
                'speech_act': round(alert.confidence_breakdown.speech_act_confidence, 2),
                'cultural_match': round(alert.confidence_breakdown.cultural_match_confidence, 2),
                'politeness': round(alert.confidence_breakdown.politeness_detection_confidence, 2),
                'explanation_quality': round(alert.confidence_breakdown.explanation_quality, 2)
            },
            
            'explanation': {
                'summary': alert.detailed_explanation.summary,
                'cultural_context': alert.detailed_explanation.cultural_context,
                'specific_issues': alert.detailed_explanation.specific_issues,
                'why_problematic': alert.detailed_explanation.why_problematic,
                'impact_assessment': alert.detailed_explanation.impact_assessment,
                'recommendations': alert.detailed_explanation.recommendation,
                'examples': alert.detailed_explanation.examples
            },
            
            'sender_profile': alert.cultural_dimensions_analysis['sender_profile'],
            'receiver_profile': alert.cultural_dimensions_analysis['receiver_profile'],
            
            'timestamp': alert.timestamp
        }
        
        print("Response prepared, sending...")
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'Backend is working!',
        'cultures': list(system.cultural_graph.cultural_profiles.keys())
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 CROSS-CULTURAL MISCOMMUNICATION IDENTIFIER V2.0")
    print("="*80)
    print("✅ System initialized successfully!")
    print(f"✅ {len(system.cultural_graph.cultural_profiles)} cultures supported")
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🧪 Test backend at: http://localhost:5000/test")
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
