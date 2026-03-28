"""
Web Interface for Cross-Cultural Miscommunication Identifier
Flask-based web application for easy interaction with the system.
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import our main system
try:
    from main_system import CrossCulturalMiscommunicationIdentifier, RiskLevel
except ImportError:
    print("Please ensure main_system.py is in the same directory")
    exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize the system
system = CrossCulturalMiscommunicationIdentifier()

@app.route('/')
def index():
    """Main page"""
    cultures = list(system.cultural_graph.cultural_profiles.keys())
    return render_template('index.html', cultures=cultures)

@app.route('/api/analyze', methods=['POST'])
def analyze_message():
    """API endpoint to analyze a message"""
    try:
        data = request.get_json()
        
        text = data.get('text', '').strip()
        sender_culture = data.get('sender_culture', '').lower()
        receiver_culture = data.get('receiver_culture', '').lower()
        sender_role = data.get('sender_role', 'colleague')
        receiver_role = data.get('receiver_role', 'colleague')
        
        if not all([text, sender_culture, receiver_culture]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Analyze the message
        alert = system.analyze_message(
            text, sender_culture, receiver_culture, sender_role, receiver_role
        )
        
        # Prepare response
        response = {
            'original_text': alert.original_text,
            'risk_level': alert.risk_level.value,
            'explanation': alert.explanation,
            'suggested_rewrite': alert.suggested_rewrite,
            'confidence_score': round(alert.confidence_score, 2),
            'sender_culture': sender_culture.title(),
            'receiver_culture': receiver_culture.title(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cultures')
def get_cultures():
    """Get available cultures and their profiles"""
    cultures_data = {}
    for name, profile in system.cultural_graph.cultural_profiles.items():
        cultures_data[name] = {
            'name': profile.name,
            'power_distance': profile.power_distance,
            'individualism': profile.individualism,
            'directness_tolerance': profile.directness_tolerance,
            'formality_preference': profile.formality_preference,
            'hierarchy_respect': profile.hierarchy_respect
        }
    return jsonify(cultures_data)

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple messages at once"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        results = []
        for msg in messages:
            try:
                alert = system.analyze_message(
                    msg['text'], 
                    msg['sender_culture'], 
                    msg['receiver_culture'],
                    msg.get('sender_role', 'colleague'),
                    msg.get('receiver_role', 'colleague')
                )
                
                results.append({
                    'original_text': alert.original_text,
                    'risk_level': alert.risk_level.value,
                    'explanation': alert.explanation,
                    'suggested_rewrite': alert.suggested_rewrite,
                    'confidence_score': round(alert.confidence_score, 2)
                })
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cultural_comparison')
def cultural_comparison():
    """Compare how different cultures would interpret the same message"""
    text = request.args.get('text')
    sender = request.args.get('sender', 'american')
    
    if not text:
        return jsonify({'error': 'Text parameter required'}), 400
    
    cultures = list(system.cultural_graph.cultural_profiles.keys())
    results = {}
    
    for receiver in cultures:
        if receiver != sender:
            try:
                alert = system.analyze_message(text, sender, receiver)
                results[receiver] = {
                    'risk_level': alert.risk_level.value,
                    'confidence': round(alert.confidence_score, 2),
                    'explanation': alert.explanation,
                    'rewrite': alert.suggested_rewrite
                }
            except Exception as e:
                results[receiver] = {'error': str(e)}
    
    return jsonify(results)

@app.route('/demo')
def demo():
    """Demo page with pre-loaded examples"""
    demo_cases = [
        {
            'text': 'Submit the report by 5 PM today.',
            'sender': 'american',
            'receiver': 'japanese',
            'description': 'Direct command from American to Japanese colleague'
        },
        {
            'text': 'I was wondering if you might possibly help me with this task.',
            'sender': 'british',
            'receiver': 'german',
            'description': 'Overly polite British request to direct German colleague'
        },
        {
            'text': 'This approach is completely wrong.',
            'sender': 'german',
            'receiver': 'indian',
            'description': 'Direct German criticism to hierarchy-conscious Indian'
        }
    ]
    
    return render_template('demo.html', demo_cases=demo_cases)

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/api/stats')
def get_stats():
    """Get system statistics and cultural profiles visualization"""
    try:
        # Generate cultural comparison chart
        plt.figure(figsize=(12, 8))
        
        cultures = list(system.cultural_graph.cultural_profiles.keys())
        dimensions = ['power_distance', 'individualism', 'directness_tolerance', 
                     'formality_preference', 'hierarchy_respect']
        
        # Create radar chart data
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(len(dimensions)) * 2 * 3.14159 for n in range(len(dimensions))]
        angles += angles[:1]  # Complete the circle
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, culture in enumerate(cultures):
            profile = system.cultural_graph.cultural_profiles[culture]
            values = [
                profile.power_distance,
                profile.individualism,
                profile.directness_tolerance,
                profile.formality_preference,
                profile.hierarchy_respect
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=culture.title(), color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.replace('_', ' ').title() for d in dimensions])
        ax.set_ylim(0, 1)
        ax.set_title('Cultural Dimensions Comparison', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)
        
        # Save plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'cultural_chart': img_str,
            'total_cultures': len(cultures),
            'supported_dimensions': len(dimensions),
            'cultures': cultures
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HTML Templates (inline for simplicity)
@app.before_first_request
def create_templates():
    """Create template files"""
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Cross-Cultural Miscommunication Identifier{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .risk-high { background-color: #ffebee; border-left: 4px solid #f44336; }
        .risk-medium { background-color: #fff3e0; border-left: 4px solid #ff9800; }
        .risk-low { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
        .culture-card { transition: transform 0.2s; }
        .culture-card:hover { transform: translateY(-2px); }
        .analysis-result { margin-top: 20px; padding: 20px; border-radius: 8px; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-globe"></i> Cross-Cultural AI
            </a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Home</a>
                <a class="nav-link" href="/demo">Demo</a>
                <a class="nav-link" href="/about">About</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    with open('templates/base.html', 'w') as f:
        f.write(base_template)
    
    # Main page template
    index_template = '''{% extends "base.html" %}
{% block content %}
<div class="row">
    <div class="col-md-8 mx-auto">
        <div class="text-center mb-4">
            <h1><i class="fas fa-comments"></i> Cross-Cultural Miscommunication Identifier</h1>
            <p class="lead">Detect and resolve cross-cultural communication issues in real-time</p>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-edit"></i> Analyze Your Message</h5>
            </div>
            <div class="card-body">
                <form id="analysisForm">
                    <div class="mb-3">
                        <label for="messageText" class="form-label">Message Text</label>
                        <textarea class="form-control" id="messageText" rows="3" 
                                placeholder="Enter the message you want to analyze..."></textarea>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <label for="senderCulture" class="form-label">Sender Culture</label>
                            <select class="form-select" id="senderCulture">
                                {% for culture in cultures %}
                                <option value="{{ culture }}">{{ culture.title() }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label for="receiverCulture" class="form-label">Receiver Culture</label>
                            <select class="form-select" id="receiverCulture">
                                {% for culture in cultures %}
                                <option value="{{ culture }}">{{ culture.title() }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                    
                    <div class="text-center mt-3">
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="fas fa-search"></i> Analyze Message
                        </button>
                    </div>
                </form>
            </div>
        </div>
        
        <div id="analysisResult" style="display: none;"></div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
$('#analysisForm').on('submit', function(e) {
    e.preventDefault();
    
    const data = {
        text: $('#messageText').val(),
        sender_culture: $('#senderCulture').val(),
        receiver_culture: $('#receiverCulture').val()
    };
    
    if (!data.text.trim()) {
        alert('Please enter a message to analyze');
        return;
    }
    
    $.ajax({
        url: '/api/analyze',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(data),
        success: function(result) {
            displayResult(result);
        },
        error: function(xhr) {
            alert('Error: ' + xhr.responseJSON.error);
        }
    });
});

function displayResult(result) {
    const riskClass = 'risk-' + result.risk_level;
    const riskIcon = result.risk_level === 'high' ? 'fas fa-exclamation-triangle' :
                     result.risk_level === 'medium' ? 'fas fa-exclamation-circle' :
                     'fas fa-check-circle';
    
    const html = `
        <div class="analysis-result ${riskClass}">
            <h5><i class="${riskIcon}"></i> Analysis Results</h5>
            <div class="row">
                <div class="col-md-6">
                    <strong>Risk Level:</strong> 
                    <span class="badge bg-${result.risk_level === 'high' ? 'danger' : 
                                          result.risk_level === 'medium' ? 'warning' : 'success'}">
                        ${result.risk_level.toUpperCase()}
                    </span>
                </div>
                <div class="col-md-6">
                    <strong>Confidence:</strong> ${result.confidence_score}
                </div>
            </div>
            <div class="mt-3">
                <strong>Explanation:</strong>
                <p>${result.explanation}</p>
            </div>
            <div class="mt-3">
                <strong>Suggested Rewrite:</strong>
                <div class="alert alert-info">
                    "${result.suggested_rewrite}"
                </div>
            </div>
        </div>
    `;
    
    $('#analysisResult').html(html).show();
}
</script>
{% endblock %}'''
    
    with open('templates/index.html', 'w') as f:
        f.write(index_template)

if __name__ == '__main__':
    # Create templates on startup
    create_templates()
    
    print("Starting Cross-Cultural Miscommunication Identifier Web Server...")
    print("Visit http://localhost:5000 to use the application")
    
    app.run(debug=True, host='0.0.0.0', port=5000)