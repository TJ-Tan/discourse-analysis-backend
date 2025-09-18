# Comprehensive metrics configuration for discourse analysis
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class MetricThreshold:
    """Defines thresholds for scoring metrics"""
    excellent: float  # Score 9-10
    good: float      # Score 7-8  
    average: float   # Score 5-6
    poor: float      # Score 3-4
    # Below poor is 1-2

@dataclass
class MetricConfig:
    """Configuration for a specific metric"""
    name: str
    description: str
    weight: float  # Weight in overall category score
    thresholds: MetricThreshold
    unit: str = ""
    evidence_required: bool = True

# Speech Analysis Metrics
SPEECH_METRICS = {
    "speaking_rate": MetricConfig(
        name="Speaking Rate",
        description="Optimal speaking pace for comprehension (120-180 WPM)",
        weight=0.25,
        thresholds=MetricThreshold(
            excellent=160,  # 140-180 WPM
            good=140,       # 120-200 WPM  
            average=120,    # 100-220 WPM
            poor=100       # Below 100 or above 220
        ),
        unit="words per minute",
        evidence_required=True
    ),
    
    "filler_ratio": MetricConfig(
        name="Filler Word Usage",
        description="Frequency of filler words (um, uh, like, etc.)",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=0.02,  # <2% filler words
            good=0.05,       # <5% filler words
            average=0.08,    # <8% filler words  
            poor=0.12        # >12% is problematic
        ),
        unit="percentage",
        evidence_required=True
    ),
    
    "speaking_clarity": MetricConfig(
        name="Speech Clarity",
        description="Articulation and pronunciation quality",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=0.95,  # 95%+ transcription confidence
            good=0.90,       # 90%+ confidence
            average=0.85,    # 85%+ confidence
            poor=0.80        # Below 80% indicates unclear speech
        ),
        unit="confidence score",
        evidence_required=True
    ),
    
    "voice_variety": MetricConfig(
        name="Voice Variety",
        description="Variation in pitch, pace, and volume",
        weight=0.15,
        thresholds=MetricThreshold(
            excellent=0.8,   # High variation in speech patterns
            good=0.6,        # Moderate variation
            average=0.4,     # Some variation
            poor=0.2         # Monotone delivery
        ),
        unit="variation index",
        evidence_required=True
    ),
    
    "pause_effectiveness": MetricConfig(
        name="Strategic Pauses",
        description="Effective use of pauses for emphasis and comprehension",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=0.8,   # Well-timed pauses
            good=0.6,        # Good pause usage
            average=0.4,     # Adequate pauses
            poor=0.2         # Poor pause timing
        ),
        unit="effectiveness score",
        evidence_required=True
    )
}

# Visual/Body Language Metrics
VISUAL_METRICS = {
    "eye_contact": MetricConfig(
        name="Eye Contact",
        description="Frequency and quality of eye contact with audience/camera",
        weight=0.25,
        thresholds=MetricThreshold(
            excellent=8.5,   # Consistent, natural eye contact
            good=7.0,        # Good eye contact with minor lapses
            average=5.5,     # Adequate eye contact
            poor=4.0         # Poor eye contact, often looking away
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "gesture_effectiveness": MetricConfig(
        name="Hand Gestures",
        description="Use of purposeful hand gestures to support content",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=8.5,   # Natural, supportive gestures
            good=7.0,        # Good use of gestures
            average=5.5,     # Some gestures, could improve
            poor=4.0         # Distracting or absent gestures
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "posture_confidence": MetricConfig(
        name="Posture & Stance",
        description="Professional posture and confident body positioning",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=8.5,   # Upright, confident posture
            good=7.0,        # Good posture with minor issues
            average=5.5,     # Adequate posture
            poor=4.0         # Poor posture, slouching
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "facial_engagement": MetricConfig(
        name="Facial Expressions",
        description="Appropriate facial expressions and engagement",
        weight=0.15,
        thresholds=MetricThreshold(
            excellent=8.5,   # Animated, engaging expressions
            good=7.0,        # Good facial engagement
            average=5.5,     # Neutral expressions
            poor=4.0         # Flat or inappropriate expressions
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "movement_purpose": MetricConfig(
        name="Movement & Positioning",
        description="Purposeful movement and spatial awareness",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=8.5,   # Strategic, purposeful movement
            good=7.0,        # Good movement patterns
            average=5.5,     # Limited but appropriate movement
            poor=4.0         # Distracting or absent movement
        ),
        unit="score (1-10)",
        evidence_required=True
    )
}

# Pedagogical Effectiveness Metrics
PEDAGOGY_METRICS = {
    "content_organization": MetricConfig(
        name="Content Structure",
        description="Logical flow and organization of material",
        weight=0.25,
        thresholds=MetricThreshold(
            excellent=8.5,   # Clear, logical progression
            good=7.0,        # Well-organized content
            average=5.5,     # Adequate organization
            poor=4.0         # Confusing or poor structure
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "concept_clarity": MetricConfig(
        name="Explanation Quality",
        description="Clarity and effectiveness of explanations",
        weight=0.25,
        thresholds=MetricThreshold(
            excellent=8.5,   # Crystal clear explanations
            good=7.0,        # Clear explanations
            average=5.5,     # Adequate clarity
            poor=4.0         # Confusing explanations
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "example_usage": MetricConfig(
        name="Use of Examples",
        description="Effective use of examples and analogies",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=8.5,   # Excellent, relevant examples
            good=7.0,        # Good examples provided
            average=5.5,     # Some examples given
            poor=4.0         # Few or poor examples
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "student_engagement": MetricConfig(
        name="Engagement Techniques",
        description="Methods used to maintain student interest and participation",
        weight=0.20,
        thresholds=MetricThreshold(
            excellent=8.5,   # Multiple engagement strategies
            good=7.0,        # Good engagement techniques
            average=5.5,     # Some engagement efforts
            poor=4.0         # Limited engagement
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "knowledge_checking": MetricConfig(
        name="Comprehension Checks",
        description="Verification of student understanding",
        weight=0.10,
        thresholds=MetricThreshold(
            excellent=8.5,   # Regular comprehension checks
            good=7.0,        # Some checking for understanding
            average=5.5,     # Limited comprehension checks
            poor=4.0         # No comprehension verification
        ),
        unit="score (1-10)",
        evidence_required=True
    )
}

def calculate_metric_score(value: float, metric: MetricConfig, reverse_scale: bool = False) -> float:
    """
    Calculate score based on metric thresholds
    
    Args:
        value: The measured value
        metric: Metric configuration
        reverse_scale: True for metrics where lower values are better (e.g., filler_ratio)
    """
    thresholds = metric.thresholds
    
    if reverse_scale:
        # For metrics where lower is better (like filler_ratio)
        if value <= thresholds.excellent:
            return 10.0
        elif value <= thresholds.good:
            return 8.5
        elif value <= thresholds.average:
            return 6.5
        elif value <= thresholds.poor:
            return 4.5
        else:
            return 2.0
    else:
        # For metrics where higher is better
        if value >= thresholds.excellent:
            return 10.0
        elif value >= thresholds.good:
            return 8.5
        elif value >= thresholds.average:
            return 6.5
        elif value >= thresholds.poor:
            return 4.5
        else:
            return 2.0

def get_metric_feedback(value: float, metric: MetricConfig, reverse_scale: bool = False) -> str:
    """Generate feedback based on metric performance"""
    score = calculate_metric_score(value, metric, reverse_scale)
    
    if score >= 9:
        return f"Excellent {metric.name.lower()} - maintain this high standard"
    elif score >= 7:
        return f"Good {metric.name.lower()} with room for minor improvements"
    elif score >= 5:
        return f"Average {metric.name.lower()} - focus on improvement strategies"
    else:
        return f"Significant improvement needed in {metric.name.lower()}"

# Overall category weights
CATEGORY_WEIGHTS = {
    "speech_analysis": 0.30,
    "body_language": 0.25, 
    "teaching_effectiveness": 0.35,
    "presentation_skills": 0.10
}