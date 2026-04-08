# Enhanced metrics configuration for discourse analysis
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

# --- MARS Rubric (Excel "Revised Rubric", v20260224) ---
# Main categories: Content 20%, Delivery 40%, Engagement 40%
MARS_RUBRIC_VERSION = "v20260224"

MARS_CONFIG = {
    "version": MARS_RUBRIC_VERSION,
    "main_categories": {
        "content": 0.20,
        "delivery": 0.40,
        "engagement": 0.40,
    },
    # Within Content (scores 0–10 each; combined with sub-weights below)
    "content_subweights": {
        "content_organisation": 0.30,
        "explanation_quality": 0.40,
        "use_of_examples_representation": 0.30,
    },
    # Criteria weights *within* each Content sub-category (sum to sub-category total)
    "content_criteria_weights": {
        "content_organisation": {
            "total": 0.30,
            "structural_sequencing": 0.10,
            "logical_consistency": 0.10,
            "closure_framing": 0.10,
        },
        "explanation_quality": {
            "total": 0.40,
            "conceptual_accuracy": 0.20,
            "causal_reasoning_depth": 0.10,
            "multi_perspective_explanation": 0.10,
        },
        "use_of_examples_representation": {
            "total": 0.30,
            "example_quality_frequency": 0.10,
            "analogy_concept_bridging": 0.10,
            "representation_diversity": 0.10,
        },
    },
    # Within Engagement main category
    "engagement_subweights": {
        "interaction_frequency": 0.40,  # Question density
        "question_quality_block": 0.40,  # CLI + SUI + QDS
        "feedback": 0.20,
    },
    # Inside question_quality_block (sums to 1.0)
    "question_quality_internal": {
        "cognitive_level_index": 0.50,  # CLI — maps from ICAP-derived question_quality
        "student_uptake_index": 0.25,
        "question_distribution_stability": 0.25,
    },
    # Inside feedback (two criteria equal)
    "feedback_internal": {
        "student_question_frequency": 0.50,
        "student_question_cognitive_level": 0.50,
    },
}


def compute_mars_delivery_category_score(speech_score: float, visual_score: float) -> float:
    """Delivery = 50% Speech Analysis + 50% Body Language (each 0–10)."""
    return 0.5 * float(speech_score) + 0.5 * float(visual_score)


def compute_mars_content_category_score_detailed(pedagogical_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Content category 0–10 from nine criteria. Sub-categories sum to 100% of Content:
    - Content Organisation 0.30 = SS 0.10 + LC 0.10 + CF 0.10
    - Explanation Quality 0.40 = Conceptual Accuracy 0.20 + Causal 0.10 + Multi-perspective 0.10
    - Use of Examples / Representation 0.30 = three criteria × 0.10 each

    Sub-score (0–10) = weighted sum of criteria / sub-category total weight.
    Final Content = 0.30×Org_sub + 0.40×Expl_sub + 0.30×Examples_sub
    (then Content is 20% of overall MARS via main_categories).
    """
    def g(key: str, default: float = 7.0) -> float:
        v = pedagogical_analysis.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    cw = MARS_CONFIG["content_criteria_weights"]
    co = cw["content_organisation"]
    org = (
        co["structural_sequencing"] * g("structural_sequencing")
        + co["logical_consistency"] * g("logical_consistency")
        + co["closure_framing"] * g("closure_framing")
    ) / co["total"]

    eq = cw["explanation_quality"]
    expl = (
        eq["conceptual_accuracy"] * g("conceptual_accuracy")
        + eq["causal_reasoning_depth"] * g("causal_reasoning_depth")
        + eq["multi_perspective_explanation"] * g("multi_perspective_explanation")
    ) / eq["total"]

    ue = cw["use_of_examples_representation"]
    exmp = (
        ue["example_quality_frequency"] * g("example_quality_frequency")
        + ue["analogy_concept_bridging"] * g("analogy_concept_bridging")
        + ue["representation_diversity"] * g("representation_diversity")
    ) / ue["total"]

    w = MARS_CONFIG["content_subweights"]
    content_before_penalty = (
        w["content_organisation"] * org
        + w["explanation_quality"] * expl
        + w["use_of_examples_representation"] * exmp
    )
    content = content_before_penalty
    # If instructor provided lecture context but the transcript appears to be about a different topic/discipline,
    # apply a deterministic penalty so "well-structured wrong-subject content" cannot score highly.
    penalty_points = 0.0
    try:
        context_provided = bool(pedagogical_analysis.get("lecture_context_provided"))
        alignment = pedagogical_analysis.get("context_alignment_score", None)
        alignment = float(alignment) if alignment is not None else None
        if context_provided and alignment is not None and alignment <= 0.25:
            penalty_points = 5.0
    except Exception:
        penalty_points = 0.0
    if penalty_points:
        content = max(0.0, float(content) - float(penalty_points))
    return {
        "content_category_score": content,
        "content_category_score_before_penalty": content_before_penalty,
        "content_context_misalignment_penalty_points": penalty_points,
        "content_organisation_score": org,
        "explanation_quality_score": expl,
        "use_of_examples_representation_score": exmp,
        "formula": (
            "Content = 0.30×Org + 0.40×Expl + 0.30×Ex; "
            "Org = (0.1×SS + 0.1×LC + 0.1×CF) / 0.3; "
            "Expl = (0.2×CA + 0.1×CR + 0.1×MP) / 0.4; "
            "Ex = (0.1×Eq + 0.1×An + 0.1×Rd) / 0.3; "
            "If context provided and alignment≤0.25: Content = max(0, Content − 5)"
        ),
    }


def compute_mars_content_category_score(pedagogical_analysis: Dict[str, Any]) -> float:
    """Content category 0–10 (single number)."""
    return float(compute_mars_content_category_score_detailed(pedagogical_analysis)["content_category_score"])


def compute_mars_engagement_category_score(interaction_analysis: Dict[str, Any]) -> float:
    """
    Engagement 40% of MARS: 40% QD + 40% (CLI/SUI/QDS block) + 20% student feedback.
    """
    qd = float(interaction_analysis.get("interaction_frequency") or 0)
    cli = float(interaction_analysis.get("question_quality") or 0)
    sui = float(interaction_analysis.get("student_uptake_index") or 0)
    qds = float(interaction_analysis.get("question_distribution_stability") or 0)
    qw = MARS_CONFIG["question_quality_internal"]
    qq_block = (
        qw["cognitive_level_index"] * cli
        + qw["student_uptake_index"] * sui
        + qw["question_distribution_stability"] * qds
    )
    sf = float(interaction_analysis.get("student_question_frequency_score") or 0)
    sc = float(interaction_analysis.get("student_question_cognitive_score") or 0)
    fb = (sf + sc) / 2.0
    ew = MARS_CONFIG["engagement_subweights"]
    return ew["interaction_frequency"] * qd + ew["question_quality_block"] * qq_block + ew["feedback"] * fb


def compute_mars_overall_score(
    content_score: float, delivery_score: float, engagement_score: float
) -> float:
    m = MARS_CONFIG["main_categories"]
    return (
        m["content"] * float(content_score)
        + m["delivery"] * float(delivery_score)
        + m["engagement"] * float(engagement_score)
    )


# Global Analysis Configuration
ANALYSIS_CONFIG = {
    "sampling": {
        "frame_interval_seconds": 30,     # Every 30 seconds
        "max_frames_analyzed": 100,       # Increased from 10 to 100
        "audio_window_seconds": 30,       # For rate analysis windows
        "transcript_char_limit": None,    # Use full transcript (was 3000)
        "min_frames_for_analysis": 5      # Minimum frames needed
    },
    
    "weights": {
        # Overall category weights - All set to 20% (0.20) for equal weighting
        "speech_analysis": 0.20,
        "body_language": 0.20, 
        "teaching_effectiveness": 0.20,
        "interaction_engagement": 0.20, 
        "presentation_skills": 0.20,
        
        # Speech analysis sub-components
        "speech_components": {
            "speaking_rate": 0.25,
            "clarity": 0.25,
            "confidence": 0.20,
            "voice_variety": 0.15,
            "pause_effectiveness": 0.15
        },
        
        # Visual analysis sub-components
        "visual_components": {
            "eye_contact": 0.25,
            "gestures": 0.20,
            "posture": 0.20,
            "engagement": 0.20,
            "professionalism": 0.15
        },
        
        # Pedagogical effectiveness sub-components
        "pedagogy_components": {
            "content_organization": 0.25,
            "engagement_techniques": 0.20,
            "communication_clarity": 0.20,
            "use_of_examples": 0.20,
            "knowledge_checking": 0.15
        }
    },
    
    "thresholds": {
        "speaking_rate": {
            "optimal_min": 140,
            "optimal_max": 180,
            "acceptable_min": 120,
            "acceptable_max": 200,
            "poor_threshold": 100
        },
        "filler_ratio": {
            "excellent": 0.02,    # <2%
            "good": 0.05,         # <5%
            "average": 0.08,      # <8%
            "poor": 0.12          # >12%
        },
        "confidence_threshold": {
            "excellent": 0.95,
            "good": 0.90,
            "average": 0.85,
            "poor": 0.80
        },
        "visual_scores": {
            "excellent": 8.5,
            "good": 7.0,
            "average": 5.5,
            "poor": 4.0
        }
    }
}

# Expanded filler word list
FILLER_WORDS = [
    # Basic fillers
    'um', 'uh', 'er', 'ah', 'eh', 'mm', 'hmm',
    
    # Common verbal fillers
    'like', 'you know', 'so', 'well', 'okay', 'right', 'actually',
    'basically', 'literally', 'obviously', 'clearly', 'essentially',
    
    # Hesitation markers
    'let me see', 'how do i say', 'what i mean is', 'sort of', 'kind of',
    'i mean', 'you see', 'as i was saying', 'where was i',
    
    # Thinking sounds
    'lets see', 'now', 'so anyway', 'moving on', 'next', 'furthermore',
    
    # Regional/informal
    'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'yep', 'nah',
    
    # Academic fillers
    'as you can see', 'as we discussed', 'moving forward', 'that being said',
    'in other words', 'to put it simply', 'if you will', 'so to speak'
]

# Speech Analysis Metrics with configurable thresholds
SPEECH_METRICS = {
    "speaking_rate": MetricConfig(
        name="Speaking Rate",
        description="Optimal speaking pace for comprehension",
        weight=ANALYSIS_CONFIG["weights"]["speech_components"]["speaking_rate"],
        thresholds=MetricThreshold(
            excellent=160,  # 140-180 WPM optimal
            good=140,       # 120-200 WPM acceptable
            average=120,    # 100-220 WPM marginal
            poor=100       # Below 100 or above 220
        ),
        unit="words per minute",
        evidence_required=True
    ),
    
    "filler_ratio": MetricConfig(
        name="Speech Fluency",
        description="Frequency of filler words and hesitations",
        weight=ANALYSIS_CONFIG["weights"]["speech_components"]["clarity"],
        thresholds=MetricThreshold(
            excellent=0.02,  # <2% filler words
            good=0.05,       # <5% filler words
            average=0.08,    # <8% filler words  
            poor=0.12        # >12% problematic
        ),
        unit="percentage",
        evidence_required=True
    ),
    
    "speaking_clarity": MetricConfig(
        name="Articulation Quality",
        description="Clarity of pronunciation and enunciation",
        weight=ANALYSIS_CONFIG["weights"]["speech_components"]["confidence"],
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
        name="Voice Modulation",
        description="Variation in pitch, pace, and volume for engagement",
        weight=ANALYSIS_CONFIG["weights"]["speech_components"]["voice_variety"],
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
        name="Strategic Pausing",
        description="Effective use of pauses for emphasis and comprehension",
        weight=ANALYSIS_CONFIG["weights"]["speech_components"]["pause_effectiveness"],
        thresholds=MetricThreshold(
            excellent=0.8,   # Well-timed strategic pauses
            good=0.6,        # Good pause usage
            average=0.4,     # Adequate pauses
            poor=0.2         # Poor pause timing or excessive gaps
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
        weight=ANALYSIS_CONFIG["weights"]["visual_components"]["eye_contact"],
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
        weight=ANALYSIS_CONFIG["weights"]["visual_components"]["gestures"],
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
        weight=ANALYSIS_CONFIG["weights"]["visual_components"]["posture"],
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
        weight=ANALYSIS_CONFIG["weights"]["visual_components"]["engagement"],
        thresholds=MetricThreshold(
            excellent=8.5,   # Animated, engaging expressions
            good=7.0,        # Good facial engagement
            average=5.5,     # Neutral expressions
            poor=4.0         # Flat or inappropriate expressions
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "professionalism": MetricConfig(
        name="Professional Appearance",
        description="Overall professional presentation and appearance",
        weight=ANALYSIS_CONFIG["weights"]["visual_components"]["professionalism"],
        thresholds=MetricThreshold(
            excellent=8.5,   # Highly professional appearance
            good=7.0,        # Good professional standards
            average=5.5,     # Adequate professionalism
            poor=4.0         # Unprofessional appearance
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
        weight=ANALYSIS_CONFIG["weights"]["pedagogy_components"]["content_organization"],
        thresholds=MetricThreshold(
            excellent=8.5,   # Clear, logical progression with signposting
            good=7.0,        # Well-organized content
            average=5.5,     # Adequate organization
            poor=4.0         # Confusing or poor structure
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "engagement_techniques": MetricConfig(
        name="Student Engagement",
        description="Methods used to maintain student interest and participation",
        weight=ANALYSIS_CONFIG["weights"]["pedagogy_components"]["engagement_techniques"],
        thresholds=MetricThreshold(
            excellent=8.5,   # Multiple varied engagement strategies
            good=7.0,        # Good engagement techniques used
            average=5.5,     # Some engagement efforts
            poor=4.0         # Limited or poor engagement
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "communication_clarity": MetricConfig(
        name="Explanation Quality",
        description="Clarity and effectiveness of explanations and concepts",
        weight=ANALYSIS_CONFIG["weights"]["pedagogy_components"]["communication_clarity"],
        thresholds=MetricThreshold(
            excellent=8.5,   # Crystal clear, accessible explanations
            good=7.0,        # Clear explanations with good examples
            average=5.5,     # Adequate clarity, some confusion
            poor=4.0         # Confusing or unclear explanations
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "use_of_examples": MetricConfig(
        name="Examples & Analogies",
        description="Effective use of examples, analogies, and illustrations",
        weight=ANALYSIS_CONFIG["weights"]["pedagogy_components"]["use_of_examples"],
        thresholds=MetricThreshold(
            excellent=8.5,   # Excellent, relevant, varied examples
            good=7.0,        # Good examples that illuminate concepts
            average=5.5,     # Some examples provided
            poor=4.0         # Few, poor, or confusing examples
        ),
        unit="score (1-10)",
        evidence_required=True
    ),
    
    "knowledge_checking": MetricConfig(
        name="Comprehension Checks",
        description="Verification of student understanding throughout lecture",
        weight=ANALYSIS_CONFIG["weights"]["pedagogy_components"]["knowledge_checking"],
        thresholds=MetricThreshold(
            excellent=8.5,   # Regular, varied comprehension checks
            good=7.0,        # Some effective checking for understanding
            average=5.5,     # Limited comprehension checks
            poor=4.0         # No or ineffective comprehension verification
        ),
        unit="score (1-10)",
        evidence_required=True
    )
}

def calculate_metric_score(value: float, metric: MetricConfig, reverse_scale: bool = False) -> float:
    """
    Calculate score based on metric thresholds with improved granularity
    
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
            # Linear interpolation between excellent and good
            ratio = (value - thresholds.excellent) / (thresholds.good - thresholds.excellent)
            return 10.0 - (ratio * 1.5)  # 8.5 to 10.0 range
        elif value <= thresholds.average:
            ratio = (value - thresholds.good) / (thresholds.average - thresholds.good)
            return 8.5 - (ratio * 1.5)  # 7.0 to 8.5 range
        elif value <= thresholds.poor:
            ratio = (value - thresholds.average) / (thresholds.poor - thresholds.average)
            return 7.0 - (ratio * 2.0)  # 5.0 to 7.0 range
        else:
            # Beyond poor threshold
            excess = (value - thresholds.poor) / thresholds.poor
            return max(1.0, 5.0 - (excess * 4.0))  # 1.0 to 5.0 range
    else:
        # For metrics where higher is better
        if value >= thresholds.excellent:
            return 10.0
        elif value >= thresholds.good:
            ratio = (value - thresholds.good) / (thresholds.excellent - thresholds.good)
            return 8.5 + (ratio * 1.5)  # 8.5 to 10.0 range
        elif value >= thresholds.average:
            ratio = (value - thresholds.average) / (thresholds.good - thresholds.average)
            return 7.0 + (ratio * 1.5)  # 7.0 to 8.5 range
        elif value >= thresholds.poor:
            ratio = (value - thresholds.poor) / (thresholds.average - thresholds.poor)
            return 5.0 + (ratio * 2.0)  # 5.0 to 7.0 range
        else:
            # Below poor threshold
            if thresholds.poor > 0:
                ratio = value / thresholds.poor
                return max(1.0, ratio * 5.0)  # 1.0 to 5.0 range
            else:
                return 2.0

def get_metric_feedback(value: float, metric: MetricConfig, reverse_scale: bool = False) -> str:
    """Generate specific feedback based on metric performance"""
    score = calculate_metric_score(value, metric, reverse_scale)
    
    if score >= 9:
        return f"Excellent {metric.name.lower()} - maintain this high standard"
    elif score >= 8:
        return f"Strong {metric.name.lower()} with minor areas for refinement"
    elif score >= 7:
        return f"Good {metric.name.lower()} with room for improvement"
    elif score >= 6:
        return f"Average {metric.name.lower()} - focus on targeted improvements"
    elif score >= 4:
        return f"Below average {metric.name.lower()} - requires attention"
    else:
        return f"Significant improvement needed in {metric.name.lower()}"

def get_configurable_parameters() -> Dict[str, Any]:
    """Return all configurable parameters for frontend UI"""
    return {
        "category_weights": ANALYSIS_CONFIG["weights"],
        "thresholds": ANALYSIS_CONFIG["thresholds"],
        "sampling_config": ANALYSIS_CONFIG["sampling"],
        "filler_words": FILLER_WORDS,
        "metrics_config": {
            "speech": {name: {"weight": config.weight, "thresholds": config.thresholds.__dict__} 
                     for name, config in SPEECH_METRICS.items()},
            "visual": {name: {"weight": config.weight, "thresholds": config.thresholds.__dict__} 
                      for name, config in VISUAL_METRICS.items()},
            "pedagogy": {name: {"weight": config.weight, "thresholds": config.thresholds.__dict__} 
                        for name, config in PEDAGOGY_METRICS.items()}
        }
    }

def update_configuration(new_config: Dict[str, Any]) -> bool:
    """Update configuration with new values from frontend"""
    try:
        # Update weights
        if "category_weights" in new_config:
            ANALYSIS_CONFIG["weights"].update(new_config["category_weights"])
        
        # Update thresholds
        if "thresholds" in new_config:
            ANALYSIS_CONFIG["thresholds"].update(new_config["thresholds"])
        
        # Update sampling config
        if "sampling_config" in new_config:
            ANALYSIS_CONFIG["sampling"].update(new_config["sampling_config"])
        
        # Update filler words
        if "filler_words" in new_config:
            global FILLER_WORDS
            FILLER_WORDS = new_config["filler_words"]
        
        return True
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return False