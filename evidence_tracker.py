from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from datetime import timedelta

@dataclass
class Evidence:
    """Single piece of evidence with timestamp and details"""
    timestamp: float  # Time in seconds
    category: str     # e.g., "speech", "visual", "content"
    metric: str       # e.g., "filler_words", "eye_contact" 
    observation: str  # Human-readable description
    severity: str     # "positive", "neutral", "negative"
    confidence: float # 0.0 to 1.0
    raw_data: Dict[str, Any] = None  # Additional data

class EvidenceTracker:
    """Tracks evidence throughout video analysis"""
    
    def __init__(self):
        self.evidence: List[Evidence] = []
        self.filler_word_timestamps: List[float] = []
        self.visual_analysis_timestamps: List[float] = []
        
    def add_evidence(self, evidence: Evidence):
        """Add a piece of evidence"""
        self.evidence.append(evidence)
        
    def add_speech_evidence(self, transcript_words: List[Dict], filler_words: List[str]):
        """Extract evidence from speech transcript with timestamps"""
        if not transcript_words:
            return
            
        # Track filler words with timestamps
        for word_data in transcript_words:
            word = word_data.get('word', '').lower().strip('.,!?')
            start_time = word_data.get('start', 0)
            
            if word in filler_words:
                self.filler_word_timestamps.append(start_time)
                
                evidence = Evidence(
                    timestamp=start_time,
                    category="speech",
                    metric="filler_words",
                    observation=f"Filler word '{word}' at {self.format_timestamp(start_time)}",
                    severity="negative",
                    confidence=0.9,
                    raw_data={"pause_duration": pause_duration, "context": f"After '{current_word.get('word', '')}' before '{next_word.get('word', '')}'"}
                )
                self.add_evidence(evidence)
            elif 1.0 <= pause_duration <= 2.0:  # Strategic pause
                evidence = Evidence(
                    timestamp=current_end,
                    category="speech",
                    metric="pauses",
                    observation=f"Strategic pause ({pause_duration:.1f}s) at {self.format_timestamp(current_end)}",
                    severity="positive",
                    confidence=0.7,
                    raw_data={"pause_duration": pause_duration, "context": f"After '{current_word.get('word', '')}' before '{next_word.get('word', '')}'"}
                )
                self.add_evidence(evidence)
    
    def add_visual_evidence(self, frame_time: float, analysis_result: Dict[str, Any]):
        """Add evidence from visual analysis of a frame"""
        scores = analysis_result.get('scores', {})
        observations = analysis_result.get('observations', [])
        
        # Track visual analysis timestamps
        self.visual_analysis_timestamps.append(frame_time)
        
        # Create evidence for each visual metric
        for metric, score in scores.items():
            severity = "positive" if score >= 8 else "negative" if score < 6 else "neutral"
            
            evidence = Evidence(
                timestamp=frame_time,
                category="visual",
                metric=metric,
                observation=f"{metric.replace('_', ' ').title()}: {score:.1f}/10 at {self.format_timestamp(frame_time)}",
                severity=severity,
                confidence=0.8,
                raw_data={"score": score, "detailed_observations": observations}
            )
            self.add_evidence(evidence)
        
        # Add specific observations
        for obs in observations:
            if isinstance(obs, str):
                # Determine severity based on keywords
                severity = self.classify_observation_severity(obs)
                
                evidence = Evidence(
                    timestamp=frame_time,
                    category="visual",
                    metric="general_observation",
                    observation=f"At {self.format_timestamp(frame_time)}: {obs}",
                    severity=severity,
                    confidence=0.7,
                    raw_data={"observation": obs}
                )
                self.add_evidence(evidence)
    
    def classify_observation_severity(self, observation: str) -> str:
        """Classify observation as positive, negative, or neutral based on keywords"""
        positive_keywords = ['excellent', 'good', 'confident', 'engaging', 'clear', 'professional', 'effective']
        negative_keywords = ['poor', 'weak', 'unclear', 'distracting', 'awkward', 'nervous', 'monotone']
        
        obs_lower = observation.lower()
        
        if any(keyword in obs_lower for keyword in positive_keywords):
            return "positive"
        elif any(keyword in obs_lower for keyword in negative_keywords):
            return "negative"
        else:
            return "neutral"
    
    def get_word_context(self, transcript_words: List[Dict], target_word: Dict, context_size: int = 3) -> str:
        """Get context around a specific word"""
        try:
            target_index = transcript_words.index(target_word)
            start_idx = max(0, target_index - context_size)
            end_idx = min(len(transcript_words), target_index + context_size + 1)
            
            context_words = transcript_words[start_idx:end_idx]
            return " ".join([w.get('word', '') for w in context_words])
        except (ValueError, KeyError):
            return ""
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    def get_evidence_by_category(self, category: str) -> List[Evidence]:
        """Get all evidence for a specific category"""
        return [e for e in self.evidence if e.category == category]
    
    def get_evidence_by_metric(self, metric: str) -> List[Evidence]:
        """Get all evidence for a specific metric"""
        return [e for e in self.evidence if e.metric == metric]
    
    def get_evidence_by_severity(self, severity: str) -> List[Evidence]:
        """Get all evidence of a specific severity"""
        return [e for e in self.evidence if e.severity == severity]
    
    def get_evidence_summary(self) -> Dict[str, Any]:
        """Generate a summary of all evidence"""
        total_evidence = len(self.evidence)
        
        summary = {
            "total_evidence_points": total_evidence,
            "by_category": {},
            "by_severity": {},
            "key_timestamps": {
                "filler_words": self.filler_word_timestamps[:10],  # Top 10
                "visual_analysis": self.visual_analysis_timestamps
            },
            "timeline": self.get_evidence_timeline()
        }
        
        # Count by category
        for category in ["speech", "visual", "content"]:
            summary["by_category"][category] = len(self.get_evidence_by_category(category))
        
        # Count by severity
        for severity in ["positive", "negative", "neutral"]:
            summary["by_severity"][severity] = len(self.get_evidence_by_severity(severity))
        
        return summary
    
    def get_evidence_timeline(self) -> List[Dict[str, Any]]:
        """Get chronological timeline of evidence"""
        sorted_evidence = sorted(self.evidence, key=lambda e: e.timestamp)
        
        timeline = []
        for evidence in sorted_evidence:
            timeline.append({
                "timestamp": evidence.timestamp,
                "formatted_time": self.format_timestamp(evidence.timestamp),
                "category": evidence.category,
                "metric": evidence.metric,
                "observation": evidence.observation,
                "severity": evidence.severity,
                "confidence": evidence.confidence
            })
        
        return timeline
    
    def get_detailed_feedback_with_evidence(self) -> Dict[str, List[str]]:
        """Generate detailed feedback organized by metric with specific evidence"""
        feedback = {
            "speech_strengths": [],
            "speech_improvements": [], 
            "visual_strengths": [],
            "visual_improvements": [],
            "general_observations": []
        }
        
        # Speech feedback with evidence
        speech_evidence = self.get_evidence_by_category("speech")
        positive_speech = [e for e in speech_evidence if e.severity == "positive"]
        negative_speech = [e for e in speech_evidence if e.severity == "negative"]
        
        # Group positive speech evidence
        for evidence in positive_speech:
            feedback["speech_strengths"].append(
                f"{evidence.observation} (confidence: {evidence.confidence:.0%})"
            )
        
        # Group negative speech evidence with improvement suggestions
        filler_evidence = [e for e in negative_speech if e.metric == "filler_words"]
        if filler_evidence:
            timestamps = [self.format_timestamp(e.timestamp) for e in filler_evidence[:5]]
            feedback["speech_improvements"].append(
                f"Reduce filler words - noticed at: {', '.join(timestamps)}"
            )
        
        rate_evidence = [e for e in negative_speech if e.metric == "speaking_rate"]
        if rate_evidence:
            timestamps = [self.format_timestamp(e.timestamp) for e in rate_evidence[:3]]
            feedback["speech_improvements"].append(
                f"Adjust speaking pace - check segments at: {', '.join(timestamps)}"
            )
        
        # Visual feedback with evidence
        visual_evidence = self.get_evidence_by_category("visual")
        positive_visual = [e for e in visual_evidence if e.severity == "positive"]
        negative_visual = [e for e in visual_evidence if e.severity == "negative"]
        
        for evidence in positive_visual:
            if evidence.metric != "general_observation":
                feedback["visual_strengths"].append(evidence.observation)
        
        for evidence in negative_visual:
            if evidence.metric != "general_observation":
                feedback["visual_improvements"].append(evidence.observation)
        
        return feedback
    
    def export_evidence_report(self) -> Dict[str, Any]:
        """Export comprehensive evidence report"""
        return {
            "summary": self.get_evidence_summary(),
            "detailed_timeline": self.get_evidence_timeline(),
            "feedback_with_evidence": self.get_detailed_feedback_with_evidence(),
            "metric_specific_evidence": {
                "filler_words": [e.__dict__ for e in self.get_evidence_by_metric("filler_words")],
                "speaking_rate": [e.__dict__ for e in self.get_evidence_by_metric("speaking_rate")],
                "eye_contact": [e.__dict__ for e in self.get_evidence_by_metric("eye_contact")],
                "gestures": [e.__dict__ for e in self.get_evidence_by_metric("gesture_effectiveness")]
            }
        }0.9,
                    raw_data={"word": word, "context": self.get_word_context(transcript_words, word_data)}
                )
                self.add_evidence(evidence)
        
        # Analyze speaking rate variations
        self.analyze_speaking_rate_evidence(transcript_words)
        
        # Detect long pauses
        self.detect_pause_evidence(transcript_words)
    
    def analyze_speaking_rate_evidence(self, transcript_words: List[Dict]):
        """Analyze speaking rate variations and create evidence"""
        if len(transcript_words) < 10:
            return
            
        # Calculate speaking rate in 30-second windows
        window_size = 30  # seconds
        words_per_window = []
        
        current_window_start = 0
        current_window_words = 0
        
        for word_data in transcript_words:
            word_time = word_data.get('start', 0)
            
            if word_time - current_window_start >= window_size:
                if current_window_words > 0:
                    rate = (current_window_words / window_size) * 60  # Convert to WPM
                    words_per_window.append({
                        'start_time': current_window_start,
                        'end_time': word_time,
                        'rate': rate
                    })
                    
                    # Create evidence for unusual speaking rates
                    if rate > 200:  # Too fast
                        evidence = Evidence(
                            timestamp=current_window_start,
                            category="speech",
                            metric="speaking_rate",
                            observation=f"Speaking very fast ({rate:.0f} WPM) from {self.format_timestamp(current_window_start)} to {self.format_timestamp(word_time)}",
                            severity="negative",
                            confidence=0.8,
                            raw_data={"rate": rate, "window_start": current_window_start, "window_end": word_time}
                        )
                        self.add_evidence(evidence)
                    elif rate < 100:  # Too slow
                        evidence = Evidence(
                            timestamp=current_window_start,
                            category="speech",
                            metric="speaking_rate", 
                            observation=f"Speaking slowly ({rate:.0f} WPM) from {self.format_timestamp(current_window_start)} to {self.format_timestamp(word_time)}",
                            severity="negative",
                            confidence=0.8,
                            raw_data={"rate": rate, "window_start": current_window_start, "window_end": word_time}
                        )
                        self.add_evidence(evidence)
                    elif 140 <= rate <= 180:  # Optimal rate
                        evidence = Evidence(
                            timestamp=current_window_start,
                            category="speech",
                            metric="speaking_rate",
                            observation=f"Excellent speaking pace ({rate:.0f} WPM) from {self.format_timestamp(current_window_start)} to {self.format_timestamp(word_time)}",
                            severity="positive",
                            confidence=0.8,
                            raw_data={"rate": rate, "window_start": current_window_start, "window_end": word_time}
                        )
                        self.add_evidence(evidence)
                
                current_window_start = word_time
                current_window_words = 0
            else:
                current_window_words += 1
    
    def detect_pause_evidence(self, transcript_words: List[Dict]):
        """Detect and analyze pauses between words"""
        for i in range(len(transcript_words) - 1):
            current_word = transcript_words[i]
            next_word = transcript_words[i + 1]
            
            current_end = current_word.get('end', 0)
            next_start = next_word.get('start', 0)
            pause_duration = next_start - current_end
            
            if pause_duration > 3.0:  # Pause longer than 3 seconds
                evidence = Evidence(
                    timestamp=current_end,
                    category="speech",
                    metric="pauses",
                    observation=f"Long pause ({pause_duration:.1f}s) at {self.format_timestamp(current_end)}",
                    severity="negative" if pause_duration > 5 else "neutral",
                    confidence=