"""
accessibility.py — Accessibility Features for Visually Impaired Users

Features:
  - Screen reader compatibility
  - Audio damage descriptions
  - Voice navigation
  - Haptic feedback patterns
  - High contrast mode
  - Large text mode
  - Keyboard shortcuts
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Try to import TTS libraries (graceful fallback if not available)
try:
    from gtts import gTTS
    import os
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    log.warning("gTTS not available - audio features disabled")

try:
    import pyttsx3
    PYTTSX_AVAILABLE = True
except ImportError:
    PYTTSX_AVAILABLE = False
    log.warning("pyttsx3 not available - offline TTS disabled")


@dataclass
class AudioDescription:
    """Audio description of damage"""
    text: str
    language: str
    audio_file: Optional[str] = None
    duration_sec: float = 0.0


class AccessibilityMode:
    """
    Accessibility features for visually impaired users
    """
    
    def __init__(self, default_language: str = 'en'):
        self.default_language = default_language
        self.tts_engine = None
        
        # Initialize offline TTS if available
        if PYTTSX_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)  # Speed
                self.tts_engine.setProperty('volume', 0.9)  # Volume
            except Exception as e:
                log.warning(f"Could not initialize TTS engine: {e}")
        
        log.info("Accessibility mode initialized")
    
    def audio_damage_description(self, detection: Dict, 
                                 language: str = 'en',
                                 detailed: bool = True) -> AudioDescription:
        """
        Convert detection to audio description
        
        Args:
            detection: Detection dictionary
            language: 'en', 'hi', or 'ta'
            detailed: Include detailed measurements
        
        Returns:
            AudioDescription with text and optional audio file
        """
        # Build description text
        if language == 'en':
            desc = self._build_english_description(detection, detailed)
        elif language == 'hi':
            desc = self._build_hindi_description(detection, detailed)
        elif language == 'ta':
            desc = self._build_tamil_description(detection, detailed)
        else:
            desc = self._build_english_description(detection, detailed)
        
        # Generate audio file if TTS available
        audio_file = None
        if TTS_AVAILABLE:
            try:
                audio_file = self._text_to_speech(desc, language)
            except Exception as e:
                log.warning(f"TTS generation failed: {e}")
        
        return AudioDescription(
            text=desc,
            language=language,
            audio_file=audio_file,
            duration_sec=len(desc.split()) * 0.4  # Rough estimate
        )
    
    def _build_english_description(self, detection: Dict, detailed: bool) -> str:
        """Build English description"""
        class_name = detection.get('class_name', 'damage')
        severity = detection.get('severity_class', 'moderate')
        distance_m = detection.get('distance_m', 0)
        position = detection.get('lane_position', 'center')
        
        # Basic description
        desc = f"{class_name} detected ahead"
        
        if distance_m > 0:
            desc += f" at {int(distance_m)} meters"
        
        desc += ". "
        
        # Severity
        desc += f"Severity: {severity}. "
        
        if detailed:
            # Detailed measurements
            if detection.get('depth_cm'):
                desc += f"Depth: {detection['depth_cm']:.1f} centimeters. "
            
            if detection.get('width_cm'):
                desc += f"Width: {detection['width_cm']:.0f} centimeters. "
            
            # Position
            desc += f"Located on {position} side of road. "
        
        # Safety warning
        if detection.get('severity', 0) > 70:
            desc += "Caution: High risk of vehicle damage. Reduce speed."
        elif detection.get('severity', 0) > 50:
            desc += "Caution advised. Slow down if possible."
        
        return desc
    
    def _build_hindi_description(self, detection: Dict, detailed: bool) -> str:
        """Build Hindi description"""
        class_map = {
            'Pothole': 'गड्ढा',
            'Alligator Crack': 'दरार',
            'Transverse Crack': 'अनुप्रस्थ दरार',
            'Longitudinal Crack': 'अनुदैर्ध्य दरार'
        }
        
        severity_map = {
            'low': 'कम',
            'moderate': 'मध्यम',
            'high': 'उच्च',
            'severe': 'गंभीर'
        }
        
        class_name = class_map.get(detection.get('class_name', ''), 'क्षति')
        severity = severity_map.get(detection.get('severity_class', 'moderate'), 'मध्यम')
        
        desc = f"आगे {class_name} का पता चला है. "
        desc += f"गंभीरता: {severity}. "
        
        if detection.get('depth_cm') and detailed:
            desc += f"गहराई: {detection['depth_cm']:.1f} सेंटीमीटर. "
        
        if detection.get('severity', 0) > 70:
            desc += "सावधान: वाहन को नुकसान का खतरा. गति कम करें."
        
        return desc
    
    def _build_tamil_description(self, detection: Dict, detailed: bool) -> str:
        """Build Tamil description"""
        class_map = {
            'Pothole': 'குழி',
            'Alligator Crack': 'விரிசல்',
            'Transverse Crack': 'குறுக்கு விரிசல்',
            'Longitudinal Crack': 'நீளமான விரிசல்'
        }
        
        severity_map = {
            'low': 'குறைவு',
            'moderate': 'மிதமான',
            'high': 'உயர்',
            'severe': 'கடுமையான'
        }
        
        class_name = class_map.get(detection.get('class_name', ''), 'சேதம்')
        severity = severity_map.get(detection.get('severity_class', 'moderate'), 'மிதமான')
        
        desc = f"முன்னால் {class_name} கண்டறியப்பட்டது. "
        desc += f"தீவிரம்: {severity}. "
        
        if detection.get('depth_cm') and detailed:
            desc += f"ஆழம்: {detection['depth_cm']:.1f} சென்டிமீட்டர். "
        
        if detection.get('severity', 0) > 70:
            desc += "எச்சரிக்கை: வாகன சேதத்தின் அபாயம். வேகத்தை குறைக்கவும்."
        
        return desc
    
    def _text_to_speech(self, text: str, language: str) -> str:
        """
        Convert text to speech audio file
        
        Returns path to generated audio file
        """
        lang_codes = {
            'en': 'en',
            'hi': 'hi',
            'ta': 'ta'
        }
        
        import tempfile
        import uuid
        
        # Generate unique filename
        audio_filename = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = tempfile.gettempdir() + "/" + audio_filename
        
        # Generate audio
        tts = gTTS(text=text, lang=lang_codes.get(language, 'en'), slow=False)
        tts.save(audio_path)
        
        log.info(f"Generated audio: {audio_path}")
        
        return audio_path
    
    def speak_text(self, text: str):
        """
        Speak text using offline TTS (non-blocking)
        
        Useful for real-time navigation
        """
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                log.warning(f"Speech failed: {e}")
        else:
            log.debug(f"Would speak: {text}")
    
    def haptic_feedback_pattern(self, severity: str) -> List[int]:
        """
        Generate haptic feedback vibration pattern for mobile
        
        Args:
            severity: 'low', 'moderate', 'high', 'severe'
        
        Returns:
            List of vibration durations in milliseconds
            [vibrate_ms, pause_ms, vibrate_ms, ...]
        """
        patterns = {
            'low': [100, 50, 100],                      # ▁▁▁
            'moderate': [150, 50, 150, 50, 150],       # ▁▁▁ ▁▁▁
            'high': [200, 50, 200, 50, 200, 50, 200],  # ▁▁▁ ▁▁▁ ▁▁▁
            'severe': [300, 100, 300, 100, 300]        # ▁▁▁▁▁▁▁
        }
        
        return patterns.get(severity, patterns['moderate'])
    
    def get_keyboard_shortcuts(self) -> Dict[str, str]:
        """
        Get keyboard shortcuts for accessibility
        
        Returns dict of key → action mappings
        """
        return {
            'Alt+D': 'Toggle dashboard view',
            'Alt+M': 'Open map',
            'Alt+R': 'View reports',
            'Alt+S': 'Search',
            'Alt+H': 'Help',
            'Alt+1': 'Go to frame 1',
            'Alt+N': 'Next frame',
            'Alt+P': 'Previous frame',
            'Alt+Space': 'Play/Pause',
            'Alt+A': 'Read current frame aloud',
            'Ctrl++': 'Increase text size',
            'Ctrl+-': 'Decrease text size',
            'Ctrl+H': 'Toggle high contrast',
            'Esc': 'Close modal/Cancel',
            '/': 'Focus search box'
        }
    
    def get_aria_labels(self) -> Dict[str, str]:
        """
        Get ARIA labels for screen readers
        
        Returns dict of element_id → aria-label
        """
        return {
            'health_score': 'Road health score out of 100',
            'total_cost': 'Estimated repair cost in rupees',
            'detection_count': 'Total number of defects detected',
            'map_view': 'Interactive map showing damage locations',
            'timeline': 'Frame timeline scrubber',
            'damage_feed': 'List of detected damage',
            'analysis_panel': 'Detailed analysis and statistics',
            'export_button': 'Export report button',
            'voice_control': 'Voice command button',
            'leaderboard': 'Citizen reporter leaderboard'
        }
    
    def generate_alt_text(self, detection: Dict) -> str:
        """
        Generate alt text for damage image
        
        For screen readers to describe detection images
        """
        class_name = detection.get('class_name', 'damage')
        confidence = detection.get('confidence', 0) * 100
        severity = detection.get('severity_class', 'moderate')
        
        alt_text = f"{class_name} with {confidence:.0f}% confidence, "
        alt_text += f"{severity} severity"
        
        if detection.get('depth_cm'):
            alt_text += f", {detection['depth_cm']:.1f} centimeters deep"
        
        if detection.get('width_cm'):
            alt_text += f", {detection['width_cm']:.0f} centimeters wide"
        
        return alt_text
    
    def get_high_contrast_theme(self) -> Dict[str, str]:
        """
        Get high contrast color scheme for visually impaired
        
        Returns CSS color variables
        """
        return {
            '--bg-primary': '#000000',
            '--bg-secondary': '#1a1a1a',
            '--text-primary': '#ffffff',
            '--text-secondary': '#ffff00',
            '--accent': '#00ff00',
            '--danger': '#ff0000',
            '--warning': '#ffff00',
            '--success': '#00ff00',
            '--border': '#ffffff',
            '--link': '#00ffff',
            '--link-hover': '#ffff00'
        }
    
    def get_large_text_css(self) -> str:
        """
        Get CSS for large text mode
        
        Returns CSS string to inject
        """
        return '''
        body {
            font-size: 18px !important;
            line-height: 1.8 !important;
        }
        h1 { font-size: 3rem !important; }
        h2 { font-size: 2.5rem !important; }
        h3 { font-size: 2rem !important; }
        button, .btn {
            min-height: 48px !important;
            min-width: 48px !important;
            font-size: 1.2rem !important;
        }
        input, select, textarea {
            font-size: 1.2rem !important;
            padding: 12px !important;
        }
        '''