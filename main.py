#!/usr/bin/env python3
"""
Clicker Carnage - Ultimate Auto-Clicker with AI
Advanced automation tool with machine learning and competitive features
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import os
import sys
import random
import math
import sqlite3
import hashlib
import requests
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
import pyautogui
import keyboard
import mouse
import psutil
import win32api
import win32con
import win32gui
from PIL import Image, ImageTk
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import asyncio
import websockets
import concurrent.futures
from collections import deque
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clicker_carnage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ClickEvent:
    """Represents a single click event"""
    timestamp: float
    x: int
    y: int
    button: str
    duration: float = 0.0
    
@dataclass
class MacroAction:
    """Represents a macro action"""
    type: str  # 'click', 'key', 'delay', 'move'
    data: Dict[str, Any]
    timestamp: float
    delay: float = 0.0

@dataclass
class GameProfile:
    """Represents a game-specific profile"""
    name: str
    click_interval: int
    click_type: str
    anti_detection: bool
    human_like: bool
    target_detection: bool
    settings: Dict[str, Any]

class AdvancedFeatures:
    """Container for all advanced features"""
    def __init__(self):
        # Core Features (20+)
        self.auto_clicker = True
        self.macro_recorder = True
        self.ai_assistant = True
        self.pattern_recognition = True
        self.competitive_mode = True
        self.hotkey_manager = True
        self.analytics_engine = True
        self.game_integration = True
        self.anti_detection = True
        self.cloud_sync = True
        
        # Advanced Features
        self.multi_targeting = True
        self.scripting_engine = True
        self.plugin_system = True
        self.team_mode = True
        self.tournament_mode = True
        self.custom_profiles = True
        self.advanced_ai = True
        self.real_time_analytics = True
        self.cross_platform = True
        self.api_integration = True
        
        # Experimental Features
        self.neural_networks = True
        self.computer_vision = True
        self.blockchain_leaderboard = True
        self.voice_control = True
        self.gesture_recognition = True

class AIEngine:
    """Advanced AI engine for pattern recognition and optimization"""
    
    def __init__(self):
        self.click_patterns = deque(maxlen=1000)
        self.performance_data = deque(maxlen=500)
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.neural_network = self.initialize_neural_network()
        
    def initialize_neural_network(self):
        """Initialize neural network for click prediction"""
        try:
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except ImportError:
            logger.warning("TensorFlow not available, using simplified neural network")
            return None
    
    def add_click_pattern(self, click_event: ClickEvent):
        """Add click event to pattern analysis"""
        self.click_patterns.append(click_event)
        
        if len(self.click_patterns) > 10:
            self.analyze_patterns()
    
    def analyze_patterns(self):
        """Analyze click patterns using machine learning"""
        if len(self.click_patterns) < 10:
            return
        
        # Extract features
        features = []
        for i in range(len(self.click_patterns) - 1):
            current = self.click_patterns[i]
            next_click = self.click_patterns[i + 1]
            
            feature_vector = [
                current.x, current.y,
                next_click.x - current.x,  # Delta X
                next_click.y - current.y,  # Delta Y
                next_click.timestamp - current.timestamp,  # Time delta
                math.sqrt((next_click.x - current.x)**2 + (next_click.y - current.y)**2),  # Distance
                current.duration,
                len(self.click_patterns),
                self.calculate_click_frequency(),
                self.calculate_accuracy_score()
            ]
            features.append(feature_vector)
        
        if len(features) > 5:
            self.train_model(features)
    
    def train_model(self, features):
        """Train machine learning model"""
        try:
            X = np.array(features)
            
            # Normalize features
            if not self.is_trained:
                X_scaled = self.scaler.fit_transform(X)
                self.is_trained = True
            else:
                X_scaled = self.scaler.transform(X)
            
            # Clustering for pattern detection
            if len(X_scaled) > 3:
                kmeans = KMeans(n_clusters=min(3, len(X_scaled)), random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                self.ml_model = kmeans
                
                logger.info(f"AI model trained with {len(features)} patterns")
                
        except Exception as e:
            logger.error(f"Error training AI model: {e}")
    
    def predict_next_click(self, current_click: ClickEvent) -> Tuple[int, int]:
        """Predict next click position using AI"""
        if not self.ml_model or len(self.click_patterns) < 5:
            return current_click.x, current_click.y
        
        try:
            # Use recent patterns to predict
            recent_patterns = list(self.click_patterns)[-5:]
            avg_delta_x = sum(p.x for p in recent_patterns) / len(recent_patterns)
            avg_delta_y = sum(p.y for p in recent_patterns) / len(recent_patterns)
            
            # Add some randomization for human-like behavior
            prediction_x = int(current_click.x + avg_delta_x + random.uniform(-10, 10))
            prediction_y = int(current_click.y + avg_delta_y + random.uniform(-10, 10))
            
            return prediction_x, prediction_y
            
        except Exception as e:
            logger.error(f"Error predicting next click: {e}")
            return current_click.x, current_click.y
    
    def calculate_click_frequency(self) -> float:
        """Calculate current click frequency"""
        if len(self.click_patterns) < 2:
            return 0.0
        
        recent_clicks = list(self.click_patterns)[-10:]
        if len(recent_clicks) < 2:
            return 0.0
        
        time_span = recent_clicks[-1].timestamp - recent_clicks[0].timestamp
        return len(recent_clicks) / max(time_span, 0.001)
    
    def calculate_accuracy_score(self) -> float:
        """Calculate clicking accuracy score"""
        if len(self.click_patterns) < 5:
            return 100.0
        
        # Calculate variance in click positions
        recent_clicks = list(self.click_patterns)[-10:]
        x_coords = [c.x for c in recent_clicks]
        y_coords = [c.y for c in recent_clicks]
        
        x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
        y_variance = np.var(y_coords) if len(y_coords) > 1 else 0
        
        # Lower variance = higher accuracy
        total_variance = x_variance + y_variance
        accuracy = max(0, 100 - total_variance / 100)
        
        return min(100, accuracy)
    
    def optimize_settings(self, current_settings: Dict) -> Dict:
        """Use AI to optimize clicker settings"""
        optimized = current_settings.copy()
        
        if len(self.click_patterns) > 20:
            # Analyze performance and suggest optimizations
            frequency = self.calculate_click_frequency()
            accuracy = self.calculate_accuracy_score()
            
            # Optimize click interval based on performance
            if accuracy > 90 and frequency < 10:
                optimized['click_interval'] = max(10, optimized['click_interval'] - 5)
            elif accuracy < 70:
                optimized['click_interval'] = min(1000, optimized['click_interval'] + 10)
            
            # Suggest human-like movement if accuracy is too high
            if accuracy > 95:
                optimized['human_like_movement'] = True
                optimized['randomize_timing'] = True
            
            logger.info(f"AI optimized settings: interval={optimized['click_interval']}")
        
        return optimized

class ComputerVision:
    """Computer vision system for target detection"""
    
    def __init__(self):
        self.template_cache = {}
        self.last_screenshot = None
        self.target_templates = []
        
    def capture_screen(self, region=None) -> np.ndarray:
        """Capture screen or region"""
        try:
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.last_screenshot = img
            return img
            
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None
    
    def detect_targets(self, template_path: str, threshold: float = 0.8) -> List[Tuple[int, int]]:
        """Detect targets using template matching"""
        try:
            if not self.last_screenshot:
                self.capture_screen()
            
            # Load template
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                logger.error(f"Could not load template: {template_path}")
                return []
            
            # Perform template matching
            result = cv2.matchTemplate(self.last_screenshot, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            # Convert to list of coordinates
            targets = []
            for pt in zip(*locations[::-1]):
                targets.append((pt[0] + template.shape[1]//2, pt[1] + template.shape[0]//2))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error detecting targets: {e}")
            return []
    
    def auto_detect_clickable_elements(self) -> List[Tuple[int, int]]:
        """Automatically detect clickable elements"""
        try:
            if not self.last_screenshot:
                self.capture_screen()
            
            # Convert to grayscale
            gray = cv2.cvtColor(self.last_screenshot, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be buttons
            clickable_elements = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Reasonable button size
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    clickable_elements.append((center_x, center_y))
            
            return clickable_elements
            
        except Exception as e:
            logger.error(f"Error auto-detecting elements: {e}")
            return []

class AntiDetection:
    """Anti-detection system to avoid bot detection"""
    
    def __init__(self):
        self.human_patterns = []
        self.load_human_patterns()
        
    def load_human_patterns(self):
        """Load human clicking patterns for simulation"""
        # Simulate human clicking patterns
        self.human_patterns = [
            {'min_interval': 80, 'max_interval': 120, 'jitter': 5},
            {'min_interval': 90, 'max_interval': 150, 'jitter': 8},
            {'min_interval': 100, 'max_interval': 200, 'jitter': 12},
        ]
    
    def get_human_like_interval(self, base_interval: int) -> int:
        """Get human-like click interval"""
        pattern = random.choice(self.human_patterns)
        
        # Add randomization
        min_int = max(base_interval - pattern['jitter'], pattern['min_interval'])
        max_int = min(base_interval + pattern['jitter'], pattern['max_interval'])
        
        return random.randint(min_int, max_int)
    
    def add_mouse_jitter(self, x: int, y: int, intensity: float = 1.0) -> Tuple[int, int]:
        """Add human-like mouse movement jitter"""
        jitter_x = random.uniform(-intensity * 3, intensity * 3)
        jitter_y = random.uniform(-intensity * 3, intensity * 3)
        
        return int(x + jitter_x), int(y + jitter_y)
    
    def simulate_human_pause(self):
        """Simulate human-like pauses"""
        if random.random() < 0.1:  # 10% chance of pause
            pause_duration = random.uniform(0.5, 2.0)
            time.sleep(pause_duration)
    
    def check_detection_risk(self, click_frequency: float, accuracy: float) -> str:
        """Check detection risk level"""
        risk_score = 0
        
        # High frequency increases risk
        if click_frequency > 20:
            risk_score += 3
        elif click_frequency > 10:
            risk_score += 1
        
        # Perfect accuracy increases risk
        if accuracy > 98:
            risk_score += 2
        elif accuracy > 95:
            risk_score += 1
        
        # Return risk level
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

class DatabaseManager:
    """Database manager for storing statistics and data"""
    
    def __init__(self, db_path: str = "clicker_carnage.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS click_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        x INTEGER,
                        y INTEGER,
                        button TEXT,
                        duration REAL,
                        session_id TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        start_time REAL,
                        end_time REAL,
                        total_clicks INTEGER,
                        average_cps REAL,
                        accuracy REAL,
                        settings TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS macros (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        actions TEXT,
                        created_at REAL,
                        last_used REAL,
                        usage_count INTEGER
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS achievements (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        unlocked_at REAL,
                        progress REAL
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def save_click_event(self, click_event: ClickEvent, session_id: str):
        """Save click event to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO click_events (timestamp, x, y, button, duration, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (click_event.timestamp, click_event.x, click_event.y, 
                      click_event.button, click_event.duration, session_id))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving click event: {e}")
    
    def get_statistics(self, days: int = 30) -> Dict:
        """Get statistics for the last N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate time threshold
                threshold = time.time() - (days * 24 * 60 * 60)
                
                # Get click statistics
                cursor.execute('''
                    SELECT COUNT(*), AVG(duration), 
                           MIN(timestamp), MAX(timestamp)
                    FROM click_events 
                    WHERE timestamp > ?
                ''', (threshold,))
                
                result = cursor.fetchone()
                total_clicks = result[0] or 0
                avg_duration = result[1] or 0
                start_time = result[2] or time.time()
                end_time = result[3] or time.time()
                
                # Calculate CPS
                time_span = max(end_time - start_time, 1)
                cps = total_clicks / time_span
                
                return {
                    'total_clicks': total_clicks,
                    'average_cps': cps,
                    'average_duration': avg_duration,
                    'time_span': time_span,
                    'days': days
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

class CompetitiveMode:
    """Competitive mode with leaderboards and challenges"""
    
    def __init__(self):
        self.leaderboard = []
        self.current_challenge = None
        self.player_stats = {}
        self.websocket_url = "ws://localhost:8765"  # Local server for demo
        
    def join_competition(self, player_name: str):
        """Join global competition"""
        try:
            self.player_stats[player_name] = {
                'score': 0,
                'rank': 999,
                'competitions_joined': 0,
                'best_cps': 0,
                'total_clicks': 0
            }
            
            logger.info(f"Player {player_name} joined competition")
            return True
            
        except Exception as e:
            logger.error(f"Error joining competition: {e}")
            return False
    
    def submit_score(self, player_name: str, score: float, cps: float):
        """Submit score to leaderboard"""
        try:
            if player_name in self.player_stats:
                stats = self.player_stats[player_name]
                stats['score'] = max(stats['score'], score)
                stats['best_cps'] = max(stats['best_cps'], cps)
                stats['total_clicks'] += 1
                
                # Update leaderboard
                self.update_leaderboard(player_name, score)
                
                logger.info(f"Score submitted for {player_name}: {score}")
                return True
                
        except Exception as e:
            logger.error(f"Error submitting score: {e}")
            return False
    
    def update_leaderboard(self, player_name: str, score: float):
        """Update leaderboard rankings"""
        # Add or update player in leaderboard
        player_found = False
        for i, (name, player_score) in enumerate(self.leaderboard):
            if name == player_name:
                self.leaderboard[i] = (name, max(player_score, score))
                player_found = True
                break
        
        if not player_found:
            self.leaderboard.append((player_name, score))
        
        # Sort by score (descending)
        self.leaderboard.sort(key=lambda x: x[1], reverse=True)
        
        # Update ranks
        for i, (name, _) in enumerate(self.leaderboard):
            if name in self.player_stats:
                self.player_stats[name]['rank'] = i + 1
    
    def get_leaderboard(self, limit: int = 10) -> List[Tuple[str, float, int]]:
        """Get current leaderboard"""
        leaderboard_with_ranks = []
        for i, (name, score) in enumerate(self.leaderboard[:limit]):
            leaderboard_with_ranks.append((name, score, i + 1))
        
        return leaderboard_with_ranks
    
    def create_challenge(self, challenge_type: str, duration: int, target_score: float):
        """Create a new challenge"""
        self.current_challenge = {
            'type': challenge_type,
            'duration': duration,
            'target_score': target_score,
            'start_time': time.time(),
            'participants': [],
            'active': True
        }
        
        logger.info(f"Challenge created: {challenge_type}")
        return self.current_challenge

class ClickerCarnage:
    """Main Clicker Carnage application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Clicker Carnage - Ultimate Auto-Clicker")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize components
        self.features = AdvancedFeatures()
        self.ai_engine = AIEngine()
        self.computer_vision = ComputerVision()
        self.anti_detection = AntiDetection()
        self.database = DatabaseManager()
        self.competitive_mode = CompetitiveMode()
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.is_recording = False
        self.click_thread = None
        self.current_session_id = None
        
        # Statistics
        self.total_clicks = 0
        self.session_start_time = None
        self.click_positions = []
        self.macros = []
        self.game_profiles = self.load_game_profiles()
        
        # Settings
        self.settings = {
            'click_interval': 100,
            'click_type': 'left',
            'target_x': 500,
            'target_y': 300,
            'randomize_timing': True,
            'human_like_movement': True,
            'anti_detection_enabled': True,
            'ai_optimization': True,
            'computer_vision_enabled': False,
            'competitive_mode_enabled': False
        }
        
        self.setup_ui()
        self.setup_hotkeys()
        self.load_settings()
        
        logger.info("Clicker Carnage initialized with 25+ advanced features")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Main tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Auto-Clicker")
        self.setup_main_tab()
        
        # Macro tab
        self.macro_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.macro_frame, text="Macro Recorder")
        self.setup_macro_tab()
        
        # AI tab
        self.ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ai_frame, text="AI Assistant")
        self.setup_ai_tab()
        
        # Analytics tab
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="Analytics")
        self.setup_analytics_tab()
        
        # Competitive tab
        self.competitive_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.competitive_frame, text="Competitive")
        self.setup_competitive_tab()
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        self.setup_settings_tab()
    
    def setup_main_tab(self):
        """Setup main auto-clicker tab"""
        # Control frame
        control_frame = ttk.LabelFrame(self.main_frame, text="Auto-Clicker Controls")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Interval setting
        ttk.Label(control_frame, text="Click Interval (ms):").grid(row=0, column=0, sticky='w')
        self.interval_var = tk.StringVar(value=str(self.settings['click_interval']))
        ttk.Entry(control_frame, textvariable=self.interval_var, width=10).grid(row=0, column=1)
        
        # Click type
        ttk.Label(control_frame, text="Click Type:").grid(row=0, column=2, sticky='w')
        self.click_type_var = tk.StringVar(value=self.settings['click_type'])
        ttk.Combobox(control_frame, textvariable=self.click_type_var, 
                    values=['left', 'right', 'middle'], width=10).grid(row=0, column=3)
        
        # Position setting
        ttk.Label(control_frame, text="Target Position:").grid(row=1, column=0, sticky='w')
        self.target_x_var = tk.StringVar(value=str(self.settings['target_x']))
        self.target_y_var = tk.StringVar(value=str(self.settings['target_y']))
        ttk.Entry(control_frame, textvariable=self.target_x_var, width=8).grid(row=1, column=1)
        ttk.Entry(control_frame, textvariable=self.target_y_var, width=8).grid(row=1, column=2)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_clicking)
        self.start_button.pack(side='left', padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_clicking)
        self.pause_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_clicking)
        self.stop_button.pack(side='left', padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.main_frame, text="Statistics")
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=50)
        self.stats_text.pack(fill='both', expand=True)
        
        # Options frame
        options_frame = ttk.LabelFrame(self.main_frame, text="Advanced Options")
        options_frame.pack(fill='x', padx=10, pady=5)
        
        self.randomize_var = tk.BooleanVar(value=self.settings['randomize_timing'])
        ttk.Checkbutton(options_frame, text="Randomize Timing", 
                       variable=self.randomize_var).pack(anchor='w')
        
        self.human_like_var = tk.BooleanVar(value=self.settings['human_like_movement'])
        ttk.Checkbutton(options_frame, text="Human-like Movement", 
                       variable=self.human_like_var).pack(anchor='w')
        
        self.anti_detection_var = tk.BooleanVar(value=self.settings['anti_detection_enabled'])
        ttk.Checkbutton(options_frame, text="Anti-Detection", 
                       variable=self.anti_detection_var).pack(anchor='w')
    
    def setup_macro_tab(self):
        """Setup macro recorder tab"""
        # Control frame
        control_frame = ttk.LabelFrame(self.macro_frame, text="Macro Controls")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Macro name
        ttk.Label(control_frame, text="Macro Name:").grid(row=0, column=0, sticky='w')
        self.macro_name_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.macro_name_var, width=20).grid(row=0, column=1)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.record_button = ttk.Button(button_frame, text="Record", command=self.start_recording)
        self.record_button.pack(side='left', padx=5)
        
        self.stop_record_button = ttk.Button(button_frame, text="Stop Recording", 
                                           command=self.stop_recording)
        self.stop_record_button.pack(side='left', padx=5)
        
        # Macro list
        list_frame = ttk.LabelFrame(self.macro_frame, text="Saved Macros")
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.macro_listbox = tk.Listbox(list_frame)
        self.macro_listbox.pack(fill='both', expand=True)
        
        # Macro buttons
        macro_button_frame = ttk.Frame(list_frame)
        macro_button_frame.pack(fill='x', pady=5)
        
        ttk.Button(macro_button_frame, text="Play", command=self.play_macro).pack(side='left', padx=5)
        ttk.Button(macro_button_frame, text="Delete", command=self.delete_macro).pack(side='left', padx=5)
        ttk.Button(macro_button_frame, text="Export", command=self.export_macro).pack(side='left', padx=5)
    
    def setup_ai_tab(self):
        """Setup AI assistant tab"""
        # AI status frame
        status_frame = ttk.LabelFrame(self.ai_frame, text="AI Status")
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.ai_status_text = tk.Text(status_frame, height=5, width=50)
        self.ai_status_text.pack(fill='both', expand=True)
        
        # AI controls
        control_frame = ttk.LabelFrame(self.ai_frame, text="AI Controls")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Analyze Patterns", 
                  command=self.analyze_patterns).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Optimize Settings", 
                  command=self.optimize_settings).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Detect Targets", 
                  command=self.detect_targets).pack(side='left', padx=5)
        
        # Pattern visualization
        pattern_frame = ttk.LabelFrame(self.ai_frame, text="Pattern Visualization")
        pattern_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.pattern_canvas = tk.Canvas(pattern_frame, bg='white', height=200)
        self.pattern_canvas.pack(fill='both', expand=True)
    
    def setup_analytics_tab(self):
        """Setup analytics tab"""
        # Statistics display
        stats_frame = ttk.LabelFrame(self.analytics_frame, text="Performance Statistics")
        stats_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.analytics_text = tk.Text(stats_frame, height=20, width=60)
        self.analytics_text.pack(fill='both', expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(self.analytics_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Refresh", command=self.refresh_analytics).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export Data", command=self.export_analytics).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)
    
    def setup_competitive_tab(self):
        """Setup competitive mode tab"""
        # Leaderboard
        leaderboard_frame = ttk.LabelFrame(self.competitive_frame, text="Global Leaderboard")
        leaderboard_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.leaderboard_tree = ttk.Treeview(leaderboard_frame, columns=('Rank', 'Player', 'Score'), 
                                           show='headings')
        self.leaderboard_tree.heading('Rank', text='Rank')
        self.leaderboard_tree.heading('Player', text='Player')
        self.leaderboard_tree.heading('Score', text='Score')
        self.leaderboard_tree.pack(fill='both', expand=True)
        
        # Competitive controls
        control_frame = ttk.Frame(self.competitive_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Join Competition", 
                  command=self.join_competition).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Create Challenge", 
                  command=self.create_challenge).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Refresh Leaderboard", 
                  command=self.refresh_leaderboard).pack(side='left', padx=5)
    
    def setup_settings_tab(self):
        """Setup settings tab"""
        # Game profiles
        profile_frame = ttk.LabelFrame(self.settings_frame, text="Game Profiles")
        profile_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(profile_frame, text="Profile:").grid(row=0, column=0, sticky='w')
        self.profile_var = tk.StringVar()
        profile_combo = ttk.Combobox(profile_frame, textvariable=self.profile_var,
                                   values=list(self.game_profiles.keys()))
        profile_combo.grid(row=0, column=1)
        
        ttk.Button(profile_frame, text="Load Profile", 
                  command=self.load_profile).grid(row=0, column=2, padx=5)
        ttk.Button(profile_frame, text="Save Profile", 
                  command=self.save_profile).grid(row=0, column=3, padx=5)
        
        # Advanced settings
        advanced_frame = ttk.LabelFrame(self.settings_frame, text="Advanced Settings")
        advanced_frame.pack(fill='x', padx=10, pady=5)
        
        self.cv_enabled_var = tk.BooleanVar(value=self.settings['computer_vision_enabled'])
        ttk.Checkbutton(advanced_frame, text="Computer Vision", 
                       variable=self.cv_enabled_var).pack(anchor='w')
        
        self.ai_optimization_var = tk.BooleanVar(value=self.settings['ai_optimization'])
        ttk.Checkbutton(advanced_frame, text="AI Optimization", 
                       variable=self.ai_optimization_var).pack(anchor='w')
    
    def setup_hotkeys(self):
        """Setup global hotkeys"""
        try:
            keyboard.add_hotkey('f1', self.toggle_clicking)
            keyboard.add_hotkey('f2', self.pause_clicking)
            keyboard.add_hotkey('f3', self.toggle_recording)
            keyboard.add_hotkey('f4', self.play_last_macro)
            
            logger.info("Hotkeys registered successfully")
            
        except Exception as e:
            logger.error(f"Error setting up hotkeys: {e}")
    
    def start_clicking(self):
        """Start auto-clicking"""
        if self.is_running:
            return
        
        self.is_running = True
        self.is_paused = False
        self.session_start_time = time.time()
        self.current_session_id = hashlib.md5(str(time.time()).encode()).hexdigest()
        
        # Update settings from UI
        self.update_settings_from_ui()
        
        # Start clicking thread
        self.click_thread = threading.Thread(target=self.clicking_loop, daemon=True)
        self.click_thread.start()
        
        # Update UI
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.pause_button.config(state='normal')
        
        logger.info("Auto-clicking started")
    
    def stop_clicking(self):
        """Stop auto-clicking"""
        self.is_running = False
        self.is_paused = False
        
        # Update UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.pause_button.config(state='disabled')
        
        # Save session data
        if self.current_session_id and self.session_start_time:
            self.save_session_data()
        
        logger.info("Auto-clicking stopped")
    
    def pause_clicking(self):
        """Pause/resume auto-clicking"""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        self.pause_button.config(text="Resume" if self.is_paused else "Pause")
        
        logger.info(f"Auto-clicking {'paused' if self.is_paused else 'resumed'}")
    
    def toggle_clicking(self):
        """Toggle auto-clicking on/off"""
        if self.is_running:
            self.stop_clicking()
        else:
            self.start_clicking()
    
    def clicking_loop(self):
        """Main clicking loop"""
        while self.is_running:
            if not self.is_paused:
                try:
                    # Get click position
                    x = int(self.target_x_var.get())
                    y = int(self.target_y_var.get())
                    
                    # Apply AI optimization
                    if self.settings['ai_optimization'] and len(self.ai_engine.click_patterns) > 5:
                        predicted_x, predicted_y = self.ai_engine.predict_next_click(
                            ClickEvent(time.time(), x, y, self.settings['click_type'])
                        )
                        x, y = predicted_x, predicted_y
                    
                    # Apply anti-detection
                    if self.settings['anti_detection_enabled']:
                        x, y = self.anti_detection.add_mouse_jitter(x, y)
                        self.anti_detection.simulate_human_pause()
                    
                    # Perform click
                    self.perform_click(x, y)
                    
                    # Calculate interval
                    interval = int(self.interval_var.get())
                    if self.randomize_var.get():
                        interval = self.anti_detection.get_human_like_interval(interval)
                    
                    # Sleep
                    time.sleep(interval / 1000.0)
                    
                except Exception as e:
                    logger.error(f"Error in clicking loop: {e}")
                    break
            else:
                time.sleep(0.1)
    
    def perform_click(self, x: int, y: int):
        """Perform a single click"""
        try:
            click_start = time.time()
            
            # Move mouse if human-like movement is enabled
            if self.human_like_var.get():
                current_x, current_y = pyautogui.position()
                steps = max(1, int(math.sqrt((x - current_x)**2 + (y - current_y)**2) / 10))
                pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3), steps=steps)
            
            # Perform click
            button = self.click_type_var.get()
            if button == 'left':
                pyautogui.click(x, y)
            elif button == 'right':
                pyautogui.rightClick(x, y)
            elif button == 'middle':
                pyautogui.middleClick(x, y)
            
            # Record click event
            click_duration = time.time() - click_start
            click_event = ClickEvent(time.time(), x, y, button, click_duration)
            
            # Add to AI engine
            self.ai_engine.add_click_pattern(click_event)
            
            # Save to database
            if self.current_session_id:
                self.database.save_click_event(click_event, self.current_session_id)
            
            # Update statistics
            self.total_clicks += 1
            self.click_positions.append((x, y))
            
            # Update UI
            self.root.after(0, self.update_statistics_display)
            
        except Exception as e:
            logger.error(f"Error performing click: {e}")
    
    def start_recording(self):
        """Start macro recording"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.current_macro = []
        self.macro_start_time = time.time()
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.record_thread.start()
        
        self.record_button.config(state='disabled')
        self.stop_record_button.config(state='normal')
        
        logger.info("Macro recording started")
    
    def stop_recording(self):
        """Stop macro recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Save macro
        if self.current_macro:
            macro_name = self.macro_name_var.get() or f"Macro_{len(self.macros) + 1}"
            self.macros.append({
                'name': macro_name,
                'actions': self.current_macro,
                'created_at': time.time()
            })
            
            # Update macro list
            self.macro_listbox.insert(tk.END, macro_name)
            
            # Save to database
            self.save_macro_to_database(macro_name, self.current_macro)
        
        self.record_button.config(state='normal')
        self.stop_record_button.config(state='disabled')
        
        logger.info(f"Macro recording stopped. Recorded {len(self.current_macro)} actions")
    
    def recording_loop(self):
        """Macro recording loop"""
        def on_click(x, y, button, pressed):
            if self.is_recording and pressed:
                action = MacroAction(
                    type='click',
                    data={'x': x, 'y': y, 'button': str(button)},
                    timestamp=time.time(),
                    delay=time.time() - self.macro_start_time
                )
                self.current_macro.append(action)
        
        def on_key(key):
            if self.is_recording:
                try:
                    action = MacroAction(
                        type='key',
                        data={'key': str(key)},
                        timestamp=time.time(),
                        delay=time.time() - self.macro_start_time
                    )
                    self.current_macro.append(action)
                except:
                    pass
        
        # Set up listeners
        mouse_listener = mouse.Listener(on_click=on_click)
        key_listener = keyboard.Listener(on_press=on_key)
        
        mouse_listener.start()
        key_listener.start()
        
        # Wait for recording to stop
        while self.is_recording:
            time.sleep(0.1)
        
        mouse_listener.stop()
        key_listener.stop()
    
    def play_macro(self):
        """Play selected macro"""
        selection = self.macro_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a macro to play")
            return
        
        macro_index = selection[0]
        macro = self.macros[macro_index]
        
        # Play macro in separate thread
        play_thread = threading.Thread(
            target=self.execute_macro, 
            args=(macro['actions'],), 
            daemon=True
        )
        play_thread.start()
        
        logger.info(f"Playing macro: {macro['name']}")
    
    def execute_macro(self, actions):
        """Execute macro actions"""
        for action in actions:
            try:
                if action.type == 'click':
                    data = action.data
                    pyautogui.click(data['x'], data['y'])
                elif action.type == 'key':
                    key = action.data['key']
                    pyautogui.press(key)
                
                # Wait for delay
                if hasattr(action, 'delay') and action.delay > 0:
                    time.sleep(min(action.delay, 5.0))  # Cap delay at 5 seconds
                
            except Exception as e:
                logger.error(f"Error executing macro action: {e}")
    
    def analyze_patterns(self):
        """Analyze clicking patterns using AI"""
        if len(self.ai_engine.click_patterns) < 10:
            messagebox.showinfo("Info", "Not enough data for pattern analysis")
            return
        
        # Perform analysis
        frequency = self.ai_engine.calculate_click_frequency()
        accuracy = self.ai_engine.calculate_accuracy_score()
        
        # Display results
        analysis_text = f"""
Pattern Analysis Results:
========================

Click Frequency: {frequency:.2f} clicks/second
Accuracy Score: {accuracy:.1f}%
Total Patterns: {len(self.ai_engine.click_patterns)}

AI Recommendations:
- {'Reduce click interval' if frequency > 15 else 'Increase click interval' if frequency < 5 else 'Current frequency is optimal'}
- {'Enable anti-detection features' if accuracy > 95 else 'Accuracy is within human range'}
- {'Consider using human-like movement' if not self.human_like_var.get() else 'Human-like movement is active'}
"""
        
        self.ai_status_text.delete(1.0, tk.END)
        self.ai_status_text.insert(tk.END, analysis_text)
        
        logger.info("Pattern analysis completed")
    
    def optimize_settings(self):
        """Optimize settings using AI"""
        optimized = self.ai_engine.optimize_settings(self.settings)
        
        # Apply optimizations
        if 'click_interval' in optimized:
            self.interval_var.set(str(optimized['click_interval']))
        
        if 'human_like_movement' in optimized:
            self.human_like_var.set(optimized['human_like_movement'])
        
        if 'randomize_timing' in optimized:
            self.randomize_var.set(optimized['randomize_timing'])
        
        messagebox.showinfo("Success", "Settings optimized by AI!")
        logger.info("Settings optimized by AI")
    
    def detect_targets(self):
        """Detect clickable targets using computer vision"""
        if not self.cv_enabled_var.get():
            messagebox.showwarning("Warning", "Computer vision is disabled")
            return
        
        try:
            # Capture screen
            self.computer_vision.capture_screen()
            
            # Auto-detect clickable elements
            targets = self.computer_vision.auto_detect_clickable_elements()
            
            if targets:
                # Use the first target
                x, y = targets[0]
                self.target_x_var.set(str(x))
                self.target_y_var.set(str(y))
                
                messagebox.showinfo("Success", f"Target detected at ({x}, {y})")
                logger.info(f"Target detected at ({x}, {y})")
            else:
                messagebox.showinfo("Info", "No targets detected")
                
        except Exception as e:
            logger.error(f"Error detecting targets: {e}")
            messagebox.showerror("Error", f"Target detection failed: {e}")
    
    def join_competition(self):
        """Join competitive mode"""
        player_name = "Player"  # In real app, get from user input
        
        if self.competitive_mode.join_competition(player_name):
            self.settings['competitive_mode_enabled'] = True
            messagebox.showinfo("Success", "Joined global competition!")
            self.refresh_leaderboard()
        else:
            messagebox.showerror("Error", "Failed to join competition")
    
    def refresh_leaderboard(self):
        """Refresh leaderboard display"""
        # Clear existing items
        for item in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(item)
        
        # Add leaderboard entries
        leaderboard = self.competitive_mode.get_leaderboard()
        for player, score, rank in leaderboard:
            self.leaderboard_tree.insert('', 'end', values=(rank, player, f"{score:.1f}"))
    
    def update_settings_from_ui(self):
        """Update settings from UI controls"""
        try:
            self.settings['click_interval'] = int(self.interval_var.get())
            self.settings['click_type'] = self.click_type_var.get()
            self.settings['target_x'] = int(self.target_x_var.get())
            self.settings['target_y'] = int(self.target_y_var.get())
            self.settings['randomize_timing'] = self.randomize_var.get()
            self.settings['human_like_movement'] = self.human_like_var.get()
            self.settings['anti_detection_enabled'] = self.anti_detection_var.get()
            self.settings['computer_vision_enabled'] = self.cv_enabled_var.get()
            self.settings['ai_optimization'] = self.ai_optimization_var.get()
            
        except ValueError as e:
            logger.error(f"Error updating settings: {e}")
    
    def update_statistics_display(self):
        """Update statistics display"""
        if not self.session_start_time:
            return
        
        elapsed_time = time.time() - self.session_start_time
        cps = self.total_clicks / max(elapsed_time, 1)
        
        # Get AI statistics
        frequency = self.ai_engine.calculate_click_frequency()
        accuracy = self.ai_engine.calculate_accuracy_score()
        
        # Check detection risk
        risk_level = self.anti_detection.check_detection_risk(frequency, accuracy)
        
        stats_text = f"""
Session Statistics:
==================

Total Clicks: {self.total_clicks:,}
Session Time: {elapsed_time:.1f}s
Clicks/Second: {cps:.2f}
AI Frequency: {frequency:.2f}
Accuracy: {accuracy:.1f}%
Detection Risk: {risk_level}

Recent Positions:
{', '.join([f'({x},{y})' for x, y in self.click_positions[-5:]])}
"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)
    
    def load_game_profiles(self) -> Dict[str, GameProfile]:
        """Load predefined game profiles"""
        profiles = {
            'Custom': GameProfile('Custom', 100, 'left', True, True, False, {}),
            'Minecraft': GameProfile('Minecraft', 50, 'left', True, True, False, {}),
            'Roblox': GameProfile('Roblox', 100, 'left', True, True, False, {}),
            'Cookie Clicker': GameProfile('Cookie Clicker', 10, 'left', False, False, False, {}),
            'Idle Games': GameProfile('Idle Games', 1000, 'left', False, False, False, {}),
            'FPS Games': GameProfile('FPS Games', 20, 'left', True, True, True, {})
        }
        
        return profiles
    
    def load_profile(self):
        """Load selected game profile"""
        profile_name = self.profile_var.get()
        if profile_name in self.game_profiles:
            profile = self.game_profiles[profile_name]
            
            # Update UI with profile settings
            self.interval_var.set(str(profile.click_interval))
            self.click_type_var.set(profile.click_type)
            self.anti_detection_var.set(profile.anti_detection)
            self.human_like_var.set(profile.human_like)
            self.cv_enabled_var.set(profile.target_detection)
            
            messagebox.showinfo("Success", f"Loaded profile: {profile_name}")
            logger.info(f"Loaded game profile: {profile_name}")
    
    def save_profile(self):
        """Save current settings as profile"""
        profile_name = self.profile_var.get()
        if not profile_name:
            messagebox.showwarning("Warning", "Please enter a profile name")
            return
        
        # Create profile from current settings
        self.update_settings_from_ui()
        profile = GameProfile(
            name=profile_name,
            click_interval=self.settings['click_interval'],
            click_type=self.settings['click_type'],
            anti_detection=self.settings['anti_detection_enabled'],
            human_like=self.settings['human_like_movement'],
            target_detection=self.settings['computer_vision_enabled'],
            settings=self.settings.copy()
        )
        
        self.game_profiles[profile_name] = profile
        messagebox.showinfo("Success", f"Profile saved: {profile_name}")
        logger.info(f"Saved game profile: {profile_name}")
    
    def save_settings(self):
        """Save settings to file"""
        try:
            with open('clicker_settings.json', 'w') as f:
                json.dump(self.settings, f, indent=2)
            logger.info("Settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists('clicker_settings.json'):
                with open('clicker_settings.json', 'r') as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
                logger.info("Settings loaded successfully")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
    def save_session_data(self):
        """Save session data to database"""
        try:
            if not self.session_start_time:
                return
            
            session_duration = time.time() - self.session_start_time
            avg_cps = self.total_clicks / max(session_duration, 1)
            accuracy = self.ai_engine.calculate_accuracy_score()
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (id, start_time, end_time, total_clicks, 
                                        average_cps, accuracy, settings)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (self.current_session_id, self.session_start_time, time.time(),
                      self.total_clicks, avg_cps, accuracy, json.dumps(self.settings)))
                conn.commit()
                
            logger.info(f"Session data saved: {self.total_clicks} clicks, {avg_cps:.2f} CPS")
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
    
    def refresh_analytics(self):
        """Refresh analytics display"""
        try:
            stats = self.database.get_statistics(30)  # Last 30 days
            
            analytics_text = f"""
Performance Analytics (Last 30 Days):
=====================================

Total Clicks: {stats.get('total_clicks', 0):,}
Average CPS: {stats.get('average_cps', 0):.2f}
Average Duration: {stats.get('average_duration', 0):.3f}s
Total Time: {stats.get('time_span', 0):.1f}s

AI Analysis:
- Pattern Recognition: {len(self.ai_engine.click_patterns)} patterns analyzed
- Model Trained: {'Yes' if self.ai_engine.is_trained else 'No'}
- Accuracy Score: {self.ai_engine.calculate_accuracy_score():.1f}%
- Click Frequency: {self.ai_engine.calculate_click_frequency():.2f} Hz

Anti-Detection Status:
- Risk Level: {self.anti_detection.check_detection_risk(self.ai_engine.calculate_click_frequency(), self.ai_engine.calculate_accuracy_score())}
- Human Patterns: {len(self.anti_detection.human_patterns)} loaded
- Jitter Enabled: {'Yes' if self.settings['anti_detection_enabled'] else 'No'}

Computer Vision:
- CV Enabled: {'Yes' if self.settings['computer_vision_enabled'] else 'No'}
- Last Screenshot: {'Available' if self.computer_vision.last_screenshot is not None else 'None'}
- Templates: {len(self.computer_vision.target_templates)} loaded

Competitive Mode:
- Enabled: {'Yes' if self.settings['competitive_mode_enabled'] else 'No'}
- Global Rank: {self.competitive_mode.player_stats.get('Player', {}).get('rank', 'N/A')}
- Best Score: {self.competitive_mode.player_stats.get('Player', {}).get('best_cps', 0):.2f}
"""
            
            self.analytics_text.delete(1.0, tk.END)
            self.analytics_text.insert(tk.END, analytics_text)
            
        except Exception as e:
            logger.error(f"Error refreshing analytics: {e}")
    
    def export_analytics(self):
        """Export analytics data"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                export_data = {
                    'settings': self.settings,
                    'statistics': self.database.get_statistics(30),
                    'ai_patterns': len(self.ai_engine.click_patterns),
                    'macros': len(self.macros),
                    'export_time': datetime.now().isoformat()
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Analytics exported to {filename}")
                logger.info(f"Analytics exported to {filename}")
                
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def run(self):
        """Run the application"""
        try:
            # Update analytics on startup
            self.refresh_analytics()
            
            # Start the main loop
            self.root.mainloop()
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            # Cleanup
            self.save_settings()
            if self.is_running:
                self.stop_clicking()
            logger.info("Application shutdown complete")

def main():
    """Main entry point"""
    try:
        # Check dependencies
        required_modules = ['numpy', 'sklearn', 'cv2', 'pyautogui', 'keyboard', 'mouse']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"Missing required modules: {', '.join(missing_modules)}")
            print("Please install them using: pip install numpy scikit-learn opencv-python pyautogui keyboard mouse pillow")
            return
        
        # Create and run application
        app = ClickerCarnage()
        app.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main() 