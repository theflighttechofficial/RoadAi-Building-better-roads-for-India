"""
gamification.py — Citizen Engagement Gamification System

Features:
  - Point system for verified reports
  - Badges and achievements
  - City/ward/street leaderboards
  - Weekly challenges
  - Rewards tracking
  - Social sharing
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

log = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile with gamification stats"""
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    total_points: int = 0
    reports_submitted: int = 0
    reports_verified: int = 0
    reports_fixed: int = 0
    badges: List[str] = None
    rank: int = 0
    ward: Optional[str] = None
    joined_date: str = None
    
    def __post_init__(self):
        if self.badges is None:
            self.badges = []
        if self.joined_date is None:
            self.joined_date = datetime.now().isoformat()


@dataclass
class Badge:
    """Achievement badge"""
    badge_id: str
    name: str
    description: str
    icon: str
    requirement: Dict
    points_reward: int


@dataclass
class Challenge:
    """Community challenge"""
    challenge_id: str
    title: str
    description: str
    goal_type: str  # 'reports', 'fixes', 'streak'
    goal_value: int
    current_progress: int
    start_date: str
    end_date: str
    reward: str
    participants: int = 0
    ward: Optional[str] = None
    status: str = 'active'  # active, completed, expired


class GamificationEngine:
    """
    Main gamification engine
    """
    
    # Point system
    POINTS = {
        'report_submitted': 10,
        'report_verified': 20,
        'first_reporter': 15,
        'high_quality_photo': 5,
        'gps_tagged': 3,
        'report_fixed': 50,
        'weekly_streak': 20,
        'challenge_complete': 100,
        'referral': 25,
    }
    
    # Badge definitions
    BADGES = [
        Badge(
            badge_id='pothole_hunter',
            name='Pothole Hunter',
            description='Report 100 potholes',
            icon='🎯',
            requirement={'reports_verified': 100},
            points_reward=500
        ),
        Badge(
            badge_id='night_owl',
            name='Night Owl',
            description='Report 20 issues at night',
            icon='🦉',
            requirement={'night_reports': 20},
            points_reward=200
        ),
        Badge(
            badge_id='monsoon_warrior',
            name='Monsoon Warrior',
            description='Report 30 issues during rain',
            icon='⛈️',
            requirement={'rain_reports': 30},
            points_reward=300
        ),
        Badge(
            badge_id='speed_demon',
            name='Speed Demon',
            description='Report 10 issues in 24 hours',
            icon='⚡',
            requirement={'reports_in_24h': 10},
            points_reward=150
        ),
        Badge(
            badge_id='civic_champion',
            name='Civic Champion',
            description='50 of your reports got fixed',
            icon='🏆',
            requirement={'reports_fixed': 50},
            points_reward=1000
        ),
        Badge(
            badge_id='pioneer',
            name='Pioneer',
            description='First to report in your ward',
            icon='🚀',
            requirement={'first_in_ward': True},
            points_reward=100
        ),
        Badge(
            badge_id='streak_master',
            name='Streak Master',
            description='Report for 7 consecutive days',
            icon='🔥',
            requirement={'daily_streak': 7},
            points_reward=250
        ),
    ]
    
    def __init__(self, db_path: str = "gamification.db"):
        self.db_path = Path(db_path)
        self._init_database()
        log.info("Gamification engine initialized")
    
    def _init_database(self):
        """Initialize SQLite database for gamification"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                total_points INTEGER DEFAULT 0,
                reports_submitted INTEGER DEFAULT 0,
                reports_verified INTEGER DEFAULT 0,
                reports_fixed INTEGER DEFAULT 0,
                badges TEXT DEFAULT '[]',
                rank INTEGER DEFAULT 0,
                ward TEXT,
                joined_date TEXT
            )
        ''')
        
        # Activity log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                action TEXT,
                points INTEGER,
                timestamp TEXT,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Challenges
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS challenges (
                challenge_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                goal_type TEXT,
                goal_value INTEGER,
                current_progress INTEGER DEFAULT 0,
                start_date TEXT,
                end_date TEXT,
                reward TEXT,
                participants INTEGER DEFAULT 0,
                ward TEXT,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Leaderboards (materialized view, updated periodically)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leaderboard (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                total_points INTEGER,
                ward TEXT,
                rank_global INTEGER,
                rank_ward INTEGER,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, user_id: str, name: str, email: str = None, 
                     phone: str = None, ward: str = None) -> UserProfile:
        """Register new user"""
        profile = UserProfile(
            user_id=user_id,
            name=name,
            email=email,
            phone=phone,
            ward=ward
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO users 
            (user_id, name, email, phone, ward, joined_date, badges)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, name, email, phone, ward, profile.joined_date, '[]'))
        
        conn.commit()
        conn.close()
        
        log.info(f"User registered: {user_id} ({name})")
        return profile
    
    def award_points(self, user_id: str, action: str, metadata: Dict = None) -> Dict:
        """
        Award points for user action
        
        Args:
            user_id: User identifier
            action: Action type (e.g., 'report_submitted')
            metadata: Additional context (severity, time, etc.)
        
        Returns:
            Award summary with points, badges, rank
        """
        if metadata is None:
            metadata = {}
        
        # Calculate base points
        points = self.POINTS.get(action, 0)
        
        # Apply multipliers
        if metadata.get('severity') == 'critical':
            points = int(points * 2)  # 2x for critical damage
        
        if metadata.get('time_of_day') == 'night':
            points = int(points * 1.5)  # 1.5x for night reports
        
        if metadata.get('weather') == 'rain':
            points = int(points * 1.3)  # 1.3x for rain reports
        
        # Update user points
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET total_points = total_points + ?
            WHERE user_id = ?
        ''', (points, user_id))
        
        # Update activity counters
        if action == 'report_submitted':
            cursor.execute('UPDATE users SET reports_submitted = reports_submitted + 1 WHERE user_id = ?', (user_id,))
        elif action == 'report_verified':
            cursor.execute('UPDATE users SET reports_verified = reports_verified + 1 WHERE user_id = ?', (user_id,))
        elif action == 'report_fixed':
            cursor.execute('UPDATE users SET reports_fixed = reports_fixed + 1 WHERE user_id = ?', (user_id,))
        
        # Log activity
        cursor.execute('''
            INSERT INTO activity_log (user_id, action, points, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, action, points, datetime.now().isoformat(), json.dumps(metadata)))
        
        conn.commit()
        
        # Get updated profile
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        # Check badge unlocks
        new_badges = self.check_badge_unlocks(user_id)
        
        # Update leaderboard
        self.update_leaderboard()
        
        # Get current rank
        rank = self.get_user_rank(user_id)
        
        result = {
            'points_earned': points,
            'total_points': row[4] if row else 0,
            'new_badges': new_badges,
            'rank_global': rank['global'],
            'rank_ward': rank['ward'],
            'action': action
        }
        
        log.info(f"Awarded {points} points to {user_id} for {action}")
        
        return result
    
    def check_badge_unlocks(self, user_id: str) -> List[Dict]:
        """Check if user unlocked new badges"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return []
        
        user_badges = json.loads(row[8])  # badges column
        reports_verified = row[6]
        reports_fixed = row[7]
        
        new_badges = []
        
        for badge in self.BADGES:
            if badge.badge_id in user_badges:
                continue  # Already has this badge
            
            # Check requirements
            unlocked = False
            
            if 'reports_verified' in badge.requirement:
                if reports_verified >= badge.requirement['reports_verified']:
                    unlocked = True
            
            if 'reports_fixed' in badge.requirement:
                if reports_fixed >= badge.requirement['reports_fixed']:
                    unlocked = True
            
            # Add more requirement checks as needed
            
            if unlocked:
                user_badges.append(badge.badge_id)
                new_badges.append({
                    'badge_id': badge.badge_id,
                    'name': badge.name,
                    'description': badge.description,
                    'icon': badge.icon,
                    'points_reward': badge.points_reward
                })
                
                # Award badge points
                cursor.execute('''
                    UPDATE users 
                    SET total_points = total_points + ?,
                        badges = ?
                    WHERE user_id = ?
                ''', (badge.points_reward, json.dumps(user_badges), user_id))
                
                log.info(f"Badge unlocked: {badge.name} for {user_id}")
        
        conn.commit()
        conn.close()
        
        return new_badges
    
    def get_leaderboard(self, scope: str = 'global', ward: str = None, limit: int = 100) -> List[Dict]:
        """
        Get leaderboard
        
        Args:
            scope: 'global', 'ward', or 'street'
            ward: Ward name (required if scope='ward')
            limit: Number of top users
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if scope == 'global':
            cursor.execute('''
                SELECT user_id, name, total_points, ward, reports_verified, reports_fixed, badges
                FROM users
                ORDER BY total_points DESC
                LIMIT ?
            ''', (limit,))
        elif scope == 'ward' and ward:
            cursor.execute('''
                SELECT user_id, name, total_points, ward, reports_verified, reports_fixed, badges
                FROM users
                WHERE ward = ?
                ORDER BY total_points DESC
                LIMIT ?
            ''', (ward, limit))
        else:
            cursor.execute('''
                SELECT user_id, name, total_points, ward, reports_verified, reports_fixed, badges
                FROM users
                ORDER BY total_points DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        leaderboard = []
        for i, row in enumerate(rows, 1):
            leaderboard.append({
                'rank': i,
                'user_id': row[0],
                'name': row[1],
                'total_points': row[2],
                'ward': row[3],
                'reports_verified': row[4],
                'reports_fixed': row[5],
                'badges': json.loads(row[6]) if row[6] else [],
                'badge_count': len(json.loads(row[6])) if row[6] else 0
            })
        
        return leaderboard
    
    def get_user_rank(self, user_id: str) -> Dict:
        """Get user's rank (global and ward)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user's ward
        cursor.execute('SELECT ward FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        ward = row[0] if row else None
        
        # Global rank
        cursor.execute('''
            SELECT COUNT(*) + 1
            FROM users
            WHERE total_points > (SELECT total_points FROM users WHERE user_id = ?)
        ''', (user_id,))
        global_rank = cursor.fetchone()[0]
        
        # Ward rank
        ward_rank = None
        if ward:
            cursor.execute('''
                SELECT COUNT(*) + 1
                FROM users
                WHERE ward = ? AND total_points > (SELECT total_points FROM users WHERE user_id = ?)
            ''', (ward, user_id))
            ward_rank = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'global': global_rank,
            'ward': ward_rank
        }
    
    def update_leaderboard(self):
        """Update materialized leaderboard view"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete old data
        cursor.execute('DELETE FROM leaderboard')
        
        # Rebuild
        cursor.execute('''
            INSERT INTO leaderboard (user_id, name, total_points, ward, rank_global, rank_ward, last_updated)
            SELECT 
                user_id,
                name,
                total_points,
                ward,
                ROW_NUMBER() OVER (ORDER BY total_points DESC) as rank_global,
                ROW_NUMBER() OVER (PARTITION BY ward ORDER BY total_points DESC) as rank_ward,
                ?
            FROM users
        ''', (datetime.now().isoformat(),))
        
        conn.commit()
        conn.close()
    
    def create_challenge(self, title: str, description: str, goal_type: str,
                        goal_value: int, duration_days: int, reward: str,
                        ward: str = None) -> Challenge:
        """Create community challenge"""
        challenge_id = f"challenge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        challenge = Challenge(
            challenge_id=challenge_id,
            title=title,
            description=description,
            goal_type=goal_type,
            goal_value=goal_value,
            current_progress=0,
            start_date=datetime.now().isoformat(),
            end_date=(datetime.now() + timedelta(days=duration_days)).isoformat(),
            reward=reward,
            ward=ward
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO challenges 
            (challenge_id, title, description, goal_type, goal_value, 
             start_date, end_date, reward, ward, current_progress, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (challenge.challenge_id, challenge.title, challenge.description,
              challenge.goal_type, challenge.goal_value, challenge.start_date,
              challenge.end_date, challenge.reward, challenge.ward, 0, 'active'))
        
        conn.commit()
        conn.close()
        
        log.info(f"Challenge created: {title}")
        
        return challenge
    
    def get_active_challenges(self, ward: str = None) -> List[Dict]:
        """Get active challenges"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if ward:
            cursor.execute('''
                SELECT * FROM challenges 
                WHERE status = 'active' AND (ward = ? OR ward IS NULL)
                AND end_date > ?
            ''', (ward, datetime.now().isoformat()))
        else:
            cursor.execute('''
                SELECT * FROM challenges 
                WHERE status = 'active' AND end_date > ?
            ''', (datetime.now().isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        challenges = []
        for row in rows:
            challenges.append({
                'challenge_id': row[0],
                'title': row[1],
                'description': row[2],
                'goal_type': row[3],
                'goal_value': row[4],
                'current_progress': row[5],
                'start_date': row[6],
                'end_date': row[7],
                'reward': row[8],
                'participants': row[9],
                'ward': row[10],
                'status': row[11],
                'completion_pct': (row[5] / row[4] * 100) if row[4] > 0 else 0
            })
        
        return challenges
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get complete user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get recent activity
        cursor.execute('''
            SELECT action, points, timestamp, metadata
            FROM activity_log
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (user_id,))
        
        activity = []
        for act_row in cursor.fetchall():
            activity.append({
                'action': act_row[0],
                'points': act_row[1],
                'timestamp': act_row[2],
                'metadata': json.loads(act_row[3]) if act_row[3] else {}
            })
        
        conn.close()
        
        rank = self.get_user_rank(user_id)
        
        profile = {
            'user_id': row[0],
            'name': row[1],
            'email': row[2],
            'phone': row[3],
            'total_points': row[4],
            'reports_submitted': row[5],
            'reports_verified': row[6],
            'reports_fixed': row[7],
            'badges': json.loads(row[8]) if row[8] else [],
            'ward': row[10],
            'joined_date': row[11],
            'rank_global': rank['global'],
            'rank_ward': rank['ward'],
            'recent_activity': activity
        }
        
        return profile
    
    def get_stats(self) -> Dict:
        """Get overall gamification statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*), SUM(total_points), SUM(reports_verified) FROM users')
        row = cursor.fetchone()
        
        cursor.execute('SELECT COUNT(*) FROM challenges WHERE status = "active"')
        active_challenges = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_users': row[0] or 0,
            'total_points_awarded': row[1] or 0,
            'total_reports_verified': row[2] or 0,
            'active_challenges': active_challenges,
            'avg_points_per_user': (row[1] / row[0]) if row[0] > 0 else 0
        }