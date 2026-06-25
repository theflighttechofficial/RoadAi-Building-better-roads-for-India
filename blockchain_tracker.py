"""
blockchain_tracker.py — Blockchain-Powered Repair Tracking

Features:
  - Immutable detection records
  - Smart contract repair management
  - Milestone-based payments
  - Citizen verification
  - Public transparency portal
  - Tamper-proof audit trail
"""

import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sqlite3
import logging

log = logging.getLogger(__name__)


@dataclass
class BlockchainRecord:
    """Blockchain record for pothole detection"""
    record_id: str
    timestamp: int
    gps_lat: float
    gps_lon: float
    severity: str
    depth_cm: float
    photo_hash: str
    reporter_id: str
    status: str
    block_hash: str
    previous_hash: str


@dataclass
class RepairContract:
    """Smart contract for repair job"""
    contract_id: str
    detection_id: str
    contractor_id: str
    contractor_name: str
    amount_inr: float
    milestones: List[Dict]
    created_at: int
    warranty_days: int
    status: str
    total_released: float = 0.0


class SimpleBlockchain:
    """
    Simplified blockchain implementation
    
    Note: This is a proof-of-concept. For production, use:
    - Ethereum/Polygon for public blockchain
    - Hyperledger Fabric for private consortium
    - IPFS for photo storage
    """
    
    def __init__(self, db_path: str = "blockchain.db"):
        self.db_path = Path(db_path)
        self._init_database()
        log.info("Blockchain initialized")
    
    def _init_database(self):
        """Initialize blockchain database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Blockchain records
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blockchain (
                record_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                gps_lat REAL,
                gps_lon REAL,
                severity TEXT,
                depth_cm REAL,
                photo_hash TEXT,
                reporter_id TEXT,
                status TEXT,
                block_hash TEXT,
                previous_hash TEXT,
                block_number INTEGER
            )
        ''')
        
        # Repair contracts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contracts (
                contract_id TEXT PRIMARY KEY,
                detection_id TEXT,
                contractor_id TEXT,
                contractor_name TEXT,
                amount_inr REAL,
                milestones TEXT,
                created_at INTEGER,
                warranty_days INTEGER,
                status TEXT,
                total_released REAL DEFAULT 0,
                completed_at INTEGER,
                FOREIGN KEY (detection_id) REFERENCES blockchain(record_id)
            )
        ''')
        
        # Contract events (milestone completions, verifications)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contract_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT,
                event_type TEXT,
                actor_id TEXT,
                timestamp INTEGER,
                data TEXT,
                FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
            )
        ''')
        
        # Citizen verifications
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT,
                citizen_id TEXT,
                photo_hash TEXT,
                approved BOOLEAN,
                timestamp INTEGER,
                comments TEXT,
                FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _calculate_hash(self, data: Dict) -> str:
        """Calculate SHA-256 hash of data"""
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _get_last_block_hash(self) -> str:
        """Get hash of last block in chain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT block_hash FROM blockchain 
            ORDER BY block_number DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return row[0]
        else:
            # Genesis block
            return "0" * 64
    
    def record_detection(self, gps: tuple, severity: str, depth_cm: float,
                        photo_hash: str, reporter_id: str, 
                        metadata: Dict = None) -> Dict:
        """
        Record pothole detection on blockchain
        
        Args:
            gps: (latitude, longitude)
            severity: Severity level
            depth_cm: Depth in centimeters
            photo_hash: SHA-256 hash of photo
            reporter_id: User who reported
            metadata: Additional detection data
        
        Returns:
            Blockchain record with immutable hash
        """
        record_id = f"detect_{int(time.time())}_{reporter_id[:8]}"
        timestamp = int(time.time())
        
        # Get previous block hash
        previous_hash = self._get_last_block_hash()
        
        # Create block data
        block_data = {
            'record_id': record_id,
            'timestamp': timestamp,
            'gps_lat': gps[0],
            'gps_lon': gps[1],
            'severity': severity,
            'depth_cm': depth_cm,
            'photo_hash': photo_hash,
            'reporter_id': reporter_id,
            'status': 'reported',
            'previous_hash': previous_hash,
            'metadata': metadata or {}
        }
        
        # Calculate block hash
        block_hash = self._calculate_hash(block_data)
        
        # Get next block number
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT MAX(block_number) FROM blockchain')
        max_block = cursor.fetchone()[0]
        block_number = (max_block + 1) if max_block is not None else 1
        
        # Insert into blockchain
        cursor.execute('''
            INSERT INTO blockchain 
            (record_id, timestamp, gps_lat, gps_lon, severity, depth_cm,
             photo_hash, reporter_id, status, block_hash, previous_hash, block_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (record_id, timestamp, gps[0], gps[1], severity, depth_cm,
              photo_hash, reporter_id, 'reported', block_hash, previous_hash, block_number))
        
        conn.commit()
        conn.close()
        
        log.info(f"Blockchain record created: {record_id} (block #{block_number})")
        
        return {
            'record_id': record_id,
            'block_number': block_number,
            'block_hash': block_hash,
            'timestamp': timestamp,
            'status': 'recorded'
        }
    
    def create_repair_contract(self, detection_id: str, contractor_id: str,
                               contractor_name: str, amount_inr: float,
                               warranty_days: int = 30) -> Dict:
        """
        Create smart contract for repair
        
        Milestone-based payment:
        1. Contract awarded → 20% advance
        2. Work started → 30% (photo proof required)
        3. Work completed → hold 50%
        4. Citizen verification → release 50%
        5. Warranty period → complete
        """
        contract_id = f"contract_{int(time.time())}_{contractor_id[:8]}"
        
        milestones = [
            {
                'stage': 'awarded',
                'percent': 20,
                'amount': amount_inr * 0.20,
                'status': 'pending',
                'completed_at': None
            },
            {
                'stage': 'started',
                'percent': 30,
                'amount': amount_inr * 0.30,
                'status': 'pending',
                'completed_at': None,
                'requires': 'photo_proof'
            },
            {
                'stage': 'completed',
                'percent': 0,
                'amount': 0,
                'status': 'pending',
                'completed_at': None,
                'requires': 'photo_proof'
            },
            {
                'stage': 'verified',
                'percent': 50,
                'amount': amount_inr * 0.50,
                'status': 'pending',
                'completed_at': None,
                'requires': 'citizen_verification_3'
            },
            {
                'stage': 'warranty',
                'percent': 0,
                'amount': 0,
                'status': 'pending',
                'completed_at': None,
                'duration_days': warranty_days
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO contracts
            (contract_id, detection_id, contractor_id, contractor_name, amount_inr,
             milestones, created_at, warranty_days, status, total_released)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (contract_id, detection_id, contractor_id, contractor_name, amount_inr,
              json.dumps(milestones), int(time.time()), warranty_days, 'active', 0))
        
        # Log event
        cursor.execute('''
            INSERT INTO contract_events (contract_id, event_type, actor_id, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (contract_id, 'contract_created', contractor_id, int(time.time()),
              json.dumps({'amount': amount_inr, 'warranty_days': warranty_days})))
        
        # Automatically release first milestone (awarded)
        self._release_milestone(cursor, contract_id, 0, contractor_id)
        
        conn.commit()
        conn.close()
        
        log.info(f"Repair contract created: {contract_id}")
        
        return {
            'contract_id': contract_id,
            'detection_id': detection_id,
            'contractor_id': contractor_id,
            'amount_inr': amount_inr,
            'status': 'active',
            'milestones': milestones
        }
    
    def _release_milestone(self, cursor, contract_id: str, milestone_index: int,
                          actor_id: str):
        """Release payment for milestone"""
        # Get contract
        cursor.execute('SELECT milestones, total_released FROM contracts WHERE contract_id = ?',
                      (contract_id,))
        row = cursor.fetchone()
        
        if not row:
            return
        
        milestones = json.loads(row[0])
        total_released = row[1]
        
        if milestone_index >= len(milestones):
            return
        
        milestone = milestones[milestone_index]
        
        if milestone['status'] == 'completed':
            return  # Already released
        
        # Mark as completed
        milestone['status'] = 'completed'
        milestone['completed_at'] = int(time.time())
        
        # Update total released
        total_released += milestone['amount']
        
        # Update database
        cursor.execute('''
            UPDATE contracts 
            SET milestones = ?, total_released = ?
            WHERE contract_id = ?
        ''', (json.dumps(milestones), total_released, contract_id))
        
        # Log event
        cursor.execute('''
            INSERT INTO contract_events (contract_id, event_type, actor_id, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (contract_id, 'milestone_released', actor_id, int(time.time()),
              json.dumps({'milestone': milestone_index, 'amount': milestone['amount']})))
        
        log.info(f"Milestone {milestone_index} released for {contract_id}: ₹{milestone['amount']}")
    
    def update_work_status(self, contract_id: str, status: str, 
                          photo_hash: str, actor_id: str) -> Dict:
        """
        Contractor updates work status
        
        Args:
            contract_id: Contract ID
            status: 'started' or 'completed'
            photo_hash: SHA-256 hash of proof photo
            actor_id: Contractor ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get contract
        cursor.execute('SELECT milestones FROM contracts WHERE contract_id = ?',
                      (contract_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {'error': 'Contract not found'}
        
        milestones = json.loads(row[0])
        
        # Release appropriate milestone
        if status == 'started':
            self._release_milestone(cursor, contract_id, 1, actor_id)
            
            # Log photo proof
            cursor.execute('''
                INSERT INTO contract_events (contract_id, event_type, actor_id, timestamp, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (contract_id, 'work_started', actor_id, int(time.time()),
                  json.dumps({'photo_hash': photo_hash})))
        
        elif status == 'completed':
            # Mark completed but don't release payment yet (needs citizen verification)
            milestones[2]['status'] = 'completed'
            milestones[2]['completed_at'] = int(time.time())
            
            cursor.execute('UPDATE contracts SET milestones = ? WHERE contract_id = ?',
                          (json.dumps(milestones), contract_id))
            
            cursor.execute('''
                INSERT INTO contract_events (contract_id, event_type, actor_id, timestamp, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (contract_id, 'work_completed', actor_id, int(time.time()),
                  json.dumps({'photo_hash': photo_hash})))
        
        conn.commit()
        conn.close()
        
        return {'status': 'updated', 'contract_id': contract_id}
    
    def citizen_verify_repair(self, contract_id: str, citizen_id: str,
                             photo_hash: str, approved: bool, 
                             comments: str = None) -> Dict:
        """
        Citizen verifies repair completion
        
        Requires 3+ unique citizen approvals to release payment
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Record verification
        cursor.execute('''
            INSERT INTO verifications (contract_id, citizen_id, photo_hash, approved, timestamp, comments)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (contract_id, citizen_id, photo_hash, approved, int(time.time()), comments))
        
        # Count approvals
        cursor.execute('''
            SELECT COUNT(DISTINCT citizen_id) 
            FROM verifications 
            WHERE contract_id = ? AND approved = 1
        ''', (contract_id,))
        
        approval_count = cursor.fetchone()[0]
        
        # Check if threshold met (3 approvals)
        if approval_count >= 3:
            # Release final payment
            self._release_milestone(cursor, contract_id, 3, 'system_auto')
            
            # Start warranty period
            cursor.execute('SELECT milestones FROM contracts WHERE contract_id = ?', 
                          (contract_id,))
            milestones = json.loads(cursor.fetchone()[0])
            
            milestones[4]['status'] = 'active'
            milestones[4]['completed_at'] = int(time.time())
            
            cursor.execute('UPDATE contracts SET milestones = ?, status = ? WHERE contract_id = ?',
                          (json.dumps(milestones), 'warranty', contract_id))
            
            status_msg = 'payment_released'
        else:
            status_msg = f'verified_{approval_count}/3'
        
        conn.commit()
        conn.close()
        
        log.info(f"Citizen verification: {contract_id} - {status_msg}")
        
        return {
            'status': status_msg,
            'approval_count': approval_count,
            'threshold': 3
        }
    
    def get_contract(self, contract_id: str) -> Optional[Dict]:
        """Get contract details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM contracts WHERE contract_id = ?', (contract_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get events
        cursor.execute('''
            SELECT event_type, actor_id, timestamp, data
            FROM contract_events
            WHERE contract_id = ?
            ORDER BY timestamp DESC
        ''', (contract_id,))
        
        events = []
        for event_row in cursor.fetchall():
            events.append({
                'event_type': event_row[0],
                'actor_id': event_row[1],
                'timestamp': event_row[2],
                'data': json.loads(event_row[3]) if event_row[3] else {}
            })
        
        # Get verifications
        cursor.execute('''
            SELECT citizen_id, approved, timestamp, comments
            FROM verifications
            WHERE contract_id = ?
        ''', (contract_id,))
        
        verifications = []
        for ver_row in cursor.fetchall():
            verifications.append({
                'citizen_id': ver_row[0],
                'approved': bool(ver_row[1]),
                'timestamp': ver_row[2],
                'comments': ver_row[3]
            })
        
        conn.close()
        
        contract = {
            'contract_id': row[0],
            'detection_id': row[1],
            'contractor_id': row[2],
            'contractor_name': row[3],
            'amount_inr': row[4],
            'milestones': json.loads(row[5]),
            'created_at': row[6],
            'warranty_days': row[7],
            'status': row[8],
            'total_released': row[9],
            'completed_at': row[10],
            'events': events,
            'verifications': verifications,
            'verification_count': len([v for v in verifications if v['approved']])
        }
        
        return contract
    
    def get_public_explorer_data(self, city: str = None) -> Dict:
        """
        Generate data for public blockchain explorer
        
        Returns statistics and recent records for transparency portal
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('SELECT COUNT(*), AVG(depth_cm) FROM blockchain')
        total_records, avg_depth = cursor.fetchone()
        
        cursor.execute('SELECT COUNT(*), SUM(amount_inr), SUM(total_released) FROM contracts')
        total_contracts, total_allocated, total_released = cursor.fetchone()
        
        # Recent detections
        cursor.execute('''
            SELECT record_id, timestamp, gps_lat, gps_lon, severity, status, block_number
            FROM blockchain
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        
        recent_detections = []
        for row in cursor.fetchall():
            recent_detections.append({
                'record_id': row[0],
                'timestamp': row[1],
                'gps': [row[2], row[3]],
                'severity': row[4],
                'status': row[5],
                'block_number': row[6]
            })
        
        # Active contracts
        cursor.execute('''
            SELECT contract_id, contractor_name, amount_inr, status, total_released
            FROM contracts
            WHERE status IN ('active', 'warranty')
            ORDER BY created_at DESC
            LIMIT 20
        ''')
        
        active_contracts = []
        for row in cursor.fetchall():
            active_contracts.append({
                'contract_id': row[0],
                'contractor_name': row[1],
                'amount_inr': row[2],
                'status': row[3],
                'total_released': row[4],
                'completion_pct': (row[4] / row[2] * 100) if row[2] > 0 else 0
            })
        
        conn.close()
        
        explorer_data = {
            'statistics': {
                'total_detections': total_records or 0,
                'avg_depth_cm': round(avg_depth, 1) if avg_depth else 0,
                'total_contracts': total_contracts or 0,
                'total_allocated_inr': total_allocated or 0,
                'total_released_inr': total_released or 0,
                'pending_release_inr': (total_allocated or 0) - (total_released or 0),
                'transparency_score': 100  # Always 100% with blockchain
            },
            'recent_detections': recent_detections,
            'active_contracts': active_contracts,
            'last_updated': int(time.time())
        }
        
        return explorer_data
    
    def verify_chain_integrity(self) -> Dict:
        """
        Verify blockchain integrity
        
        Checks if any blocks have been tampered with
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM blockchain ORDER BY block_number')
        rows = cursor.fetchall()
        conn.close()
        
        issues = []
        
        for i, row in enumerate(rows):
            record_id, timestamp, gps_lat, gps_lon, severity, depth_cm, photo_hash, reporter_id, status, block_hash, previous_hash, block_number = row
            
            # Recalculate hash
            block_data = {
                'record_id': record_id,
                'timestamp': timestamp,
                'gps_lat': gps_lat,
                'gps_lon': gps_lon,
                'severity': severity,
                'depth_cm': depth_cm,
                'photo_hash': photo_hash,
                'reporter_id': reporter_id,
                'status': status,
                'previous_hash': previous_hash
            }
            
            calculated_hash = self._calculate_hash(block_data)
            
            if calculated_hash != block_hash:
                issues.append({
                    'block_number': block_number,
                    'record_id': record_id,
                    'issue': 'hash_mismatch',
                    'stored_hash': block_hash,
                    'calculated_hash': calculated_hash
                })
            
            # Check chain linkage
            if i > 0:
                prev_row = rows[i-1]
                if previous_hash != prev_row[9]:  # prev_row's block_hash
                    issues.append({
                        'block_number': block_number,
                        'record_id': record_id,
                        'issue': 'broken_chain',
                        'expected_previous': prev_row[9],
                        'actual_previous': previous_hash
                    })
        
        return {
            'valid': len(issues) == 0,
            'total_blocks': len(rows),
            'issues': issues
        }