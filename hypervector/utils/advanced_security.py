"""Advanced security and monitoring for HDC systems.

Provides comprehensive security, privacy protection, and monitoring
for production hyperdimensional computing deployments.
"""

import torch
import hashlib
import secrets
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
from functools import wraps

from ..core.hypervector import HyperVector
from .logging import get_logger

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AccessLevel(Enum):
    """Access permission levels."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    security_level: SecurityLevel
    access_levels: List[AccessLevel]
    session_token: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class AuditLogEntry:
    """Audit log entry for security monitoring."""
    timestamp: float
    user_id: str
    operation: str
    resource: str
    success: bool
    security_level: SecurityLevel
    details: Dict[str, Any]


class EncryptionManager:
    """Manages encryption and decryption of sensitive HDC data."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        
        self.master_key = master_key
        self.fernet = Fernet(master_key)
        self.derived_keys = {}
        
        logger.info("Encryption manager initialized")
    
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_hypervector(self, hv: HyperVector, security_level: SecurityLevel) -> Dict[str, Any]:
        """Encrypt hypervector based on security level."""
        try:
            # Serialize hypervector data
            data = {
                'vector_data': hv.data.cpu().numpy().tolist(),
                'mode': hv.mode,
                'metadata': getattr(hv, 'metadata', {})
            }
            
            serialized = json.dumps(data).encode()
            
            if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                # Use double encryption for highest security
                encrypted_once = self.fernet.encrypt(serialized)
                encrypted_twice = self.fernet.encrypt(encrypted_once)
                encrypted_data = encrypted_twice
            else:
                # Single encryption for lower security levels
                encrypted_data = self.fernet.encrypt(serialized)
            
            return {
                'encrypted_data': base64.b64encode(encrypted_data).decode(),
                'security_level': security_level.value,
                'timestamp': time.time(),
                'checksum': hashlib.sha256(encrypted_data).hexdigest()
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_hypervector(self, encrypted_package: Dict[str, Any]) -> HyperVector:
        """Decrypt hypervector package."""
        try:
            encrypted_data = base64.b64decode(encrypted_package['encrypted_data'])
            security_level = SecurityLevel(encrypted_package['security_level'])
            
            # Verify checksum
            expected_checksum = encrypted_package['checksum']
            actual_checksum = hashlib.sha256(encrypted_data).hexdigest()
            
            if expected_checksum != actual_checksum:
                raise SecurityError("Data integrity check failed - possible tampering")
            
            if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                # Double decryption
                decrypted_once = self.fernet.decrypt(encrypted_data)
                decrypted_data = self.fernet.decrypt(decrypted_once)
            else:
                # Single decryption
                decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Deserialize
            data = json.loads(decrypted_data.decode())
            
            vector_data = torch.tensor(data['vector_data'])
            hv = HyperVector(vector_data, mode=data['mode'])
            
            if 'metadata' in data:
                hv.metadata = data['metadata']
            
            return hv
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise


class AccessControl:
    """Role-based access control for HDC operations."""
    
    def __init__(self):
        self.user_permissions = {}
        self.role_permissions = {
            'user': [AccessLevel.READ],
            'analyst': [AccessLevel.read, AccessLevel.execute],
            'researcher': [AccessLevel.read, AccessLevel.write, AccessLevel.execute],
            'admin': [AccessLevel.read, AccessLevel.write, AccessLevel.execute, AccessLevel.admin]
        }
        
        self.resource_security_levels = {}
        self.active_sessions = {}
        
        logger.info("Access control system initialized")
    
    def create_user_session(self, user_id: str, role: str, security_clearance: SecurityLevel) -> str:
        """Create authenticated user session."""
        session_token = secrets.token_urlsafe(32)
        
        permissions = self.role_permissions.get(role, [AccessLevel.read])
        
        context = SecurityContext(
            user_id=user_id,
            security_level=security_clearance,
            access_levels=permissions,
            session_token=session_token,
            timestamp=time.time(),
            metadata={'role': role}
        )
        
        self.active_sessions[session_token] = context
        
        logger.info(f"Created session for user {user_id} with role {role}")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[SecurityContext]:
        """Validate session token and return security context."""
        if session_token not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_token]
        
        # Check session timeout (24 hours)
        if time.time() - context.timestamp > 86400:
            del self.active_sessions[session_token]
            return None
        
        return context
    
    def check_permission(self, session_token: str, operation: str, 
                        resource: str, required_level: AccessLevel) -> bool:
        """Check if user has permission for operation."""
        context = self.validate_session(session_token)
        if not context:
            return False
        
        # Check access level
        if required_level not in context.access_levels:
            return False
        
        # Check security clearance for resource
        resource_security_level = self.resource_security_levels.get(resource, SecurityLevel.PUBLIC)
        
        # Security level hierarchy
        security_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        user_clearance = security_hierarchy[context.security_level]
        required_clearance = security_hierarchy[resource_security_level]
        
        return user_clearance >= required_clearance
    
    def set_resource_security_level(self, resource: str, level: SecurityLevel):
        """Set security level for a resource."""
        self.resource_security_levels[resource] = level
        logger.info(f"Set security level for {resource} to {level.value}")


class AuditLogger:
    """Comprehensive audit logging for security monitoring."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.audit_entries: List[AuditLogEntry] = []
        self._lock = threading.Lock()
        
        logger.info(f"Audit logger initialized: {log_file}")
    
    def log_operation(self, user_id: str, operation: str, resource: str, 
                     success: bool, security_level: SecurityLevel, 
                     details: Optional[Dict[str, Any]] = None):
        """Log security-relevant operation."""
        entry = AuditLogEntry(
            timestamp=time.time(),
            user_id=user_id,
            operation=operation,
            resource=resource,
            success=success,
            security_level=security_level,
            details=details or {}
        )
        
        with self._lock:
            self.audit_entries.append(entry)
            self._write_to_file(entry)
    
    def _write_to_file(self, entry: AuditLogEntry):
        """Write audit entry to file."""
        try:
            log_line = {
                'timestamp': entry.timestamp,
                'user_id': entry.user_id,
                'operation': entry.operation,
                'resource': entry.resource,
                'success': entry.success,
                'security_level': entry.security_level.value,
                'details': entry.details
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_line) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def search_logs(self, user_id: Optional[str] = None, 
                   operation: Optional[str] = None,
                   time_range: Optional[Tuple[float, float]] = None) -> List[AuditLogEntry]:
        """Search audit logs with filters."""
        results = self.audit_entries.copy()
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if operation:
            results = [e for e in results if e.operation == operation]
        
        if time_range:
            start_time, end_time = time_range
            results = [e for e in results if start_time <= e.timestamp <= end_time]
        
        return results


class ThreatDetector:
    """Detects potential security threats and anomalies."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            'request_rate': 100,  # requests per minute
            'failure_rate': 0.1,  # 10% failure rate
            'access_pattern_deviation': 0.3
        }
        
        self.threat_indicators = {
            'brute_force': 0,
            'data_exfiltration': 0,
            'privilege_escalation': 0,
            'anomalous_access': 0
        }
        
        logger.info("Threat detection system initialized")
    
    def analyze_access_pattern(self, audit_entries: List[AuditLogEntry], 
                             user_id: str) -> Dict[str, float]:
        """Analyze user access patterns for anomalies."""
        user_entries = [e for e in audit_entries if e.user_id == user_id]
        
        if len(user_entries) < 10:
            return {'risk_score': 0.0, 'confidence': 0.0}
        
        # Calculate metrics
        recent_entries = [e for e in user_entries if time.time() - e.timestamp < 3600]  # Last hour
        
        request_rate = len(recent_entries) / max(1, len(user_entries))
        failure_rate = sum(1 for e in recent_entries if not e.success) / max(1, len(recent_entries))
        
        # Check for unusual resource access
        accessed_resources = set(e.resource for e in recent_entries)
        historical_resources = set(e.resource for e in user_entries[:-len(recent_entries)])
        
        new_resources_ratio = len(accessed_resources - historical_resources) / max(1, len(accessed_resources))
        
        # Calculate risk score
        risk_factors = {
            'high_request_rate': min(1.0, request_rate / 0.1),
            'high_failure_rate': min(1.0, failure_rate / 0.1),
            'unusual_resources': new_resources_ratio
        }
        
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        confidence = min(1.0, len(recent_entries) / 50.0)
        
        return {
            'risk_score': risk_score,
            'confidence': confidence,
            'risk_factors': risk_factors
        }
    
    def detect_data_exfiltration(self, audit_entries: List[AuditLogEntry]) -> List[Dict[str, Any]]:
        """Detect potential data exfiltration attempts."""
        alerts = []
        
        # Group by user
        user_activities = {}
        for entry in audit_entries:
            if entry.user_id not in user_activities:
                user_activities[entry.user_id] = []
            user_activities[entry.user_id].append(entry)
        
        for user_id, activities in user_activities.items():
            recent_activities = [a for a in activities if time.time() - a.timestamp < 1800]  # 30 minutes
            
            # Check for bulk data access
            read_operations = [a for a in recent_activities if 'read' in a.operation.lower()]
            
            if len(read_operations) > 50:  # More than 50 read operations in 30 minutes
                alerts.append({
                    'type': 'potential_data_exfiltration',
                    'user_id': user_id,
                    'severity': 'high',
                    'details': f'{len(read_operations)} read operations in 30 minutes',
                    'timestamp': time.time()
                })
        
        return alerts
    
    def update_threat_indicators(self, audit_entries: List[AuditLogEntry]):
        """Update threat indicators based on recent activity."""
        recent_entries = [e for e in audit_entries if time.time() - e.timestamp < 3600]
        
        # Brute force detection
        failed_logins = [e for e in recent_entries if 'login' in e.operation and not e.success]
        user_failed_attempts = {}
        
        for entry in failed_logins:
            user_failed_attempts[entry.user_id] = user_failed_attempts.get(entry.user_id, 0) + 1
        
        max_failed_attempts = max(user_failed_attempts.values()) if user_failed_attempts else 0
        self.threat_indicators['brute_force'] = min(1.0, max_failed_attempts / 10.0)
        
        # Data exfiltration indicator
        exfiltration_alerts = self.detect_data_exfiltration(recent_entries)
        self.threat_indicators['data_exfiltration'] = min(1.0, len(exfiltration_alerts) / 5.0)


class SecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_manager = EncryptionManager(encryption_key)
        self.access_control = AccessControl()
        self.audit_logger = AuditLogger()
        self.threat_detector = ThreatDetector()
        
        self._security_policies = {
            'require_encryption': True,
            'audit_all_operations': True,
            'enable_threat_detection': True,
            'session_timeout': 86400  # 24 hours
        }
        
        logger.info("Security manager initialized")
    
    def secure_operation(self, operation_name: str, required_access: AccessLevel, 
                        security_level: SecurityLevel = SecurityLevel.INTERNAL):
        """Decorator for securing HDC operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract session token from kwargs
                session_token = kwargs.pop('session_token', None)
                if not session_token:
                    raise SecurityError("Session token required")
                
                # Validate permissions
                if not self.access_control.check_permission(
                    session_token, operation_name, func.__name__, required_access
                ):
                    self.audit_logger.log_operation(
                        session_token, operation_name, func.__name__, 
                        False, security_level, {'reason': 'access_denied'}
                    )
                    raise SecurityError("Access denied")
                
                # Execute operation
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Log successful operation
                    self.audit_logger.log_operation(
                        session_token, operation_name, func.__name__,
                        True, security_level, {
                            'execution_time': end_time - start_time,
                            'args_count': len(args),
                            'kwargs_count': len(kwargs)
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log failed operation
                    self.audit_logger.log_operation(
                        session_token, operation_name, func.__name__,
                        False, security_level, {
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        }
                    )
                    raise
            
            return wrapper
        return decorator
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status."""
        recent_entries = self.audit_logger.search_logs(
            time_range=(time.time() - 3600, time.time())
        )
        
        self.threat_detector.update_threat_indicators(recent_entries)
        
        return {
            'active_sessions': len(self.access_control.active_sessions),
            'recent_operations': len(recent_entries),
            'threat_indicators': self.threat_detector.threat_indicators,
            'security_policies': self._security_policies,
            'system_health': 'healthy'  # Could be computed based on various factors
        }
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        status = self.get_security_status()
        
        report = "# Security Status Report\n\n"
        report += f"**Generated**: {time.ctime()}\n\n"
        
        report += "## System Status\n"
        report += f"- Active Sessions: {status['active_sessions']}\n"
        report += f"- Recent Operations: {status['recent_operations']}\n"
        report += f"- System Health: {status['system_health']}\n\n"
        
        report += "## Threat Indicators\n"
        for threat, level in status['threat_indicators'].items():
            risk_level = "HIGH" if level > 0.7 else "MEDIUM" if level > 0.3 else "LOW"
            report += f"- {threat}: {level:.3f} ({risk_level})\n"
        
        report += "\n## Security Policies\n"
        for policy, enabled in status['security_policies'].items():
            report += f"- {policy}: {'✅ Enabled' if enabled else '❌ Disabled'}\n"
        
        return report


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Global security manager instance
_global_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


# Example secured operations
def create_secure_session(user_id: str, role: str, security_clearance: SecurityLevel) -> str:
    """Create secure user session."""
    security_manager = get_security_manager()
    return security_manager.access_control.create_user_session(user_id, role, security_clearance)


@get_security_manager().secure_operation('hypervector_creation', AccessLevel.WRITE)
def secure_hypervector_creation(data: torch.Tensor, mode: str = "dense", **kwargs) -> HyperVector:
    """Securely create hypervector with access control."""
    return HyperVector(data, mode=mode)


@get_security_manager().secure_operation('hypervector_binding', AccessLevel.EXECUTE)
def secure_bind(hv1: HyperVector, hv2: HyperVector, **kwargs) -> HyperVector:
    """Securely bind hypervectors with access control."""
    from ..core.operations import bind
    return bind(hv1, hv2)
