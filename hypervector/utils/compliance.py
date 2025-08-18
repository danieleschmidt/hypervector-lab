"""Data privacy and regulatory compliance utilities for global deployment."""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"  # PII/PHI data


class RegulationLevel(Enum):
    """Regulatory compliance levels."""
    NONE = "none"
    GDPR = "gdpr"        # EU General Data Protection Regulation
    CCPA = "ccpa"        # California Consumer Privacy Act
    HIPAA = "hipaa"      # Health Insurance Portability and Accountability Act
    PIPEDA = "pipeda"    # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"        # Lei Geral de Proteção de Dados (Brazil)
    PDPA = "pdpa"        # Personal Data Protection Act (Singapore/Thailand)


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for audit trails."""
    timestamp: float
    operation: str
    data_type: str
    classification: DataClassification
    regulation_level: RegulationLevel
    user_id: Optional[str] = None
    purpose: Optional[str] = None
    retention_period: Optional[int] = None  # Days
    consent_given: bool = False
    processing_location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        record = asdict(self)
        record['classification'] = record['classification'].value
        record['regulation_level'] = record['regulation_level'].value
        return record


class ConsentManager:
    """Manage user consent for data processing under various regulations."""
    
    def __init__(self):
        """Initialize consent manager."""
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.consent_templates = self._load_consent_templates()
        
    def _load_consent_templates(self) -> Dict[str, Dict[str, str]]:
        """Load consent templates for different regulations."""
        return {
            'gdpr': {
                'en': "Do you consent to processing your data for {purpose}? You have the right to withdraw consent at any time.",
                'es': "¿Consiente el procesamiento de sus datos para {purpose}? Tiene derecho a retirar el consentimiento en cualquier momento.",
                'fr': "Consentez-vous au traitement de vos données pour {purpose}? Vous avez le droit de retirer votre consentement à tout moment.",
                'de': "Stimmen Sie der Verarbeitung Ihrer Daten für {purpose} zu? Sie haben das Recht, Ihre Einwilligung jederzeit zu widerrufen."
            },
            'ccpa': {
                'en': "We will process your data for {purpose}. You have the right to opt-out of this processing.",
                'es': "Procesaremos sus datos para {purpose}. Tiene derecho a optar por no participar en este procesamiento."
            },
            'hipaa': {
                'en': "Your health information will be used for {purpose}. This use is authorized under HIPAA regulations."
            }
        }
    
    def request_consent(
        self,
        user_id: str,
        purpose: str,
        regulation: RegulationLevel,
        data_types: List[str],
        language: str = 'en'
    ) -> str:
        """Generate consent request text for user.
        
        Args:
            user_id: Unique user identifier
            purpose: Purpose of data processing
            regulation: Applicable regulation
            data_types: Types of data to be processed
            language: Language for consent text
            
        Returns:
            Consent request text
        """
        template = self.consent_templates.get(regulation.value, {}).get(language, 
                   self.consent_templates['gdpr']['en'])
        
        consent_text = template.format(purpose=purpose)
        
        # Store consent request
        consent_id = hashlib.sha256(f"{user_id}_{purpose}_{time.time()}".encode()).hexdigest()[:16]
        
        self.consent_records[consent_id] = {
            'user_id': user_id,
            'purpose': purpose,
            'regulation': regulation.value,
            'data_types': data_types,
            'language': language,
            'request_time': time.time(),
            'consent_given': None,
            'consent_time': None
        }
        
        return consent_text
    
    def record_consent(
        self,
        consent_id: str,
        consent_given: bool,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Record user's consent decision.
        
        Args:
            consent_id: Consent request identifier
            consent_given: Whether consent was given
            additional_info: Additional information about consent
        """
        if consent_id in self.consent_records:
            self.consent_records[consent_id].update({
                'consent_given': consent_given,
                'consent_time': time.time(),
                'additional_info': additional_info or {}
            })
            
            logger.info(f"Consent recorded: {consent_id} = {consent_given}")
        else:
            logger.error(f"Unknown consent ID: {consent_id}")
    
    def withdraw_consent(self, user_id: str, purpose: str):
        """Allow user to withdraw consent."""
        for consent_id, record in self.consent_records.items():
            if record['user_id'] == user_id and record['purpose'] == purpose:
                record['consent_given'] = False
                record['withdrawal_time'] = time.time()
                logger.info(f"Consent withdrawn: {consent_id}")
    
    def get_user_consents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all consent records for a user."""
        return [
            record for record in self.consent_records.values()
            if record['user_id'] == user_id
        ]


class DataRetentionManager:
    """Manage data retention policies for compliance."""
    
    def __init__(self):
        """Initialize data retention manager."""
        self.retention_policies: Dict[str, Dict[str, Any]] = {
            'personal_data': {
                'gdpr': 30,      # 30 days after purpose ends
                'ccpa': 365,     # 1 year
                'hipaa': 2555,   # 7 years
                'default': 90
            },
            'research_data': {
                'gdpr': 1095,    # 3 years
                'ccpa': 1825,    # 5 years
                'default': 1095
            },
            'analytics_data': {
                'gdpr': 90,      # 3 months
                'ccpa': 365,     # 1 year
                'default': 180
            }
        }
        
        self.scheduled_deletions: Dict[str, datetime] = {}
    
    def get_retention_period(
        self,
        data_type: str,
        regulation: RegulationLevel
    ) -> int:
        """Get retention period in days for data type and regulation.
        
        Args:
            data_type: Type of data
            regulation: Applicable regulation
            
        Returns:
            Retention period in days
        """
        policy = self.retention_policies.get(data_type, {})
        
        return policy.get(regulation.value, policy.get('default', 365))
    
    def schedule_deletion(
        self,
        data_id: str,
        data_type: str,
        regulation: RegulationLevel,
        creation_date: Optional[datetime] = None
    ):
        """Schedule data for deletion according to retention policy.
        
        Args:
            data_id: Unique data identifier
            data_type: Type of data
            regulation: Applicable regulation
            creation_date: When data was created (defaults to now)
        """
        if creation_date is None:
            creation_date = datetime.now()
        
        retention_days = self.get_retention_period(data_type, regulation)
        deletion_date = creation_date + timedelta(days=retention_days)
        
        self.scheduled_deletions[data_id] = deletion_date
        
        logger.info(f"Scheduled deletion: {data_id} on {deletion_date}")
    
    def get_items_for_deletion(self) -> List[str]:
        """Get items that should be deleted now."""
        now = datetime.now()
        
        return [
            data_id for data_id, deletion_date in self.scheduled_deletions.items()
            if deletion_date <= now
        ]
    
    def extend_retention(self, data_id: str, additional_days: int):
        """Extend retention period for specific data."""
        if data_id in self.scheduled_deletions:
            current_date = self.scheduled_deletions[data_id]
            new_date = current_date + timedelta(days=additional_days)
            self.scheduled_deletions[data_id] = new_date
            
            logger.info(f"Extended retention: {data_id} until {new_date}")


class DataMinimizer:
    """Implement data minimization principles for compliance."""
    
    def __init__(self):
        """Initialize data minimizer."""
        self.anonymization_rules: Dict[str, Dict[str, Any]] = {}
        
    def minimize_personal_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Minimize personal data according to compliance requirements.
        
        Args:
            data: Input data containing personal information
            
        Returns:
            Minimized data with reduced personal information
        """
        minimized = data.copy()
        
        # Remove direct identifiers
        direct_identifiers = [
            'name', 'full_name', 'first_name', 'last_name',
            'email', 'phone', 'ssn', 'passport', 'id_number',
            'address', 'zipcode', 'ip_address'
        ]
        
        for field in direct_identifiers:
            if field in minimized:
                minimized[field] = self._anonymize_field(field, minimized[field])
        
        # Generalize quasi-identifiers
        if 'age' in minimized and isinstance(minimized['age'], int):
            minimized['age_range'] = self._generalize_age(minimized['age'])
            del minimized['age']
        
        if 'location' in minimized:
            minimized['region'] = self._generalize_location(minimized['location'])
            del minimized['location']
        
        return minimized
    
    def _anonymize_field(self, field_name: str, value: Any) -> str:
        """Anonymize a specific field."""
        if field_name in ['email']:
            # Hash email but preserve domain for analytics
            if '@' in str(value):
                local, domain = str(value).split('@', 1)
                hashed_local = hashlib.sha256(local.encode()).hexdigest()[:8]
                return f"{hashed_local}@{domain}"
        
        # Default: hash the value
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]
    
    def _generalize_age(self, age: int) -> str:
        """Generalize age to ranges."""
        if age < 18:
            return "under_18"
        elif age < 25:
            return "18_24"
        elif age < 35:
            return "25_34"
        elif age < 50:
            return "35_49"
        elif age < 65:
            return "50_64"
        else:
            return "65_plus"
    
    def _generalize_location(self, location: str) -> str:
        """Generalize location to broader regions."""
        # Simple implementation - in practice, use proper geolocation services
        if any(city in location.lower() for city in ['new york', 'los angeles', 'chicago']):
            return "US_major_city"
        elif any(country in location.lower() for country in ['united states', 'usa']):
            return "US"
        elif any(country in location.lower() for country in ['canada']):
            return "CA"
        elif any(country in location.lower() for country in ['uk', 'britain', 'england']):
            return "UK"
        else:
            return "other"


class ComplianceAuditor:
    """Audit system for compliance monitoring."""
    
    def __init__(self, audit_log_path: Optional[str] = None):
        """Initialize compliance auditor.
        
        Args:
            audit_log_path: Path to audit log file
        """
        self.processing_records: List[DataProcessingRecord] = []
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        
        if self.audit_log_path:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def record_processing(
        self,
        operation: str,
        data_type: str,
        classification: DataClassification,
        regulation_level: RegulationLevel,
        user_id: Optional[str] = None,
        purpose: Optional[str] = None,
        consent_given: bool = False,
        processing_location: Optional[str] = None
    ):
        """Record data processing activity.
        
        Args:
            operation: Type of operation performed
            data_type: Type of data processed
            classification: Data classification level
            regulation_level: Applicable regulation
            user_id: User identifier (if applicable)
            purpose: Purpose of processing
            consent_given: Whether consent was obtained
            processing_location: Where processing occurred
        """
        record = DataProcessingRecord(
            timestamp=time.time(),
            operation=operation,
            data_type=data_type,
            classification=classification,
            regulation_level=regulation_level,
            user_id=user_id,
            purpose=purpose,
            consent_given=consent_given,
            processing_location=processing_location
        )
        
        self.processing_records.append(record)
        
        # Write to audit log
        if self.audit_log_path:
            self._write_audit_log(record)
        
        logger.debug(f"Recorded processing: {operation} for {data_type}")
    
    def _write_audit_log(self, record: DataProcessingRecord):
        """Write audit record to log file."""
        try:
            with open(self.audit_log_path, 'a') as f:
                json.dump(record.to_dict(), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def generate_compliance_report(
        self,
        regulation: RegulationLevel,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report for specific regulation.
        
        Args:
            regulation: Regulation to report on
            start_date: Start of reporting period
            end_date: End of reporting period
            
        Returns:
            Compliance report
        """
        if start_date:
            start_timestamp = start_date.timestamp()
        else:
            start_timestamp = 0
            
        if end_date:
            end_timestamp = end_date.timestamp()
        else:
            end_timestamp = time.time()
        
        # Filter records
        filtered_records = [
            record for record in self.processing_records
            if (record.regulation_level == regulation and
                start_timestamp <= record.timestamp <= end_timestamp)
        ]
        
        # Generate report
        report = {
            'regulation': regulation.value,
            'period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            },
            'total_processing_activities': len(filtered_records),
            'activities_by_type': {},
            'activities_by_classification': {},
            'consent_compliance': {
                'total_requiring_consent': 0,
                'consent_obtained': 0,
                'consent_rate': 0.0
            },
            'data_subjects_affected': len(set(
                record.user_id for record in filtered_records 
                if record.user_id
            ))
        }
        
        # Analyze activities
        for record in filtered_records:
            # By type
            if record.data_type not in report['activities_by_type']:
                report['activities_by_type'][record.data_type] = 0
            report['activities_by_type'][record.data_type] += 1
            
            # By classification
            classification = record.classification.value
            if classification not in report['activities_by_classification']:
                report['activities_by_classification'][classification] = 0
            report['activities_by_classification'][classification] += 1
            
            # Consent analysis
            if record.classification == DataClassification.PERSONAL:
                report['consent_compliance']['total_requiring_consent'] += 1
                if record.consent_given:
                    report['consent_compliance']['consent_obtained'] += 1
        
        # Calculate consent rate
        total_consent_required = report['consent_compliance']['total_requiring_consent']
        if total_consent_required > 0:
            consent_obtained = report['consent_compliance']['consent_obtained']
            report['consent_compliance']['consent_rate'] = consent_obtained / total_consent_required
        
        return report
    
    def check_compliance_violations(self) -> List[Dict[str, Any]]:
        """Check for potential compliance violations.
        
        Returns:
            List of potential violations
        """
        violations = []
        
        # Check for personal data processing without consent
        for record in self.processing_records:
            if (record.classification == DataClassification.PERSONAL and
                record.regulation_level in [RegulationLevel.GDPR, RegulationLevel.CCPA] and
                not record.consent_given):
                
                violations.append({
                    'type': 'missing_consent',
                    'record': record.to_dict(),
                    'severity': 'high',
                    'description': f"Personal data processed without consent: {record.operation}"
                })
        
        # Check for data retention violations
        retention_manager = DataRetentionManager()
        items_for_deletion = retention_manager.get_items_for_deletion()
        
        if items_for_deletion:
            violations.append({
                'type': 'retention_violation',
                'items': items_for_deletion,
                'severity': 'medium',
                'description': f"Data items past retention period: {len(items_for_deletion)} items"
            })
        
        return violations


# Global compliance instances
_consent_manager = ConsentManager()
_retention_manager = DataRetentionManager()
_data_minimizer = DataMinimizer()
_compliance_auditor = ComplianceAuditor()

# Convenience functions
def request_consent(user_id: str, purpose: str, regulation: RegulationLevel, 
                   data_types: List[str], language: str = 'en') -> str:
    """Request user consent for data processing."""
    return _consent_manager.request_consent(user_id, purpose, regulation, data_types, language)

def record_consent(consent_id: str, consent_given: bool, additional_info: Optional[Dict[str, Any]] = None):
    """Record user consent decision."""
    _consent_manager.record_consent(consent_id, consent_given, additional_info)

def minimize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Minimize personal data for compliance."""
    return _data_minimizer.minimize_personal_data(data)

def record_processing(operation: str, data_type: str, classification: DataClassification,
                     regulation_level: RegulationLevel, **kwargs):
    """Record data processing activity."""
    _compliance_auditor.record_processing(operation, data_type, classification, regulation_level, **kwargs)

def generate_compliance_report(regulation: RegulationLevel, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """Generate compliance report."""
    return _compliance_auditor.generate_compliance_report(regulation, start_date, end_date)

def check_compliance() -> List[Dict[str, Any]]:
    """Check for compliance violations."""
    return _compliance_auditor.check_compliance_violations()

def schedule_data_deletion(data_id: str, data_type: str, regulation: RegulationLevel, 
                         creation_date: Optional[datetime] = None):
    """Schedule data for deletion."""
    _retention_manager.schedule_deletion(data_id, data_type, regulation, creation_date)