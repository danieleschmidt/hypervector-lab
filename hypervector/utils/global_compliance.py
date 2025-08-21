"""Global compliance and regulatory framework for HDC systems.

Provides GDPR, CCPA, PDPA compliance and multi-region support
for global deployment of hyperdimensional computing systems.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California, USA
    PDPA = "pdpa"  # Singapore, Thailand
    PIPEDA = "pipeda"  # Canada
    LGPD = "lgpd"  # Brazil
    PRIVACY_ACT = "privacy_act"  # Australia


class DataClassification(Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    RESEARCH = "research"
    STATISTICS = "statistics"


@dataclass
class DataSubject:
    """Data subject information for compliance tracking."""
    subject_id: str
    region: str
    applicable_frameworks: List[ComplianceFramework]
    consent_given: bool
    consent_timestamp: float
    data_retention_period: int  # days
    processing_purposes: List[ProcessingPurpose]
    metadata: Dict[str, Any]


@dataclass
class ComplianceRecord:
    """Compliance audit record."""
    record_id: str
    timestamp: float
    framework: ComplianceFramework
    operation: str
    data_subject: Optional[str]
    data_classification: DataClassification
    processing_purpose: ProcessingPurpose
    compliance_status: bool
    details: Dict[str, Any]


class RegionalComplianceManager:
    """Manages regional compliance requirements."""
    
    def __init__(self):
        self.regional_requirements = {
            "EU": {
                "frameworks": [ComplianceFramework.GDPR],
                "data_retention_max": 2555,  # 7 years in days
                "consent_required": True,
                "right_to_erasure": True,
                "data_portability": True,
                "lawful_basis_required": True,
                "dpo_required": True  # Data Protection Officer
            },
            "US_CA": {
                "frameworks": [ComplianceFramework.CCPA],
                "data_retention_max": 1825,  # 5 years
                "consent_required": False,  # Opt-out model
                "right_to_delete": True,
                "right_to_know": True,
                "sale_disclosure": True,
                "non_discrimination": True
            },
            "SG": {
                "frameworks": [ComplianceFramework.PDPA],
                "consent_required": True,
                "notification_required": True,
                "data_protection_officer": True,
                "breach_notification_hours": 72
            },
            "CA": {
                "frameworks": [ComplianceFramework.PIPEDA],
                "consent_required": True,
                "data_minimization": True,
                "accuracy_requirement": True,
                "safeguards_required": True
            },
            "BR": {
                "frameworks": [ComplianceFramework.LGPD],
                "consent_required": True,
                "dpo_required": True,
                "data_protection_impact_assessment": True,
                "breach_notification_hours": 72
            },
            "AU": {
                "frameworks": [ComplianceFramework.PRIVACY_ACT],
                "consent_required": True,
                "data_breach_notification": True,
                "privacy_policy_required": True,
                "access_rights": True
            }
        }
        
        self.compliance_records: List[ComplianceRecord] = []
        logger.info("Regional compliance manager initialized")
    
    def get_regional_requirements(self, region: str) -> Dict[str, Any]:
        """Get compliance requirements for a specific region."""
        return self.regional_requirements.get(region, {})
    
    def validate_data_processing(self, region: str, data_classification: DataClassification,
                               processing_purpose: ProcessingPurpose,
                               data_subject: Optional[DataSubject] = None) -> bool:
        """Validate if data processing is compliant for the region."""
        requirements = self.get_regional_requirements(region)
        
        if not requirements:
            logger.warning(f"No compliance requirements defined for region: {region}")
            return False
        
        # Check consent requirements
        if requirements.get("consent_required", False):
            if not data_subject or not data_subject.consent_given:
                return False
        
        # Check data retention limits
        if data_subject and "data_retention_max" in requirements:
            days_since_consent = (time.time() - data_subject.consent_timestamp) / 86400
            if days_since_consent > requirements["data_retention_max"]:
                return False
        
        # Check if processing purpose is valid
        sensitive_data = data_classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL]
        if sensitive_data and processing_purpose not in [ProcessingPurpose.CONSENT, ProcessingPurpose.LEGAL_OBLIGATION]:
            if region == "EU" and processing_purpose not in [ProcessingPurpose.LEGITIMATE_INTERESTS]:
                return False
        
        return True
    
    def log_compliance_event(self, region: str, operation: str, 
                           data_classification: DataClassification,
                           processing_purpose: ProcessingPurpose,
                           data_subject_id: Optional[str] = None) -> str:
        """Log compliance event for audit trail."""
        frameworks = self.regional_requirements.get(region, {}).get("frameworks", [])
        
        for framework in frameworks:
            record_id = hashlib.sha256(f"{time.time()}_{region}_{operation}".encode()).hexdigest()[:16]
            
            compliance_status = self.validate_data_processing(
                region, data_classification, processing_purpose
            )
            
            record = ComplianceRecord(
                record_id=record_id,
                timestamp=time.time(),
                framework=framework,
                operation=operation,
                data_subject=data_subject_id,
                data_classification=data_classification,
                processing_purpose=processing_purpose,
                compliance_status=compliance_status,
                details={"region": region}
            )
            
            self.compliance_records.append(record)
            
            if not compliance_status:
                logger.warning(f"Compliance violation detected: {record_id}")
        
        return record_id


class DataAnonymizer:
    """Anonymizes and pseudonymizes data for compliance."""
    
    def __init__(self, anonymization_key: Optional[str] = None):
        self.anonymization_key = anonymization_key or "default_key_change_in_production"
        self.anonymization_mappings = {}
        
    def anonymize_identifier(self, identifier: str, method: str = "hash") -> str:
        """Anonymize personal identifier."""
        if method == "hash":
            # One-way hash anonymization
            combined = f"{identifier}_{self.anonymization_key}"
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        elif method == "pseudonymize":
            # Reversible pseudonymization
            if identifier in self.anonymization_mappings:
                return self.anonymization_mappings[identifier]
            
            pseudo_id = f"pseudo_{len(self.anonymization_mappings):06d}"
            self.anonymization_mappings[identifier] = pseudo_id
            return pseudo_id
        
        else:
            raise ValueError(f"Unknown anonymization method: {method}")
    
    def anonymize_hypervector_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize metadata in hypervectors."""
        anonymized = metadata.copy()
        
        # Anonymize common personal identifiers
        personal_fields = ['user_id', 'email', 'name', 'phone', 'address', 'ip_address']
        
        for field in personal_fields:
            if field in anonymized:
                anonymized[field] = self.anonymize_identifier(str(anonymized[field]))
        
        # Remove direct identifiers
        direct_identifiers = ['ssn', 'passport', 'driver_license', 'credit_card']
        for field in direct_identifiers:
            if field in anonymized:
                anonymized[field] = "[REDACTED]"
        
        return anonymized
    
    def generate_synthetic_data(self, original_data: List[Any], 
                              privacy_budget: float = 1.0) -> List[Any]:
        """Generate synthetic data for privacy-preserving analytics."""
        # Simplified synthetic data generation
        # In production, would use differential privacy techniques
        
        synthetic_data = []
        for item in original_data:
            if isinstance(item, (int, float)):
                # Add calibrated noise
                noise = (hash(str(item)) % 100 - 50) / 100.0 * privacy_budget
                synthetic_item = item + noise
            elif isinstance(item, str):
                # Generate synthetic string
                synthetic_item = self.anonymize_identifier(item)
            else:
                synthetic_item = item  # Keep as-is for complex types
            
            synthetic_data.append(synthetic_item)
        
        return synthetic_data


class ConsentManager:
    """Manages user consent for data processing."""
    
    def __init__(self):
        self.consent_records = {}
        self.consent_templates = {
            ComplianceFramework.GDPR: {
                "purposes": ["research", "analytics", "service_improvement"],
                "retention_period": 2555,  # 7 years
                "withdrawal_right": True,
                "explicit_consent": True
            },
            ComplianceFramework.CCPA: {
                "purposes": ["business_operations", "service_provision"],
                "opt_out_available": True,
                "sale_disclosure": True
            }
        }
        
    def request_consent(self, subject_id: str, region: str, 
                       purposes: List[ProcessingPurpose]) -> Dict[str, Any]:
        """Generate consent request based on regional requirements."""
        requirements = RegionalComplianceManager().get_regional_requirements(region)
        frameworks = requirements.get("frameworks", [])
        
        consent_request = {
            "subject_id": subject_id,
            "region": region,
            "frameworks": [f.value for f in frameworks],
            "purposes": [p.value for p in purposes],
            "timestamp": time.time(),
            "explicit_consent_required": requirements.get("consent_required", False),
            "withdrawal_available": True,
            "retention_period_days": requirements.get("data_retention_max", 365)
        }
        
        return consent_request
    
    def record_consent(self, subject_id: str, consent_given: bool, 
                      purposes: List[ProcessingPurpose],
                      region: str) -> DataSubject:
        """Record consent decision."""
        requirements = RegionalComplianceManager().get_regional_requirements(region)
        frameworks = [ComplianceFramework(f) for f in requirements.get("frameworks", [])]
        
        data_subject = DataSubject(
            subject_id=subject_id,
            region=region,
            applicable_frameworks=frameworks,
            consent_given=consent_given,
            consent_timestamp=time.time(),
            data_retention_period=requirements.get("data_retention_max", 365),
            processing_purposes=purposes,
            metadata={"consent_version": "1.0", "region": region}
        )
        
        self.consent_records[subject_id] = data_subject
        
        logger.info(f"Consent recorded for {subject_id}: {consent_given}")
        return data_subject
    
    def withdraw_consent(self, subject_id: str) -> bool:
        """Withdraw consent and trigger data deletion."""
        if subject_id in self.consent_records:
            self.consent_records[subject_id].consent_given = False
            logger.info(f"Consent withdrawn for {subject_id}")
            return True
        return False
    
    def get_consent_status(self, subject_id: str) -> Optional[DataSubject]:
        """Get current consent status for a subject."""
        return self.consent_records.get(subject_id)


class DataRetentionManager:
    """Manages data retention and deletion policies."""
    
    def __init__(self):
        self.retention_policies = {}
        self.deletion_queue = []
        
    def set_retention_policy(self, data_type: str, region: str, 
                           retention_days: int, auto_delete: bool = True):
        """Set data retention policy for a data type and region."""
        policy_key = f"{data_type}_{region}"
        self.retention_policies[policy_key] = {
            "retention_days": retention_days,
            "auto_delete": auto_delete,
            "created_at": time.time()
        }
        
        logger.info(f"Retention policy set: {policy_key} - {retention_days} days")
    
    def check_retention_compliance(self, data_created_at: float, 
                                 data_type: str, region: str) -> bool:
        """Check if data is within retention period."""
        policy_key = f"{data_type}_{region}"
        policy = self.retention_policies.get(policy_key)
        
        if not policy:
            # Default to 365 days if no policy set
            retention_days = 365
        else:
            retention_days = policy["retention_days"]
        
        age_days = (time.time() - data_created_at) / 86400
        return age_days <= retention_days
    
    def schedule_deletion(self, data_id: str, data_type: str, 
                         deletion_timestamp: float):
        """Schedule data for deletion."""
        deletion_item = {
            "data_id": data_id,
            "data_type": data_type,
            "deletion_timestamp": deletion_timestamp,
            "scheduled_at": time.time()
        }
        
        self.deletion_queue.append(deletion_item)
        logger.info(f"Scheduled deletion for {data_id} at {time.ctime(deletion_timestamp)}")
    
    def process_deletion_queue(self) -> List[str]:
        """Process pending deletions and return deleted item IDs."""
        current_time = time.time()
        deleted_items = []
        remaining_queue = []
        
        for item in self.deletion_queue:
            if current_time >= item["deletion_timestamp"]:
                # Simulate deletion
                deleted_items.append(item["data_id"])
                logger.info(f"Deleted data: {item['data_id']}")
            else:
                remaining_queue.append(item)
        
        self.deletion_queue = remaining_queue
        return deleted_items


class GlobalComplianceFramework:
    """Main framework coordinating all compliance components."""
    
    def __init__(self):
        self.regional_manager = RegionalComplianceManager()
        self.anonymizer = DataAnonymizer()
        self.consent_manager = ConsentManager()
        self.retention_manager = DataRetentionManager()
        
        # Initialize default retention policies
        self._initialize_default_policies()
        
        logger.info("Global compliance framework initialized")
    
    def _initialize_default_policies(self):
        """Initialize default data retention policies."""
        regions = ["EU", "US_CA", "SG", "CA", "BR", "AU"]
        data_types = ["hypervector", "metadata", "logs", "analytics"]
        
        for region in regions:
            requirements = self.regional_manager.get_regional_requirements(region)
            default_retention = requirements.get("data_retention_max", 365)
            
            for data_type in data_types:
                self.retention_manager.set_retention_policy(
                    data_type, region, default_retention
                )
    
    def process_data_with_compliance(self, data: Any, region: str, 
                                   subject_id: Optional[str] = None,
                                   data_classification: DataClassification = DataClassification.INTERNAL,
                                   processing_purpose: ProcessingPurpose = ProcessingPurpose.RESEARCH) -> Dict[str, Any]:
        """Process data with full compliance checking."""
        # Check compliance
        data_subject = None
        if subject_id:
            data_subject = self.consent_manager.get_consent_status(subject_id)
        
        is_compliant = self.regional_manager.validate_data_processing(
            region, data_classification, processing_purpose, data_subject
        )
        
        if not is_compliant:
            raise ComplianceViolationError(
                f"Data processing not compliant for region {region}"
            )
        
        # Log compliance event
        record_id = self.regional_manager.log_compliance_event(
            region, "data_processing", data_classification, processing_purpose, subject_id
        )
        
        # Process data based on classification
        processed_data = data
        
        if data_classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL]:
            # Anonymize if dealing with personal data
            if isinstance(data, dict):
                processed_data = self.anonymizer.anonymize_hypervector_metadata(data)
            elif hasattr(data, 'metadata'):
                data.metadata = self.anonymizer.anonymize_hypervector_metadata(
                    getattr(data, 'metadata', {})
                )
        
        return {
            "processed_data": processed_data,
            "compliance_record_id": record_id,
            "region": region,
            "compliant": is_compliant,
            "anonymized": data_classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL]
        }
    
    def handle_data_subject_request(self, request_type: str, subject_id: str, 
                                  region: str) -> Dict[str, Any]:
        """Handle data subject requests (access, deletion, portability)."""
        requirements = self.regional_manager.get_regional_requirements(region)
        
        if request_type == "access":
            # Right to access personal data
            data_subject = self.consent_manager.get_consent_status(subject_id)
            if data_subject:
                return {
                    "request_type": "access",
                    "subject_id": subject_id,
                    "data_found": True,
                    "data_summary": asdict(data_subject),
                    "compliance_records": [
                        asdict(r) for r in self.regional_manager.compliance_records 
                        if r.data_subject == subject_id
                    ]
                }
            else:
                return {"request_type": "access", "subject_id": subject_id, "data_found": False}
        
        elif request_type == "deletion" or request_type == "erasure":
            # Right to deletion/erasure
            if requirements.get("right_to_erasure", False) or requirements.get("right_to_delete", False):
                # Withdraw consent
                consent_withdrawn = self.consent_manager.withdraw_consent(subject_id)
                
                # Schedule deletion
                deletion_time = time.time() + 86400  # 24 hours from now
                self.retention_manager.schedule_deletion(
                    subject_id, "all_personal_data", deletion_time
                )
                
                return {
                    "request_type": "deletion",
                    "subject_id": subject_id,
                    "consent_withdrawn": consent_withdrawn,
                    "deletion_scheduled": deletion_time,
                    "status": "scheduled"
                }
            else:
                return {
                    "request_type": "deletion",
                    "subject_id": subject_id,
                    "status": "not_supported",
                    "reason": "Region does not support right to deletion"
                }
        
        elif request_type == "portability":
            # Right to data portability
            if requirements.get("data_portability", False):
                data_subject = self.consent_manager.get_consent_status(subject_id)
                if data_subject:
                    # Generate portable data package
                    portable_data = {
                        "subject_data": asdict(data_subject),
                        "export_format": "json",
                        "export_timestamp": time.time(),
                        "compliance_records": [
                            asdict(r) for r in self.regional_manager.compliance_records 
                            if r.data_subject == subject_id
                        ]
                    }
                    
                    return {
                        "request_type": "portability",
                        "subject_id": subject_id,
                        "portable_data": portable_data,
                        "status": "completed"
                    }
            
            return {
                "request_type": "portability",
                "subject_id": subject_id,
                "status": "not_supported"
            }
        
        else:
            return {
                "request_type": request_type,
                "subject_id": subject_id,
                "status": "unsupported_request_type"
            }
    
    def generate_compliance_report(self, region: Optional[str] = None) -> str:
        """Generate comprehensive compliance report."""
        report = "# Global Compliance Report\n\n"
        report += f"**Generated**: {time.ctime()}\n\n"
        
        # Filter records by region if specified
        records = self.regional_manager.compliance_records
        if region:
            records = [r for r in records if r.details.get("region") == region]
        
        # Compliance statistics
        total_records = len(records)
        compliant_records = sum(1 for r in records if r.compliance_status)
        compliance_rate = (compliant_records / total_records * 100) if total_records > 0 else 100
        
        report += f"## Summary\n"
        report += f"- **Total Compliance Records**: {total_records}\n"
        report += f"- **Compliant Operations**: {compliant_records}\n"
        report += f"- **Compliance Rate**: {compliance_rate:.1f}%\n"
        report += f"- **Active Consent Records**: {len(self.consent_manager.consent_records)}\n"
        report += f"- **Pending Deletions**: {len(self.retention_manager.deletion_queue)}\n\n"
        
        # Framework breakdown
        framework_stats = {}
        for record in records:
            framework = record.framework.value
            if framework not in framework_stats:
                framework_stats[framework] = {"total": 0, "compliant": 0}
            framework_stats[framework]["total"] += 1
            if record.compliance_status:
                framework_stats[framework]["compliant"] += 1
        
        report += "## Framework Compliance\n"
        for framework, stats in framework_stats.items():
            rate = (stats["compliant"] / stats["total"] * 100) if stats["total"] > 0 else 100
            report += f"- **{framework.upper()}**: {stats['compliant']}/{stats['total']} ({rate:.1f}%)\n"
        
        return report


class ComplianceViolationError(Exception):
    """Exception raised for compliance violations."""
    pass


# Global compliance framework instance
_global_compliance = None

def get_compliance_framework() -> GlobalComplianceFramework:
    """Get global compliance framework instance."""
    global _global_compliance
    if _global_compliance is None:
        _global_compliance = GlobalComplianceFramework()
    return _global_compliance
