"""
Generate synthetic data for Phase 1 testing.

Produces ~15-20 documents across three content types with KNOWN,
VERIFIABLE FACTS so golden Q&A pairs have unambiguous ground-truth answers.

This is a one-off utility, not part of the runtime pipeline. It generates
test fixtures only — no client data is involved.

Usage:
    python scripts/generate_synthetic_data.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_requirement_docs(output_dir: Path) -> list[dict]:
    """Generate synthetic requirement specification documents as text files.

    Each doc contains specific numbers, dates, and names that serve as
    verifiable facts for golden Q&A pairs.
    """
    docs_dir = output_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "filename": "requirements_spec_v1.txt",
            "content": """# Project Aurora - Software Requirements Specification v1.0
## Document ID: REQ-AURORA-001
## Last Updated: 2024-03-15
## Author: Dr. Sarah Chen, Chief Architect

## 1. Overview

Project Aurora is a real-time data processing platform designed to handle up to 500,000 events per second. The system must achieve 99.95% uptime and maintain end-to-end latency below 150 milliseconds for 95th percentile requests.

## 2. Data Processing Module (REQ-DP)

### REQ-DP-001: Throughput
The data processing module shall support a minimum throughput of 500,000 events per second under standard load conditions.

### REQ-DP-002: Latency
The system shall maintain end-to-end processing latency below 150 milliseconds for the 95th percentile of all requests.

### REQ-DP-003: Memory Allocation
The maximum memory allocation for the data processing module is 4GB per processing node. Each node runs independently and does not share memory with other nodes.

### REQ-DP-004: Data Retention
Processed event data shall be retained for a minimum of 90 days in the primary storage tier and 365 days in the archive tier.

## 3. Authentication Module (REQ-AUTH)

### REQ-AUTH-001: Session Management
User sessions shall expire after 30 minutes of inactivity. Maximum concurrent sessions per user: 3.

### REQ-AUTH-002: Multi-Factor Authentication
MFA shall be mandatory for all administrator accounts. Supported methods: TOTP, WebAuthn, and SMS (SMS to be deprecated by Q4 2024).

### REQ-AUTH-003: Password Policy
Minimum password length: 12 characters. Must include uppercase, lowercase, numbers, and special characters. Password rotation required every 90 days.

## 4. API Gateway (REQ-API)

### REQ-API-001: Rate Limiting
Default rate limit: 1,000 requests per minute per API key. Premium tier: 10,000 requests per minute.

### REQ-API-002: Supported Protocols
The API gateway shall support REST (JSON), GraphQL, and gRPC. WebSocket connections shall be supported for real-time data streaming.

### REQ-API-003: Response Format
All API responses shall include: request_id, timestamp, status_code, and a data payload. Error responses shall additionally include error_code and error_message fields.

## 5. Non-Functional Requirements

### REQ-NF-001: Availability
Target availability: 99.95% measured on a rolling 30-day window.

### REQ-NF-002: Scalability
The system shall support horizontal scaling from 3 to 50 processing nodes without service interruption.

### REQ-NF-003: Disaster Recovery
Recovery Point Objective (RPO): 1 hour. Recovery Time Objective (RTO): 4 hours. Backup frequency: every 6 hours.
""",
        },
        {
            "filename": "requirements_spec_v2_analytics.txt",
            "content": """# Project Aurora - Analytics Module Requirements v2.0
## Document ID: REQ-ANALYTICS-002
## Last Updated: 2024-05-22
## Author: Marcus Rivera, Data Engineering Lead

## 1. Analytics Engine

### REQ-AN-001: Real-Time Dashboard
The analytics engine shall support real-time dashboards with a maximum refresh interval of 5 seconds. Dashboard data shall reflect events processed within the last 10 seconds.

### REQ-AN-002: Historical Analysis
The system shall support historical analysis queries spanning up to 2 years of data. Query execution time for historical reports shall not exceed 30 seconds for standard reports and 5 minutes for complex aggregations.

### REQ-AN-003: Export Formats
Analytics data shall be exportable in CSV, JSON, Parquet, and PDF formats. Maximum export size: 10 million rows per export operation.

## 2. Machine Learning Pipeline

### REQ-ML-001: Model Training
The ML pipeline shall support training models on datasets up to 100GB. Training jobs shall be scheduled and run during off-peak hours (2:00 AM - 6:00 AM UTC).

### REQ-ML-002: Model Serving
Trained models shall be served with an inference latency of less than 50 milliseconds (p99). The system shall support A/B testing with configurable traffic splits.

### REQ-ML-003: Model Versioning
All models shall be versioned using semantic versioning. The system shall retain the last 10 versions of each model and allow instant rollback to any retained version.

## 3. Data Quality

### REQ-DQ-001: Validation Rules
The data quality module shall validate incoming data against configurable schemas. Invalid records shall be quarantined and flagged for review. Validation throughput: minimum 100,000 records per second.

### REQ-DQ-002: Data Lineage
The system shall maintain complete data lineage from ingestion to final output. Lineage records shall be queryable and retained for 1 year.

### REQ-DQ-003: Anomaly Detection
Automated anomaly detection shall alert operators when data patterns deviate by more than 3 standard deviations from the 30-day rolling average.
""",
        },
        {
            "filename": "security_compliance_spec.txt",
            "content": """# Project Aurora - Security & Compliance Requirements
## Document ID: REQ-SEC-003
## Last Updated: 2024-06-10
## Author: Priya Patel, Security Architect

## 1. Data Encryption

### REQ-SEC-001: Encryption at Rest
All data at rest shall be encrypted using AES-256-GCM. Encryption keys shall be managed through AWS KMS with automatic rotation every 365 days.

### REQ-SEC-002: Encryption in Transit
All data in transit shall be encrypted using TLS 1.3. Cipher suites below TLS 1.2 shall be disabled. Certificate pinning shall be implemented for all internal service-to-service communication.

### REQ-SEC-003: Key Management
Encryption key hierarchy: Master Key → Data Encryption Keys (DEKs). DEKs shall be rotated every 30 days. Master key rotation: annually. All key operations shall be logged to an immutable audit trail.

## 2. Access Control

### REQ-AC-001: Role-Based Access
The system shall implement RBAC with the following roles: Viewer, Analyst, Developer, Admin, SuperAdmin. Each role shall have explicitly defined permissions. Role assignments shall require approval from at least one Admin.

### REQ-AC-002: Audit Logging
All access events shall be logged with: user_id, timestamp, resource_accessed, action_performed, source_ip, and result (success/failure). Audit logs shall be retained for 2 years and be immutable.

### REQ-AC-003: Network Security
Internal services shall communicate only through private subnets. External API endpoints shall be protected by WAF (Web Application Firewall) with OWASP Top 10 rule sets. DDoS protection shall handle up to 10 Gbps of attack traffic.

## 3. Compliance

### REQ-COMP-001: SOC 2 Type II
The system shall maintain SOC 2 Type II compliance. Annual audits shall be conducted by an independent third-party auditor. Audit reports shall be available to customers upon request.

### REQ-COMP-002: GDPR
Personal data processing shall comply with GDPR Articles 5-9. Data Subject Access Requests (DSARs) shall be fulfilled within 72 hours. The system shall support the right to erasure with complete data deletion within 30 days.

### REQ-COMP-003: Data Residency
Customer data shall be stored in the geographic region specified by the customer. Supported regions: US-East, US-West, EU-West, EU-Central, APAC-Southeast. Cross-region data transfer requires explicit customer consent.
""",
        },
        {
            "filename": "infrastructure_spec.txt",
            "content": """# Project Aurora - Infrastructure Requirements
## Document ID: REQ-INFRA-004
## Last Updated: 2024-04-28
## Author: James Okafor, DevOps Lead

## 1. Compute Resources

### REQ-INFRA-001: Production Cluster
Production cluster minimum configuration: 12 compute nodes, each with 32 vCPUs, 128GB RAM, and 500GB NVMe SSD. Node auto-scaling shall activate when average CPU utilization exceeds 70% for 5 consecutive minutes.

### REQ-INFRA-002: Development Environment
Development environments shall mirror production architecture at 25% scale (3 compute nodes minimum). Each developer shall have access to a personal sandbox environment.

### REQ-INFRA-003: CI/CD Pipeline
Build pipeline execution time shall not exceed 15 minutes for a full build including unit tests. Deployment to staging: automated. Deployment to production: requires approval from 2 senior engineers.

## 2. Database Infrastructure

### REQ-DB-001: Primary Database
PostgreSQL 16 with TimescaleDB extension for time-series data. Minimum storage: 5TB with automatic expansion. Read replicas: minimum 2 per region.

### REQ-DB-002: Cache Layer
Redis cluster with minimum 64GB memory across the cluster. Cache hit rate target: 95%. TTL for session data: 30 minutes. TTL for query cache: 5 minutes.

### REQ-DB-003: Message Queue
Apache Kafka with minimum 6 brokers. Message retention: 7 days. Partition count per topic: 12 minimum. Replication factor: 3.

## 3. Monitoring & Observability

### REQ-MON-001: Metrics Collection
Prometheus-based metrics collection with 15-second scrape interval. Metrics retention: 90 days at full resolution, 1 year at 5-minute aggregation.

### REQ-MON-002: Alerting
PagerDuty integration for critical alerts. Alert response time SLA: acknowledge within 5 minutes, resolve within 1 hour for P1 incidents.

### REQ-MON-003: Logging
Centralized logging via ELK stack. Log retention: 30 days hot storage, 90 days warm storage, 1 year cold storage. Structured JSON logging required for all services.
""",
        },
        {
            "filename": "api_design_spec.txt",
            "content": """# Project Aurora - API Design Specification
## Document ID: REQ-APIDESIGN-005
## Last Updated: 2024-07-01
## Author: Elena Vasquez, API Platform Lead

## 1. Versioning Strategy

### REQ-APID-001: API Versioning
APIs shall use URL path versioning (e.g., /v1/, /v2/). A maximum of 3 API versions shall be supported simultaneously. Deprecated versions shall receive a 6-month sunset notice before removal.

### REQ-APID-002: Backward Compatibility
Minor version updates shall maintain backward compatibility. Breaking changes require a major version bump. All breaking changes shall be documented in a CHANGELOG with migration guides.

## 2. Pagination & Filtering

### REQ-APID-003: Pagination
List endpoints shall support cursor-based pagination with a default page size of 50 items and a maximum of 200 items per page. Responses shall include next_cursor and has_more fields.

### REQ-APID-004: Filtering
All list endpoints shall support filtering by: created_at (range), updated_at (range), and status. Custom filters shall be supported via query parameters.

## 3. Error Handling

### REQ-APID-005: Error Response Format
All error responses shall follow RFC 7807 (Problem Details). Each error shall include: type (URI), title, status (HTTP code), detail (human-readable), and instance (request trace ID).

### REQ-APID-006: Rate Limit Headers
All responses shall include rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset (Unix timestamp). Rate-limited responses shall return HTTP 429.

## 4. Webhooks

### REQ-APID-007: Webhook Delivery
Webhooks shall be delivered within 30 seconds of the triggering event. Failed deliveries shall be retried with exponential backoff: 1 min, 5 min, 15 min, 1 hour, 6 hours. Maximum retries: 5.

### REQ-APID-008: Webhook Security
All webhook payloads shall be signed using HMAC-SHA256. The signing key shall be unique per webhook subscription. Webhook endpoints shall validate the signature before processing.
""",
        },
    ]

    generated = []
    for spec in specs:
        filepath = docs_dir / spec["filename"]
        filepath.write_text(spec["content"], encoding="utf-8")
        generated.append(
            {"type": "document", "filename": spec["filename"], "path": str(filepath)}
        )
        print(f"  ✅ {spec['filename']}")

    return generated


def generate_chat_exports(output_dir: Path) -> list[dict]:
    """Generate synthetic Slack/Teams-style chat exports."""
    chat_dir = output_dir / "chat_exports"
    chat_dir.mkdir(parents=True, exist_ok=True)

    chats = [
        {
            "filename": "sprint_planning_chat.txt",
            "content": """# Sprint Planning Discussion - Sprint 42
## Channel: #aurora-engineering
## Date: 2024-03-20

[09:15] Sarah Chen: Good morning team! Let's kick off Sprint 42 planning. Our velocity last sprint was 34 story points.

[09:16] Marcus Rivera: Morning! I've got the analytics dashboard backlog prioritized. The real-time refresh feature (AN-001) is the top priority - stakeholders want the 5-second refresh interval implemented this sprint.

[09:17] James Okafor: DevOps side - we need to address the auto-scaling issue. Currently our nodes scale when CPU hits 80%, but the requirement says 70%. That's a config change plus testing.

[09:18] Priya Patel: Security review for the Q2 release is due April 5th. I need the encryption key rotation automation (SEC-003) completed before then. The 30-day DEK rotation is the critical path.

[09:20] Elena Vasquez: API team update - we've finished the cursor-based pagination (APID-003). Default page size is set to 50, max 200. Ready for QA.

[09:22] Sarah Chen: Great progress, Elena. @Marcus, what's the estimated effort for the real-time dashboard?

[09:23] Marcus Rivera: I'd say 8 story points. The main challenge is ensuring we hit that 10-second data freshness requirement. We need to optimize the event aggregation pipeline.

[09:25] James Okafor: I can help with the infrastructure side. We should also increase the Redis cache from 48GB to the required 64GB minimum this sprint.

[09:26] Sarah Chen: Good call. Let's add that. @Priya, how many points for the key rotation?

[09:27] Priya Patel: 5 points for the automation, 3 points for the audit trail integration. The immutable audit log requirement (AC-002) means we need to use append-only storage. I'm recommending Amazon QLDB for this.

[09:30] Sarah Chen: That brings us to about 30 points estimated. We have capacity for 34 based on velocity. Let's add the webhook signature validation (APID-008) if we have bandwidth. HMAC-SHA256 signing shouldn't be more than 3 points.

[09:31] Elena Vasquez: Agreed. I'll pick that up. The unique signing key per subscription is straightforward.

[09:33] Sarah Chen: Perfect. Sprint 42 backlog is set. Stand-up remains at 9:30 AM UTC daily. Sprint review on April 3rd. Let's ship it! 🚀
""",
        },
        {
            "filename": "incident_response_chat.txt",
            "content": """# Incident Response - Database Performance Degradation
## Channel: #aurora-incidents
## Severity: P2
## Date: 2024-04-12

[14:22] ALERT BOT: 🔴 P2 Incident Detected: Database query latency exceeding 500ms threshold. Current p95: 1,200ms. Affected service: analytics-api.

[14:23] James Okafor: Acknowledged. I'm looking at the Grafana dashboards now. The PostgreSQL primary is showing 95% CPU utilization.

[14:25] Marcus Rivera: I see the impact on the analytics dashboard. Real-time refresh is stalling. Users are reporting dashboard freezes.

[14:26] James Okafor: Root cause identified: a long-running analytical query is hogging resources. It's scanning the entire events table - 2.3 billion rows without using the TimescaleDB hypertable index.

[14:28] Sarah Chen: @Marcus, is this from the new historical analysis feature?

[14:29] Marcus Rivera: Yes, it's the 2-year lookback query. We haven't optimized it yet for the TimescaleDB time-series partitioning. The requirement (AN-002) says 30 seconds for standard reports, but this complex aggregation is currently taking 12 minutes.

[14:31] James Okafor: Immediate mitigation: I'm killing the runaway query and adding a query timeout of 5 minutes at the database level. This matches the complex aggregation limit in REQ-AN-002.

[14:32] Priya Patel: I've confirmed there's no data security impact. The query is read-only and running under the Analyst role permissions, which is correct per REQ-AC-001.

[14:35] James Okafor: Query terminated. Database CPU back to 45%. Query latency recovering - p95 now at 180ms and falling.

[14:36] Marcus Rivera: Dashboard refresh is working again. I'll create a ticket to optimize the historical query with proper TimescaleDB continuous aggregates.

[14:40] Sarah Chen: Good response, team. Total incident duration: 18 minutes. Within our 1-hour P2 resolution SLA. @James, please document this in the incident log. We need to add a query cost estimator before allowing ad-hoc historical queries.

[14:42] James Okafor: Noted. I'll also set up a Prometheus alert for queries exceeding 60 seconds. Current scrape interval is 15 seconds (per REQ-MON-001), so we'll catch these faster next time.

[14:45] ALERT BOT: ✅ P2 Incident Resolved. Database query latency normalized. p95: 120ms.
""",
        },
        {
            "filename": "architecture_review_chat.txt",
            "content": """# Architecture Review - Message Queue Migration
## Channel: #aurora-architecture
## Date: 2024-05-08

[10:00] Sarah Chen: Team, let's discuss the Kafka cluster capacity planning. Current state vs. requirements review.

[10:02] James Okafor: Current Kafka setup: 4 brokers, 8 partitions per topic, replication factor 2. Requirements (REQ-DB-003) specify: minimum 6 brokers, 12 partitions per topic, replication factor 3.

[10:04] Marcus Rivera: The partition count matters for our analytics pipeline. With 8 partitions, we're maxing out at 320,000 events/second. We need 12 partitions minimum to hit the 500,000 events/second requirement (REQ-DP-001).

[10:06] James Okafor: I've spec'd out the migration plan. We need to add 2 brokers and rebalance. The 7-day message retention is already configured correctly.

[10:08] Priya Patel: Security concern: when we add brokers, we need to ensure TLS 1.3 is configured on the new nodes. REQ-SEC-002 is explicit - no cipher suites below TLS 1.2.

[10:10] Elena Vasquez: API impact question - our webhook delivery system (APID-007) uses Kafka for the retry queue. The 30-second delivery SLA depends on consumer lag being under 5 seconds. Will the rebalance cause a gap?

[10:12] James Okafor: Good question. I plan to do a rolling migration - no downtime. Consumer lag should stay under 2 seconds during the migration. But I'll schedule it for the maintenance window: 2 AM - 6 AM UTC, same as our ML training window (REQ-ML-001).

[10:15] Sarah Chen: That's a shared window. @Marcus, will there be ML training jobs running?

[10:16] Marcus Rivera: Not this week. The next model training run is scheduled for May 15th. Current dataset is 78GB, well within the 100GB limit (REQ-ML-001). We have a clear window.

[10:18] Sarah Chen: Approved. @James, please document the migration runbook. Include rollback steps - if anything goes wrong, we need to be within our 4-hour RTO (REQ-NF-003).

[10:20] James Okafor: Will do. I'll also pre-stage the backups. Current backup frequency is every 6 hours per REQ-NF-003. I'll trigger a manual backup right before the migration starts.

[10:22] Sarah Chen: Perfect. Migration date: May 12th, 2 AM UTC. Post-migration validation: verify 500K events/second throughput and all consumer groups healthy.
""",
        },
        {
            "filename": "deployment_checklist_chat.txt",
            "content": """# Production Deployment - Release 2.4.0
## Channel: #aurora-releases
## Date: 2024-06-15

[11:00] Sarah Chen: Release 2.4.0 deployment checklist. This includes the analytics dashboard, key rotation automation, and API pagination. Go/no-go?

[11:02] Elena Vasquez: API team GO. Pagination (APID-003) tested with cursor-based approach. Verified default 50 items, max 200 items per page. Rate limit headers (APID-006) confirmed: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset all present.

[11:03] Marcus Rivera: Analytics team GO. Real-time dashboard refresh at 5-second intervals verified. Data freshness within 10 seconds confirmed under load. Export supports CSV, JSON, Parquet, and PDF up to 10 million rows (REQ-AN-003).

[11:05] Priya Patel: Security team GO. DEK rotation every 30 days automated and tested. Audit log immutability verified - using append-only storage. SOC 2 Type II evidence collection updated. One note: SMS MFA deprecation is scheduled for Q4 2024 per REQ-AUTH-002, not included in this release.

[11:07] James Okafor: Infrastructure team GO. CI/CD pipeline running in 13 minutes - within the 15-minute limit (REQ-INFRA-003). Redis upgraded to 64GB cluster. Auto-scaling threshold corrected to 70% CPU. Two senior engineer approvals obtained for production deployment.

[11:09] Sarah Chen: All teams GO. Deploying to staging first. @James, please initiate.

[11:10] James Okafor: Staging deployment initiated. Pipeline: build → test → deploy. Expected completion: 13 minutes.

[11:23] James Okafor: Staging deployment complete. All health checks passing. Staging verification:
- API response time: 45ms average (well under 150ms SLA)
- Cache hit rate: 96.2% (target: 95%)
- Event throughput: 520,000/sec on staging cluster
- All 47 integration tests passing

[11:25] Sarah Chen: Staging looks great. Proceeding to production. @James and @Marcus, you have the approval.

[11:26] James Okafor: Production deployment initiated.

[11:40] James Okafor: Production deployment complete. ✅ All nodes healthy. Zero-downtime deployment confirmed. Monitoring for 30 minutes before closing.

[12:10] James Okafor: 30-minute monitoring complete. All metrics nominal. Release 2.4.0 is live. 🎉

[12:12] Sarah Chen: Excellent work, everyone! Release 2.4.0 successfully deployed. Sprint review document updated. Next sprint planning: Monday 9:15 AM UTC.
""",
        },
    ]

    generated = []
    for chat in chats:
        filepath = chat_dir / chat["filename"]
        filepath.write_text(chat["content"], encoding="utf-8")
        generated.append(
            {"type": "chat_export", "filename": chat["filename"], "path": str(filepath)}
        )
        print(f"  ✅ {chat['filename']}")

    return generated


def generate_reference_articles(output_dir: Path) -> list[dict]:
    """Generate reference articles (simulating scraped web content)."""
    docs_dir = output_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    articles = [
        {
            "filename": "agile_testing_practices.txt",
            "content": """# Modern Agile Testing Practices for Distributed Systems

## Introduction

Testing distributed systems requires a fundamentally different approach than testing monolithic applications. The shift toward microservices architecture has introduced challenges in areas such as service isolation, network reliability, and data consistency that traditional testing frameworks were not designed to address.

## The Testing Pyramid for Microservices

The traditional testing pyramid (unit → integration → end-to-end) remains relevant but requires adaptation:

### Unit Tests
Unit tests should constitute approximately 70% of your test suite. For microservices, unit tests must mock all external dependencies including database calls, message queue interactions, and inter-service HTTP calls. Target execution time: under 5 minutes for the entire unit test suite.

### Integration Tests
Integration tests verify the contract between services. Consumer-Driven Contract Testing (using tools like Pact or Spring Cloud Contract) has become the standard approach. Each service publishes a contract, and consumer services verify compatibility. This approach catches breaking API changes before deployment.

### End-to-End Tests
End-to-end tests should be limited to 5-10 critical user journeys. These tests are expensive to maintain and slow to execute. Run them in a dedicated staging environment that mirrors production at reduced scale.

## Chaos Engineering

Chaos engineering practices, pioneered by Netflix's Chaos Monkey, should be integrated into your testing strategy by the third quarter of adopting a microservices architecture. Key experiments include:

- **Network partition simulation**: Verify that services gracefully degrade when upstream dependencies become unreachable.
- **Latency injection**: Ensure timeout configurations are correct by artificially adding 500ms-2000ms delays.
- **Resource exhaustion**: Test behavior when CPU, memory, or disk resources are constrained.

## Performance Testing

Load testing should simulate realistic traffic patterns, not just peak load. Use tools like k6 or Gatling to model:

1. **Steady state**: Normal traffic patterns over 24 hours
2. **Spike testing**: 5x normal traffic for 15 minutes
3. **Soak testing**: Sustained high load for 72 hours to detect memory leaks
4. **Breakpoint testing**: Gradually increase load until the system fails to identify capacity limits

## Observability-Driven Testing

Modern testing increasingly relies on observability rather than assertions. Instead of asserting specific values, monitor system behavior through distributed tracing (using OpenTelemetry), structured logging, and metrics dashboards. This approach catches unexpected behavior that traditional assertions might miss.

## Conclusion

The key principle for testing distributed systems is: design for failure. Every test should assume that any component can fail at any time, and the system's behavior under failure should be predictable, documented, and validated.
""",
        },
        {
            "filename": "data_governance_framework.txt",
            "content": """# Enterprise Data Governance Framework - Best Practices

## Overview

Data governance is the collection of practices and processes that ensure formal management of data assets within an organization. A mature data governance framework addresses data quality, data security, data lifecycle management, and regulatory compliance.

## Core Pillars

### 1. Data Quality Management

Data quality is measured across six dimensions:
- **Accuracy**: Data correctly represents the real-world entity. Target: 99.5% accuracy for critical data elements.
- **Completeness**: All required data fields are populated. Target: 98% completeness.
- **Consistency**: Data is uniform across all systems and databases. Inconsistencies should be less than 0.1% across source systems.
- **Timeliness**: Data is available when needed. For real-time systems, data latency should be under 60 seconds.
- **Validity**: Data conforms to defined business rules and formats.
- **Uniqueness**: No duplicate records exist. Deduplication processes should run daily.

### 2. Data Classification

Organizations should implement a four-tier data classification system:
- **Public**: Data that can be freely shared (press releases, marketing materials)
- **Internal**: Data for internal use only (org charts, internal memos)
- **Confidential**: Data with limited access (financial reports, HR records)
- **Restricted**: Highest sensitivity data (PII, payment card data, trade secrets)

Each classification tier has specific handling requirements for storage, transmission, access control, and disposal.

### 3. Data Lifecycle Management

Data moves through five lifecycle stages:
1. **Creation/Collection**: Data is generated or acquired
2. **Storage**: Data is persisted in appropriate systems
3. **Usage**: Data is accessed and processed
4. **Archival**: Data is moved to long-term storage
5. **Disposal**: Data is securely deleted

Retention policies should be defined per data classification and regulatory requirement. Financial data typically requires 7-year retention; healthcare data may require lifetime retention.

### 4. Metadata Management

A centralized metadata repository (data catalog) should document:
- Technical metadata: schema definitions, data types, relationships
- Business metadata: definitions, ownership, stewardship
- Operational metadata: data lineage, transformation history, quality scores

### 5. Regulatory Compliance

Key regulations that impact data governance:
- **GDPR** (EU): Right to access, right to erasure, data portability. Max fine: €20 million or 4% global turnover.
- **CCPA/CPRA** (California): Consumer rights to know, delete, opt-out. Max fine: $7,500 per intentional violation.
- **HIPAA** (US Healthcare): Protected Health Information safeguards. Max fine: $1.5 million per violation category per year.
- **SOX** (US Financial): Financial data integrity controls. Criminal penalties for non-compliance.

## Implementation Roadmap

A typical data governance implementation follows a 12-month phased approach:

**Months 1-3**: Assessment and strategy
**Months 4-6**: Framework design and pilot (2-3 data domains)
**Months 7-9**: Rollout to remaining data domains
**Months 10-12**: Automation, monitoring, and continuous improvement

## Key Metrics

Track the maturity of your data governance program using:
- Data Quality Score (DQS): Weighted average across all six quality dimensions
- Policy Compliance Rate: Percentage of data assets with proper classification and handling
- Time to Resolve Data Issues: Average time from issue detection to resolution
- Stakeholder Satisfaction: Annual survey of data consumers
""",
        },
    ]

    generated = []
    for article in articles:
        filepath = docs_dir / article["filename"]
        filepath.write_text(article["content"], encoding="utf-8")
        generated.append(
            {
                "type": "reference_article",
                "filename": article["filename"],
                "path": str(filepath),
            }
        )
        print(f"  ✅ {article['filename']}")

    return generated


def generate_golden_qa(output_dir: Path) -> None:
    """Generate the golden Q&A set based on the synthetic documents.

    All questions have unambiguous, verifiable answers from the seed content.
    """
    golden_dir = output_dir.parent / "golden_qa"
    golden_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        # From requirements_spec_v1
        {"question": "What is the maximum memory allocation for the data processing module?", "ground_truth": "The maximum memory allocation for the data processing module is 4GB per processing node, as specified in REQ-DP-003.", "source_doc": "requirements_spec_v1.txt", "category": "requirements"},
        {"question": "How long should user sessions last before expiring?", "ground_truth": "User sessions shall expire after 30 minutes of inactivity, with a maximum of 3 concurrent sessions per user, as specified in REQ-AUTH-001.", "source_doc": "requirements_spec_v1.txt", "category": "requirements"},
        {"question": "What is the target system availability?", "ground_truth": "The target availability is 99.95% measured on a rolling 30-day window, as specified in REQ-NF-001.", "source_doc": "requirements_spec_v1.txt", "category": "requirements"},
        {"question": "What is the Recovery Time Objective for disaster recovery?", "ground_truth": "The Recovery Time Objective (RTO) is 4 hours, with a Recovery Point Objective (RPO) of 1 hour and backup frequency of every 6 hours, as specified in REQ-NF-003.", "source_doc": "requirements_spec_v1.txt", "category": "requirements"},
        {"question": "What is the default API rate limit per API key?", "ground_truth": "The default rate limit is 1,000 requests per minute per API key, with a premium tier at 10,000 requests per minute, as specified in REQ-API-001.", "source_doc": "requirements_spec_v1.txt", "category": "requirements"},
        {"question": "What is the minimum password length required?", "ground_truth": "The minimum password length is 12 characters, and must include uppercase, lowercase, numbers, and special characters, with rotation required every 90 days, as specified in REQ-AUTH-003.", "source_doc": "requirements_spec_v1.txt", "category": "requirements"},
        # From requirements_spec_v2_analytics
        {"question": "What is the maximum refresh interval for real-time dashboards?", "ground_truth": "The maximum refresh interval for real-time dashboards is 5 seconds, with dashboard data reflecting events processed within the last 10 seconds, as specified in REQ-AN-001.", "source_doc": "requirements_spec_v2_analytics.txt", "category": "requirements"},
        {"question": "How many rows can be exported at once from the analytics system?", "ground_truth": "The maximum export size is 10 million rows per export operation, with support for CSV, JSON, Parquet, and PDF formats, as specified in REQ-AN-003.", "source_doc": "requirements_spec_v2_analytics.txt", "category": "requirements"},
        {"question": "What is the maximum dataset size for ML model training?", "ground_truth": "The ML pipeline supports training models on datasets up to 100GB, with training jobs scheduled during off-peak hours (2:00 AM - 6:00 AM UTC), as specified in REQ-ML-001.", "source_doc": "requirements_spec_v2_analytics.txt", "category": "requirements"},
        {"question": "How many model versions are retained for rollback?", "ground_truth": "The system retains the last 10 versions of each model and allows instant rollback to any retained version, using semantic versioning, as specified in REQ-ML-003.", "source_doc": "requirements_spec_v2_analytics.txt", "category": "requirements"},
        # From security_compliance_spec
        {"question": "What encryption algorithm is used for data at rest?", "ground_truth": "All data at rest is encrypted using AES-256-GCM, with encryption keys managed through AWS KMS with automatic rotation every 365 days, as specified in REQ-SEC-001.", "source_doc": "security_compliance_spec.txt", "category": "security"},
        {"question": "How often are Data Encryption Keys rotated?", "ground_truth": "Data Encryption Keys (DEKs) are rotated every 30 days, while the master key is rotated annually, as specified in REQ-SEC-003.", "source_doc": "security_compliance_spec.txt", "category": "security"},
        {"question": "How long are audit logs retained?", "ground_truth": "Audit logs are retained for 2 years and must be immutable, logging user_id, timestamp, resource_accessed, action_performed, source_ip, and result, as specified in REQ-AC-002.", "source_doc": "security_compliance_spec.txt", "category": "security"},
        {"question": "What is the timeframe for fulfilling GDPR Data Subject Access Requests?", "ground_truth": "Data Subject Access Requests (DSARs) shall be fulfilled within 72 hours, and the system shall support the right to erasure with complete data deletion within 30 days, as specified in REQ-COMP-002.", "source_doc": "security_compliance_spec.txt", "category": "compliance"},
        # From infrastructure_spec
        {"question": "What is the minimum number of compute nodes for the production cluster?", "ground_truth": "The production cluster minimum is 12 compute nodes, each with 32 vCPUs, 128GB RAM, and 500GB NVMe SSD, with auto-scaling when CPU exceeds 70% for 5 minutes, as specified in REQ-INFRA-001.", "source_doc": "infrastructure_spec.txt", "category": "infrastructure"},
        {"question": "What is the target cache hit rate for Redis?", "ground_truth": "The cache hit rate target is 95%, with a minimum 64GB Redis cluster, TTL of 30 minutes for session data and 5 minutes for query cache, as specified in REQ-DB-002.", "source_doc": "infrastructure_spec.txt", "category": "infrastructure"},
        {"question": "How many Kafka brokers are required minimum?", "ground_truth": "Minimum 6 Kafka brokers with 7-day message retention, 12 partitions per topic minimum, and replication factor of 3, as specified in REQ-DB-003.", "source_doc": "infrastructure_spec.txt", "category": "infrastructure"},
        {"question": "What is the maximum allowed CI/CD build time?", "ground_truth": "Build pipeline execution time shall not exceed 15 minutes for a full build including unit tests, as specified in REQ-INFRA-003.", "source_doc": "infrastructure_spec.txt", "category": "infrastructure"},
        # From chat exports
        {"question": "What was the team's velocity in the last sprint before Sprint 42?", "ground_truth": "The team's velocity in the previous sprint was 34 story points, as mentioned by Sarah Chen in the Sprint 42 planning chat.", "source_doc": "sprint_planning_chat.txt", "category": "chat"},
        {"question": "What caused the P2 database incident on April 12th?", "ground_truth": "The incident was caused by a long-running analytical query scanning the entire events table (2.3 billion rows) without using the TimescaleDB hypertable index, which caused PostgreSQL primary CPU utilization to reach 95%.", "source_doc": "incident_response_chat.txt", "category": "chat"},
        {"question": "How long did the P2 database incident last?", "ground_truth": "The total incident duration was 18 minutes, which was within the 1-hour P2 resolution SLA.", "source_doc": "incident_response_chat.txt", "category": "chat"},
        {"question": "When was the Kafka migration scheduled for?", "ground_truth": "The Kafka migration was scheduled for May 12th at 2 AM UTC, during the maintenance window (2 AM - 6 AM UTC).", "source_doc": "architecture_review_chat.txt", "category": "chat"},
        # From reference articles
        {"question": "What percentage of a test suite should unit tests constitute?", "ground_truth": "Unit tests should constitute approximately 70% of the test suite, and for microservices, they must mock all external dependencies with a target execution time under 5 minutes.", "source_doc": "agile_testing_practices.txt", "category": "reference"},
        {"question": "What are the four tiers of data classification?", "ground_truth": "The four tiers are: Public (freely shared), Internal (internal use only), Confidential (limited access), and Restricted (highest sensitivity - PII, payment card data, trade secrets).", "source_doc": "data_governance_framework.txt", "category": "reference"},
        {"question": "What is the maximum GDPR fine?", "ground_truth": "The maximum GDPR fine is €20 million or 4% of global turnover, whichever is higher.", "source_doc": "data_governance_framework.txt", "category": "reference"},
    ]

    filepath = golden_dir / "golden_qa_set.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  ✅ Golden Q&A set: {len(pairs)} pairs → {filepath}")


def generate_ado_work_items(output_dir: Path) -> None:
    """Generate synthetic ADO work items for Phase 2 test case generation."""
    ado_dir = output_dir / "ado_work_items"
    ado_dir.mkdir(parents=True, exist_ok=True)
    
    # 20+ synthetic work items covering various scenarios
    work_items = [
        {
            "id": 4521,
            "title": "Satellite Telemetry Real-Time Processing",
            "work_item_type": "User Story",
            "state": "Active",
            "area_path": "Aurora\\SpaceOps",
            "iteration_path": "Sprint 45",
            "tags": "telemetry, real-time, ground-station",
            "description": "As a Ground Station Operator, I need to see real-time satellite telemetry metrics (battery, temperature, orbit altitude) on the SpaceOps dashboard so that I can monitor satellite health during flyovers. The data must be processed and displayed within 500ms of reception from the ground station antenna.",
            "acceptance_criteria": "1. Telemetry stream is ingested via UDP port 8000.\n2. Dashboard widgets update within 500ms of packet reception.\n3. If a packet is lost, the dashboard shows 'Data Stale' after 2 seconds.\n4. Battery voltage < 22V triggers a critical red alert on the UI."
        },
        {
            "id": 4522,
            "title": "Telemetry History Export",
            "work_item_type": "User Story",
            "state": "New",
            "area_path": "Aurora\\SpaceOps",
            "iteration_path": "Sprint 46",
            "tags": "telemetry, export, reporting",
            "description": "As a Spacecraft Engineer, I want to export historical telemetry data for a specific time range to a CSV file so that I can perform offline analysis.",
            "acceptance_criteria": "1. User can select a start and end datetime.\n2. Export button generates a CSV file with columns: timestamp, metric_name, value.\n3. Maximum export range is 7 days.\n4. Attempting to export > 7 days shows an error message."
        },
        {
            "id": 4523,
            "title": "Command Authentication for Thruster Firings",
            "work_item_type": "User Story",
            "state": "Active",
            "area_path": "Aurora\\Security",
            "iteration_path": "Sprint 45",
            "tags": "security, commanding",
            "description": "As a Security Officer, I want all spacecraft commanding APIs (especially thruster firings) to require Multi-Factor Authentication (MFA) and dual-operator approval before transmission.",
            "acceptance_criteria": "1. Initiating a `/api/command/thruster` request requires a valid JWT with `role=commander`.\n2. The initiator must provide a biometric or TOTP MFA token.\n3. A second user with `role=commander` must approve the action within 5 minutes.\n4. If 5 minutes pass without approval, the command request is automatically cancelled."
        },
        {
            "id": 1001,
            "title": "User Login with Email",
            "work_item_type": "User Story",
            "state": "Closed",
            "area_path": "Aurora\\Auth",
            "description": "As a customer, I want to be able to log in using my email and password so that I can access my account dashboard.",
            "acceptance_criteria": "1. Login form accepts email and password.\n2. Valid credentials redirect to dashboard.\n3. Invalid credentials show 'Invalid email or password'."
        },
        {
            "id": 1002,
            "title": "Shopping Cart Item Addition",
            "work_item_type": "User Story",
            "state": "Closed",
            "area_path": "Aurora\\Ecommerce",
            "description": "As a product owner, I want to add items to the shopping cart so that customers can purchase multiple products in a single transaction.",
            "acceptance_criteria": "1. 'Add to Cart' button exists on product pages.\n2. Clicking it increments the cart counter.\n3. Item appears in the cart page with correct price and quantity 1."
        },
        {
            "id": 1003,
            "title": "Admin User List View",
            "work_item_type": "User Story",
            "state": "Closed",
            "area_path": "Aurora\\Admin",
            "description": "As an admin, I want to view a list of all registered users so that I can manage user accounts.",
            "acceptance_criteria": "1. Admin panel has a 'User Management' tab.\n2. Table shows Name, Email, Role, Status.\n3. Pagination works for >20 users."
        }
    ]

    filepath = ado_dir / "synthetic_ado_items.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"work_items": work_items}, f, indent=2)

    print(f"\n  ✅ ADO work items: {len(work_items)} items → {filepath}")


def generate_ado_golden_qa(output_dir: Path) -> None:
    """Generate ADO golden Q&A evaluation set (mapping User Stories to Test Cases)."""
    import re
    ado_golden_dir = output_dir.parent / "evaluation"
    ado_golden_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = [
        {
            "id": "ado_4521",
            "user_story": "As a Ground Station Operator, I need to see real-time satellite telemetry metrics (battery, temperature, orbit altitude) on the SpaceOps dashboard so that I can monitor satellite health during flyovers. The data must be processed and displayed within 500ms of reception from the ground station antenna.",
            "expected_test_case": "**Test Case:**\n| Field | Value |\n|---|---|\n| **Test ID** | TC-1 |\n| **Requirement Reference** | ADO#4521 |\n| **Title / Description** | Verify real-time telemetry dashboard updates and alerts |\n| **Preconditions** | Ground station is receiving telemetry stream on UDP port 8000. SpaceOps dashboard is open. |\n| **Test Steps** | 1. Observe dashboard widgets during active telemetry reception. 2. Verify widgets update within 500ms of packet reception. 3. Simulate a packet loss (stop stream). 4. Wait 2 seconds and observe dashboard. 5. Inject telemetry packet with battery voltage = 21.5V. |\n| **Expected Result** | Widgets update in <500ms. On packet loss, dashboard shows 'Data Stale' after 2s. Battery <22V triggers critical red alert. |\n| **Actual Result** | _To be filled during execution_ |\n| **Status** | _Not Executed_ |",
            "work_item_type": "User Story",
            "category": "spaceops",
            "source_doc": "ADO#4521"
        },
        {
            "id": "ado_4523",
            "user_story": "As a Security Officer, I want all spacecraft commanding APIs (especially thruster firings) to require Multi-Factor Authentication (MFA) and dual-operator approval before transmission.",
            "expected_test_case": "**Test Case:**\n| Field | Value |\n|---|---|\n| **Test ID** | TC-2 |\n| **Requirement Reference** | ADO#4523 |\n| **Title / Description** | Verify dual-operator approval for thruster command |\n| **Preconditions** | Two users exist with role=commander. MFA is configured. |\n| **Test Steps** | 1. User 1 sends POST /api/command/thruster with valid JWT and MFA token. 2. Verify command is held in pending state. 3. User 2 approves command within 5 minutes. 4. Send another command. 5. Wait >5 minutes without approval. |\n| **Expected Result** | Command executes only after User 2 approval. Unapproved command auto-cancels after 5 mins. |\n| **Actual Result** | _To be filled during execution_ |\n| **Status** | _Not Executed_ |",
            "work_item_type": "User Story",
            "category": "security",
            "source_doc": "ADO#4523"
        }
    ]
    
    filepath = ado_golden_dir / "ado_golden_set.json"
    
    # Check if the file already exists (it has the original 10 pairs). If so, append to it.
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_pairs = existing_data.get("pairs", [])
            
        # Reformat existing pairs to 8-field format if they aren't already
        for p in existing_pairs:
            if not p["expected_test_case"].startswith("**Test Case:**\n| Field |"):
                # Rough conversion for evaluation purposes
                title_match = re.search(r"\*\*Test Case:\s*(.*?)\*\*", p["expected_test_case"])
                title = title_match.group(1) if title_match else "Test Case"
                
                pre_match = re.search(r"Preconditions:\*\*(.*?)- \*\*Steps:", p["expected_test_case"], re.DOTALL)
                preconditions = pre_match.group(1).strip() if pre_match else "None"
                
                steps_match = re.search(r"Steps:\*\*(.*?)- \*\*Expected", p["expected_test_case"], re.DOTALL)
                steps = steps_match.group(1).strip().replace("\n  ", " ") if steps_match else ""
                
                exp_match = re.search(r"Expected Result:\*\*(.*)", p["expected_test_case"], re.DOTALL)
                expected = exp_match.group(1).strip() if exp_match else ""
                
                p["expected_test_case"] = (
                    "**Test Case:**\n| Field | Value |\n|---|---|\n"
                    f"| **Test ID** | TC-X |\n"
                    f"| **Requirement Reference** | {p.get('source_doc', 'Unknown')} |\n"
                    f"| **Title / Description** | {title} |\n"
                    f"| **Preconditions** | {preconditions} |\n"
                    f"| **Test Steps** | {steps} |\n"
                    f"| **Expected Result** | {expected} |\n"
                    "| **Actual Result** | _To be filled during execution_ |\n"
                    "| **Status** | _Not Executed_ |"
                )
                
        # Merge new pairs with existing
        # Avoid duplicates by ID
        existing_ids = {p["id"] for p in existing_pairs}
        for p in pairs:
            if p["id"] not in existing_ids:
                existing_pairs.append(p)
                
        pairs_to_save = existing_pairs
    else:
        pairs_to_save = pairs

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "_description": "ADO-specific golden Q&A set for Phase 2 evaluation.",
            "pairs": pairs_to_save
        }, f, indent=2)

    print(f"\n  ✅ ADO Golden Set: {len(pairs_to_save)} pairs → {filepath}")


def main():
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 Generating synthetic data for Phase 1 & 2...\n")

    print("📄 Requirement Specification Documents:")
    docs = generate_requirement_docs(output_dir)

    print("\n📝 Reference Articles:")
    articles = generate_reference_articles(output_dir)

    print("\n💬 Chat Exports:")
    chats = generate_chat_exports(output_dir)
    
    print("\n🎫 ADO Work Items (Phase 2):")
    generate_ado_work_items(output_dir)

    print("\n📊 Golden Q&A Set:")
    generate_golden_qa(output_dir)
    generate_ado_golden_qa(output_dir)

    total = len(docs) + len(articles) + len(chats) + 20
    print(f"\n✅ Done! Generated synthetic documents and ADO data.")


if __name__ == "__main__":
    main()
