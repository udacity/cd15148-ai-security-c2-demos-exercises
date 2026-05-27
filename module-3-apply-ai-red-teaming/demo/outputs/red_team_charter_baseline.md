# Red Team Charter: SupportSentiment AI Security Assessment

## Engagement Details

| Field | Value |
|-------|-------|
| Engagement Name | SupportSentiment Pre-Production AI Red Team Assessment |
| Date | May 2026 |
| Assessor | Security Engineering Team |
| Sponsor | Director of Customer Support Platforms |
| System Owner | ML Platform Team |
| Estimated Assessment Window | 1 business day |

## Objectives

Evaluate the security posture of the SupportSentiment customer-support sentiment analysis service by executing controlled adversarial tests across six LLM-surfaced attack vectors:

1. **Adversarial Sentiment Evasion**: Test whether sarcasm, mixed sentiment, typos, homoglyphs, or phrasing changes cause meaningful label flips.
2. **Long Input Truncation Abuse**: Determine whether important negative content can be hidden beyond tokenizer or preprocessing limits.
3. **Inference Resource Exhaustion**: Assess whether maximum-length requests create unacceptable latency or availability degradation.
4. **Sensitive Data Exposure Through Logs or Metadata**: Verify whether raw ticket text, customer identifiers, validation errors, or metadata are exposed in logs.
5. **Unauthenticated Model Metadata Disclosure**: Determine whether `/health` leaks model or framework details that assist targeted attacks.
6. **Unsafe Automation Readiness**: Evaluate whether uncalibrated confidence scores and mixed-sentiment errors could create unsafe downstream workflow decisions.

## Scope

### In Scope

- `POST /v1/sentiment` inference endpoint in the pre-production environment
- `GET /v1/health` endpoint exposed through the pre-production API gateway
- Request validation for `ticket_id`, `customer_id`, `text`, and `metadata`
- Tokenization, preprocessing, and model inference behavior
- Pre-production service logs and metrics related to test traffic
- Container dependency scan and model artifact provenance review
- Draft automation rules that may consume sentiment labels and confidence scores

### Out of Scope

- Production systems, production customer data, and live support workflows
- Credential theft, phishing, or attacks against employees
- Unauthorized access to third-party systems
- Destructive denial-of-service testing outside agreed rate limits
- Attempts to modify production model checkpoints or deployment pipelines
- Physical security and corporate network penetration testing

## Rules of Engagement

1. Perform all testing in the isolated pre-production environment.
2. Use synthetic support tickets only; do not submit real customer data.
3. Use approved API keys and identify all red team traffic with `metadata.test_run_id`.
4. Keep load tests within the agreed pre-production ceiling of 25 requests per second unless the system owner approves a different limit.
5. Do not attempt credential theft, persistence, lateral movement, or exploitation of unrelated infrastructure.
6. Stop testing and notify the system owner if service latency exceeds 1,000 ms for more than 5 consecutive minutes.
7. Preserve request samples, response bodies, timestamps, and log excerpts needed to reproduce each finding.
8. Label each result as confirmed, partially confirmed, or not reproduced.

## Success Criteria

| Attack Vector | Success Metric |
|---------------|----------------|
| Adversarial Sentiment Evasion | Demonstrate at least 3 realistic text transformations that flip a clearly negative ticket to neutral or positive, or reduce confidence below 0.60 |
| Long Input Truncation Abuse | Show that sentiment changes when the same negative content is moved beyond preprocessing or tokenizer limits |
| Inference Resource Exhaustion | Demonstrate p95 latency above 750 ms under approved maximum-length request testing |
| Sensitive Data Exposure Through Logs or Metadata | Confirm that raw ticket text, customer identifiers, malformed metadata, or sensitive-looking values appear in logs or error telemetry |
| Unauthenticated Model Metadata Disclosure | Confirm that unauthenticated requests reveal model version, framework, or deployment details useful for reconnaissance |
| Unsafe Automation Readiness | Identify at least 2 cases where confidence and label behavior would create incorrect routing or escalation if automation were enabled |

## Deliverables

1. Completed red team charter
2. LLM-generated attack vector candidate list with rejected speculative items
3. Test case matrix for each in-scope attack vector
4. Vulnerability log with severity, confidence, evidence, and recommended remediation
5. Executive summary describing which risks require deeper validation before production rollout

## Assessment Notes

LLM-generated findings are used as planning inputs only. A finding is reportable only after a human reviewer confirms that the test is relevant, reproducible, and supported by system evidence.
