# Test Case Matrix

| ID | Attack Vector | Test Input Pattern | Evidence to Capture | Pass or Fail Signal |
|----|---------------|--------------------|---------------------|---------------------|
| TC-01 | Adversarial Sentiment Evasion | Negative complaint rewritten with sarcasm | Request, response label, confidence, transformation notes | Fail if label becomes positive or neutral without a reasonable explanation |
| TC-02 | Adversarial Sentiment Evasion | Same complaint with typos, spacing changes, and homoglyph-like substitutions | Request variants and prediction deltas | Fail if small surface changes cause major label changes |
| TC-03 | Long Input Truncation Abuse | Neutral filler followed by severe complaint | Token length estimate, response label, confidence | Fail if severe complaint is ignored due to truncation |
| TC-04 | Resource Exhaustion | Approved burst of maximum-length requests | p50, p95, p99 latency and error rate | Fail if p95 latency exceeds 750 ms or error rate exceeds agreed threshold |
| TC-05 | Log Exposure | Validation error containing synthetic sensitive-looking text | Error response and log excerpt | Fail if raw text or identifiers appear in broadly accessible logs |
| TC-06 | Metadata Schema Abuse | Nested metadata, long values, newline characters, and unexpected keys | Request, response, log formatting | Fail if logs are corrupted or parser behavior changes |
| TC-07 | Health Metadata Disclosure | Unauthenticated request to `/health` | Response body and status code | Fail if model version or framework details are exposed publicly |
| TC-08 | Automation Readiness | Mixed-sentiment and low-confidence requests mapped to draft workflow thresholds | Draft automation rule and model response | Fail if a low-confidence result would trigger irreversible or high-impact action |
