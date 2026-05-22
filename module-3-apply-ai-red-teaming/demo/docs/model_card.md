# Model Card: SupportSentiment Classifier

## Model Details

| Field | Value |
|-------|-------|
| Model Name | `distilbert-support-sentiment-0.9.0` |
| Task | Three-class sentiment classification: positive, neutral, negative |
| Framework | Hugging Face Transformers with PyTorch |
| Base Model | DistilBERT uncased |
| Training Window | January 2024 to December 2025 |
| Intended Use | Prioritize support queues and summarize sentiment trends |
| Deployment Status | Pre-production |

## Training Data

The model was fine-tuned on approximately 1.8 million historical support tickets from email, chat, phone transcripts, and social support channels. Labels were generated from a mix of human review and legacy rule-based sentiment labels. Personally identifiable information was masked before training, but masking quality has not been independently audited.

## Evaluation Summary

| Metric | Value |
|--------|-------|
| Macro F1 | 0.89 |
| Positive F1 | 0.92 |
| Neutral F1 | 0.86 |
| Negative F1 | 0.90 |
| Average Latency | 220 ms |

## Known Limitations

- Sarcasm and mixed-sentiment tickets are sometimes classified incorrectly.
- Very long ticket histories may be truncated before inference.
- The model has not been tested against intentional adversarial text perturbations.
- Locale handling is limited; non-English tickets are routed through the same model.
- Confidence values are not calibrated for automated decision-making.

## Deployment Assumptions

- API clients are trusted enterprise systems, but the endpoint is internet reachable.
- Sentiment outputs may later trigger workflow automation such as escalation, routing, or retention priority.
- Model artifacts are stored in the container image rather than pulled dynamically at runtime.
- No customer-facing explanation is returned, only the label and confidence value.
