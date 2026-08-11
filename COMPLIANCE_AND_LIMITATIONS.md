# Limitations, Compliance Posture & Trust Considerations

This document is an honest, self-assessed audit of where this project stands
on regulatory compliance, accuracy/reliability, and user trust — written for
transparency to evaluators, and as a template for what a production version
of this system would still need to address. **This is not a legal opinion.**
No AI system (this one included) can certify regulatory compliance; the
items below marked "requires legal review" genuinely need a licensed
attorney or compliance professional before any real-world deployment
handling real users' financial decisions.

---

## 1. Regulatory Compliance

### What's currently addressed
- Every agent's system prompt includes explicit disclaimers (not investment/
  tax/legal advice; consult a licensed professional).
- The system never executes trades or moves money — no broker-dealer
  functionality exists, which avoids an entire category of regulatory
  triggers.
- Portfolio and session data are held in-memory only (`MemorySaver`) and are
  wiped on app restart — no persistent storage of user financial data in the
  current configuration.
- A structural (non-LLM-dependent) disclaimer banner is shown on every page
  load in the Streamlit UI, not just embedded in generated chat text.

### Known gaps — requires legal review before any production use
- **Portfolio Analysis Agent risk**: This agent analyzes a user's actual
  holdings and produces specific, personalized commentary. In the U.S., this
  pattern can approach the definition of "personalized investment advice"
  under the Investment Advisers Act of 1940, which has real registration
  implications for a commercial product. A prompt-level disclaimer does not
  change what the output functionally is. **This is the single highest-risk
  component in the system from a regulatory standpoint.**
- No Terms of Service or Privacy Policy exist. Required before collecting
  any real user data in a live product.
- No data deletion / right-to-be-forgotten mechanism (relevant under
  GDPR/CCPA if deployed to EU/California users with persistent storage).
- `yfinance` accesses Yahoo Finance's unofficial/undocumented endpoints,
  which sit outside Yahoo's official API terms of service. Widely used for
  personal/educational projects; not something to represent as a licensed,
  production-grade commercial data feed without a proper commercial data
  agreement (e.g., a licensed Alpha Vantage/other paid tier).
- No KYC/AML processes — not applicable to the current feature set (no
  transactions), but would become relevant if any transactional features
  were ever added.

### Recommendation for this capstone
Present this project explicitly as an **educational simulation / capstone
prototype**, not a real advisory product, in both the README and any
demo/presentation. That framing is honest, defensible, and is a stronger
answer to a compliance question than an unsupported claim of compliance.

---

## 2. Accuracy & Reliability

### What's verified and solid
- 63 automated tests, 89% code coverage (`pytest --cov`). **Important
  distinction**: this measures code correctness (functions behave as
  designed, error paths are handled), not the factual/financial accuracy of
  generated content — those are different kinds of correctness, and only the
  first is currently automated.
- Market data has tested fallback chains (yfinance → Alpha Vantage → clean
  user-facing error) and TTL caching to reduce rate-limit exposure.
- RAG retrieval failures degrade gracefully (empty result set) rather than
  crashing the agent.

### Known gaps
- **No fact-checking or hallucination-detection layer.** If RAG retrieval
  returns no relevant knowledge-base content, agents still answer from the
  LLM's own parametric knowledge, which can be outdated (e.g., contribution
  limits, tax brackets) or occasionally incorrect, with no explicit signal
  to the user that grounding failed for that specific response.
- **The 62-article knowledge base was authored by an AI assistant (as part
  of this project's development), not reviewed by a CPA, CFP, or tax
  attorney.** Content was written carefully against well-established,
  mainstream financial concepts, but "AI-authored, unreviewed by a licensed
  professional" is a meaningfully different claim than "professionally
  vetted," and should be stated as such rather than implied otherwise.
- Alpha Vantage's free tier (25 requests/day) is not sufficient for
  reliable operation beyond light demo/development use.
- No structured logging or audit trail of what the system told a specific
  user at a specific time — relevant for any future debugging of reported
  inaccuracies or compliance audits.
- No automated evaluation harness (e.g., a golden-answer test set to
  measure response quality/accuracy over time as prompts or models change).

### Recommended next steps (future work, beyond current scope)
- Add an evaluation set of financial Q&A pairs with expected key facts, and
  score agent responses against it periodically.
- Add a "confidence" or "grounded in knowledge base: yes/no" indicator to
  each response so users can see when an answer wasn't backed by curated
  content.
- Have a licensed financial professional review the knowledge base content
  before any real-world use beyond a demo/capstone context.

---

## 3. User Trust

### What's currently addressed
- Source attribution is shown with every knowledge-base-grounded response.
- Errors are caught and shown as friendly messages, never raw stack traces.
- A structural disclaimer banner (not just LLM-generated text) is shown on
  every page load and reinforced in the sidebar.
- The system is clearly presented as an AI assistant (not impersonating a
  human advisor) throughout the interface.

### Known gaps
- No user authentication — sessions are anonymous UUIDs with no durable
  identity, so there is no way to build a persistent, accountable
  relationship with a specific user across visits, and no protection if a
  session ID were somehow guessed or shared.
- No user-facing mechanism to report an inaccurate or harmful response.
- No accessibility (WCAG) audit has been performed on the Streamlit UI.
- No published data-handling explanation for users beyond this document
  (i.e., no in-app "how we handle your data" page).

---

## Summary

This project demonstrates the core technical architecture of a multi-agent
financial assistant — orchestration, RAG grounding, real-time data
integration, and a usable interface — to a genuinely solid, tested standard.
It is **not**, and does not claim to be, a compliant, production-ready
financial advisory product. The gaps above are the concrete, specific work
that would separate "working capstone prototype" from "real product,"
and are presented here deliberately rather than glossed over, since
identifying these gaps accurately is itself part of demonstrating
professional judgment on a project like this.
