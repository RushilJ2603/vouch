# Vouch

> Enterprise AI that earns the right to act.

Vouch is a model- and agent-agnostic control plane for enterprise AI. It sits between an AI system
and the tools or workflows that system can affect. For every proposed action, Vouch verifies the
supporting claims, estimates the probability of error, prices the consequence of failure, and
compares expected loss with authority earned from verified outcomes. Routine workflows proceed
automatically, borderline decisions receive deeper verification, and high-consequence actions
remain under human control.

Built for **Problem Statement 1: ControlPlane.ai** by **Team Velocity Gate, NIT Karnataka
Surathkal** for the Accenture Innovation Challenge 2026.

[Business proposal](ControlPlane_Round2_Business_Proposal.pdf) |
[Official problem statement](Accenture_Round2_Detailed_Problem_Statements.pdf)

## The problem Vouch solves

Enterprise AI systems can generate fluent answers while remaining uncertain about the facts or
actions behind them. A customer-service agent may correctly explain a charge but propose the wrong
refund. A knowledge agent may cite an unsupported policy. A decision-support system may encounter
private data or an injected instruction. Treating every response as simply safe or unsafe ignores
the most important distinction: the consequence of being wrong.

Vouch controls the authority to act. It evaluates each response in context, verifies the claims
that matter, measures the exposure created by the proposed action, and applies trust earned from
previous verified outcomes. This produces a decision that is proportional to both uncertainty and
consequence.

The governing comparison is intentionally simple:

```text
expected loss = calibrated probability of error x action exposure

release when action exposure is below the configured review-cost floor
otherwise, release when expected loss <= earned authority budget
check harder when independent evidence could change a borderline decision
otherwise, retain human control
```

An uncertain but reversible low-value action can therefore receive a different route from an
equally uncertain high-value or irreversible action. Trust also remains narrow: success on one
action type, value band or configuration does not grant authority elsewhere. The review-cost floor
provides bounded day-one autonomy only where the full consequence of failure costs less than manual
review; larger actions must earn authority from verified outcomes.

## System at a glance

| Stage | What enters | What Vouch does | What leaves |
|---|---|---|---|
| **Propose** | Agent response, checkable claims and proposed action | Normalizes the result through a provider-neutral typed contract | One decision candidate |
| **Verify** | Candidate, business records and policy | Runs factual, privacy, security and policy checks | An auditable evidence vector |
| **Price** | Evidence and action context | Calibrates error probability and calculates consequence-aware exposure | Expected loss |
| **Authorize** | Expected loss, scoped history and fixed invariants | Compares risk with earned authority and applies the active verification mode | Release, Check harder or Human control |
| **Learn** | Decision record and later verified outcome | Updates scoped trust, monitors drift and activates revocation controls | New authority budget for subsequent decisions |

## Solution capabilities

| Capability | Evidence |
|---|---|
| Three decision routes | Release, Check harder and Human control |
| Three verification modes | Fast, Adaptive and Deep |
| Reference provider stack | DeepSeek as the primary agent and GLM as the policy-selected Tier 2 evaluator |
| Consequence-aware gate | Probability of error x exposure compared with earned budget |
| Flexible governance | Different checks, latency posture and trust scope by enterprise job |
| Audit trail | Response, action, evidence, reason, provider path and latency in one record |

## Decision routes

Vouch returns one of three operational routes. The route controls what the surrounding workflow is
permitted to do next.

| Route | Meaning | Typical result |
|---|---|---|
| **Release** | Evidence and earned authority are sufficient for the measured exposure | The response and permitted action continue automatically |
| **Check harder** | Additional independent evidence could materially change the decision | Tier 2 verification runs, risk is recomputed and the gate decides again |
| **Human control** | Expected loss exceeds earned authority or required evidence is unavailable | The response remains visible, but the consequential action is retained for review |

Fixed safety invariants sit outside earned trust. A policy-window violation, prohibited action,
secret leak, prompt injection or configured hard limit can block execution regardless of prior
success. This prevents historical performance from overriding non-negotiable controls.

## Key concepts

| Concept | Meaning |
|---|---|
| **Structured claim** | A factual statement from the agent that can be compared with an authoritative source |
| **Probability of error** | A calibrated estimate derived from evidence checks and risk signals, rather than the agent's confidence |
| **Action exposure** | The configured consequence of failure, based on action type, value, reversibility and policy context |
| **Expected loss** | Probability of error multiplied by action exposure |
| **Review-cost floor** | Base authority for actions whose entire failure exposure is cheaper than manual review |
| **Earned authority budget** | The maximum expected loss permitted by verified historical performance for the exact trust scope |
| **Trust scope** | The combination of agent, action, value band and configuration within which experience is valid |
| **Tier 2** | An optional independent evaluator used when deeper verification is justified by policy |
| **Outcome ledger** | The append-only record connecting every decision with its evidence, route and later verified result |
| **Configuration fingerprint** | The recorded identity of the models, policies, sensors and settings used for a decision |

## A decision in practice

Consider a customer-support agent responding to a reported duplicate charge:

1. The agent drafts a customer-facing explanation, identifies the order and proposes a refund.
2. Vouch receives the response, structured claims and proposed refund through the typed contract.
3. Deterministic checks compare the order, payment state, prior-refund state and policy window with
   the system of record. Privacy, secret and prompt-injection sensors run on the same request.
4. The calibrated error probability is combined with the exposure of the proposed refund.
5. The gate compares that expected loss with authority earned for this agent, refund action, value
   band and exact configuration.
6. A well-supported, low-exposure action can be released. A borderline case can receive an
   independent Tier 2 assessment. An unsupported, high-exposure or policy-invalid action remains
   under human control.
7. The decision and evidence are recorded. When the real outcome is confirmed, the corresponding
   trust record is updated and future authority can expand, remain stable or be revoked.

The same sequence applies beyond refunds. Claims, exposure rules, hard invariants and trust scopes
change by enterprise job, while the control mechanism remains constant.

## Model- and agent-agnostic design

Vouch governs a contract, not a specific model. The control plane receives three elements from an
upstream AI system:

- the response intended for the end user;
- structured claims that can be checked against systems of record;
- the proposed action, including its type, value and target.

The gate does not require access to model weights, hidden reasoning or provider-specific confidence.
Any model or agent can be integrated through an adapter that produces this contract. The existing
OpenAI-compatible request surface and typed schema provide the reference integration boundary.

The included implementation uses two fixed providers as a reproducible validation standard:

| Component | Function in the reference implementation | Relationship to Vouch |
|---|---|---|
| **DeepSeek** | Produces the customer-facing response, structured factual claims and proposed action | Reference primary agent; replaceable through the typed adapter boundary |
| **Vouch** | Verifies evidence, calculates risk and decides the permitted route | Provider-independent control mechanism |
| **GLM** | Produces an independent structured error estimate when Tier 2 is selected | Optional evaluator; never grants authority on its own |

DeepSeek is fixed across corpus generation and live reference runs, while GLM is fixed across
offline evaluation and live Tier 2 evaluation. This consistency makes calibration and route
comparisons reproducible. Changing either provider requires a new configuration fingerprint and a
new evidence record, but does not change the gate, exposure model or trust mechanism.

## Request lifecycle

The provider-backed reference lifecycle is:

1. The upstream agent produces a response, structured claims and a proposed action. DeepSeek fills
   this role in the reference implementation.
2. Vouch checks the claims against the system of record and runs grounding, privacy, policy, secret
   and prompt-injection checks.
3. The calibrated probability of error is multiplied by action exposure to calculate expected loss.
4. Expected loss is compared with authority earned by that agent for the specific action, value band
   and configuration.
5. The selected policy may invoke GLM for an independent Tier 2 estimate. Fast mode omits this step,
   Adaptive mode uses it for borderline decisions, and Deep mode uses it for meaningful side effects.
6. The gate returns Release, Check harder or Human control, together with the mathematical reason,
   provider path and latency.
7. The complete decision enters the audit ledger. Later verified outcomes update the relevant trust
   record and can expand or revoke authority.

Every completed request enters the same audit ledger used by the committed evaluation, so live
routes and reported results share one decision schema.

## Verification modes

| Mode | Behavior | Tier 2 timeout | Operating context |
|---|---|---:|---|
| **Fast** | Deterministic and local risk checks; Tier 2 disabled | None | Mature, reversible workflows |
| **Adaptive** | Tier 2 only where an independent estimate may change the decision | 2 s | Default enterprise control |
| **Deep** | Tighter earned budget and Tier 2 for every meaningful side effect | 15 s | High-stakes or regulated work |

The controls change business risk posture and verification depth. They never rewrite the model's
score, stored outcomes, hard invariants or evidence record.

## Policy flexibility

Vouch separates a stable decision mechanism from job-specific policy. Each deployment can select
the relevant checks, exposure model, latency budget, escalation behavior and trust scope without
changing the core gate.

| Enterprise job | Representative checks | Exposure basis | Example trust scope |
|---|---|---|---|
| Customer operations | Transaction grounding, refund policy, privacy and duplicate-action checks | Monetary value, reversibility and customer impact | Agent, action and value band |
| Internal knowledge | Source support, document authority, privacy and prompt injection | Sensitivity, distribution and downstream use | Agent, domain and source set |
| Decision support | Grounding, policy compliance, privacy and prohibited-factor checks | Decision consequence and affected population | Agent, decision class and region |
| Workflow automation | Preconditions, target validation, permission boundaries and rollback availability | Blast radius, privilege and reversibility | Agent, tool and operation class |

Latency and verification depth are policy choices. Fast mode favors deterministic evidence for
mature, reversible work. Adaptive mode spends evaluator latency only when it can affect the route.
Deep mode applies a tighter authority budget and systematic independent verification to meaningful
side effects. This makes the system tunable without substituting speed for unbounded autonomy.

## Learning, revocation and failure behavior

Authority grows only from closed-loop evidence. A released decision does not count as success until
its real outcome is verified. The ledger uses a conservative lower confidence bound so limited
favorable history cannot create disproportionate authority. Trust is recalculated within its
original scope and can be reduced immediately when confirmed failures accumulate.

The runtime is designed to fail conservatively:

- missing or stale business evidence prevents unsupported authority;
- a Tier 2 timeout returns control to policy instead of silently approving an action;
- provider and policy changes create a new fingerprint rather than inheriting incompatible trust;
- drift and repeated failures can activate the circuit breaker and restore supervision;
- fixed invariants remain binding at every trust level;
- every fallback, timeout and route transition remains visible in the audit record.

## Measured evidence

The committed evaluation contains **1,466 real DeepSeek replies** across duplicate charges,
misread charges, already-refunded orders, policy-window violations, ambiguous evidence and prompt
injection. DeepSeek supplies the controlled primary-agent workload; GLM supplies the reference Tier
2 evaluations. Vouch independently computes the final probability, exposure, earned budget and
route.

| Evidence | Result |
|---|---:|
| Responses released | **1,164** |
| Responses retained | **302** |
| Refund actions executed | **193** |
| Known-wrong refund actions released | **0** |
| Refund value released | **INR 1,727,616** |
| Global calibration error | **0.0232** |
| Median measured control latency | **35 ms** |

At 100,000 monthly decisions and the measured route mix, 79,400 decisions clear without a
mandatory review touch.

The measured route mix is produced by the same auditable comparison used in the request path:

```text
expected loss = calibrated probability of error x action exposure

action exposure <= review cost         -> release
expected loss <= earned budget        -> release
close enough to change with evidence  -> check harder
expected loss > earned budget         -> retain human control
fixed invariant failed                -> block
```

Authority is scoped by agent, action, value band and configuration. A two-sided Wilson lower bound
prevents limited favorable histories from overstating trust, and three confirmed failures activate
the circuit breaker and restore supervision.

## System architecture

```text
Enterprise agent
(DeepSeek reference)
      |
      v
Vouch request path
  deterministic checks ----+
  streaming risk sensors ---+--> calibrated P(error)
  action exposure ----------+              |
  earned trust ledger --------------------+--> gate --> response route
  Tier 2 evaluator <------------ borderline or deep path
  (GLM reference)
      |                                           |
      v                                           v
Enterprise tools                         Append-only audit log
                                                    |
                                                    v
                                            outcomes --> trust + drift
```

The fast path overlaps sensors with primary-agent generation. The Tier 2 evaluator is
policy-controlled and optional. Every decision stores the feature vector, probability, exposure,
budget, verdict, route, provider identities, configuration fingerprint and later outcome.

The components have deliberately narrow responsibilities:

| Component | Responsibility |
|---|---|
| Agent adapter | Converts an upstream model or agent result into the typed response, claim and action contract |
| Sensors | Collect deterministic and learned evidence without deciding the route |
| Calibrator | Converts the evidence vector into a probability of error |
| Exposure model | Prices the consequence of the proposed action under policy |
| Ledger | Maintains scoped outcome history and the conservative authority budget |
| Gate | Performs the final expected-loss comparison and applies fixed invariants |
| Tier 2 evaluator | Supplies independent evidence when selected by the active verification mode |
| Outcome worker | Closes decisions with verified results, recomputes trust and detects drift |
| Dashboard | Presents live decisions, evidence, route rationale, latency and portfolio-level outcomes |

The gate is the only component permitted to grant execution authority. Models and sensors provide
evidence, but neither can approve its own action.

## Application setup

Python 3.12 is required.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.lock
copy .env.example .env
# add DEEPSEEK_API_KEY and ZAI_API_KEY to .env
.venv\Scripts\python scripts\run_showcase.py
```

The application is available at [http://127.0.0.1:8501](http://127.0.0.1:8501). Committed evidence
loads without provider calls; provider keys are required only for new live executions.

## Reproduce the evidence

No API key is required for tests, bootstrap or replay.

```powershell
.venv\Scripts\python scripts\bootstrap.py
.venv\Scripts\python -m pytest -q
```

On systems with `make`:

```bash
make test
make demo
```

Fresh live runs and corpus generation spend provider credit. Page load, tests, bootstrap and replay
do not. The checked-in corpus and evaluator outputs keep every comparison reproducible.

## Repository map

```text
src/vouch/
  agent.py          typed agent contract and DeepSeek reference adapter
  proxy/            OpenAI-compatible request path and streaming scorer
  sensors/          deterministic, learned and GLM checks
  gate.py           the only component that decides
  ledger.py         Wilson evidence and earned budget
  ladder.py         allow, edit, regenerate, review and block routes
  worker/           outcome closure, trust recomputation and drift
dashboard/
  page.html         light operational control interface
  server.py         evidence and live-evaluation endpoints
config/
  vouch.yaml        model, ledger and runtime configuration
  actions.yaml      per-action exposure and hard limits
  policies.yaml     bounded latency/risk verification modes
scripts/
  run_showcase.py   application server
  bootstrap.py      reproducible calibrator and trust bootstrap
  demo_*.py         calibration, autonomy, latency and cost evidence
tests/              safety, integrity and contract suite
*.pdf               official challenge brief and final business proposal
```
