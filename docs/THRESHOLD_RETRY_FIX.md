# Threshold Retry-Attack Fix — Design

**Status:** All phases landed (1, 2, 3a, 3b, 3c). Named retry-attack family closed.
**Closes:** Mouchet'24, Okada'25, Colin de Verdière 2026 (one fix, the same retry-family attack). Live attack surface (retry of `ThresholdDecrypt`) closed by Phase 2. Theoretical retry-of-keygen surface structurally bounded by Phase 3a.
**Branch:** merged into `main` (`1dcd2c4`, `e03c5d5`, plus the Phase-3a commit landing now).

---

## 1. Threat model

Honest-but-curious **coordinator** (the Opaque server, today) plus an honest-but-curious set of **committee participants**, with one extra capability that the current implementation ignores:

> The coordinator may **retry** a partial-decryption protocol instance — for example,
> after a perceived network drop, timeout, or share-aggregation failure — and
> ask the same participant to re-emit a share for the same ciphertext.

In Lattigo's `mhe.PublicKeySwitchProtocol`, each per-node share is

```
share[i] = sk[i] · a(ct) + e[i]
```

where `e[i]` is fresh Gaussian noise drawn from each participant's internal PRNG
on every call to `GenShare`. If a participant emits a share for the same ciphertext
twice, the coordinator obtains two equations on the same `sk[i]`:

```
share[i]    = sk[i] · a(ct) + e[i]
share[i]'   = sk[i] · a(ct) + e[i]'
```

Subtracting yields `e[i] - e[i]'` — two correlated samples of the flooding
distribution. Mouchet'24 and Okada'25 show that with `O(λ)` such retries an
adaptive adversary can recover `sk[i]` regardless of the flooding magnitude.
Colin de Verdière 2026 sharpens this with adaptive active-set selection.

**Today's `pkg/crypto/threshold/threshold.go` has nothing that prevents this.**
Each `ThresholdDecrypt` call freshly constructs `mhe.NewPublicKeySwitchProtocol`
with fresh internal randomness. A retry on the same ciphertext would happily
emit a second, independently-noised share.

We are NOT yet exploited because Opaque does not retry today — but that is an
implicit invariant, not an enforced one.

## 2. Fix scope (three orthogonal pieces)

### 2.1  Per-emission guard (the load-bearing piece)

Each `Participant` gains a `RetryGuard` keyed by a deterministic fingerprint of
the per-emission inputs:

```
fingerprint = H(ciphertext_bytes || clientPK_bytes || protocol_instance_id)
```

On `GenShare`, the participant checks the guard:

| State | Action |
|-------|--------|
| Fingerprint **unseen** | Emit share, record fingerprint. |
| Fingerprint **seen** | **Refuse** (return error) — do NOT re-emit. |

Refusal (rather than memoised replay) gives the strongest property: the
coordinator cannot ever observe two independent shares for the same input.
Replay would also be safe in principle but adds correctness risk if the
underlying serialisation ever varies subtly across runs.

### 2.2  Per-instance fresh CRS for keygen rounds

`NewCommittee` currently derives one `crs` PRNG and reuses it across
`genCollectivePublicKey`, `genRelinearizationKey`, and `genGaloisKeys`. Fine
in isolation (each call samples a fresh `crp` from the same prng stream), but
this same prng is the only entropy source if a keygen round is ever retried.

Fix: every protocol round derives its own seed from `H(committee_id ‖ round_label ‖ instance_id)`
where `instance_id` is a fresh nonce supplied per protocol invocation by the
coordinator. Even an attacker who tricks the coordinator into "retrying" the
relin keygen round 2 cannot replay-attack a node, because the second instance
has a new `instance_id` and is rejected by the per-emission guard.

For Opaque the keygen-retry surface is small (keygen runs once at setup), but
the property must hold to claim closure of the named CdV 2026 attack.

### 2.3  Coordinator state machine: ABORT, never RETRY

Replace the implicit "if a goroutine errors, propagate up" pattern with an
explicit per-instance state machine:

```
init  →  fanout  →  collect  →  aggregate  →  commit
                         │             │
                         ▼             ▼
                       abort         abort
```

On any `error` or timeout:

1. Issue `Abort(instance_id)` to every participant. Participants free per-instance state.
2. The coordinator MUST mint a new `instance_id` for the retry. The retry is a
   *new protocol instance*, not a continuation of the failed one.

This is what makes the per-emission guard tight: an aborted-then-retried
ciphertext under a *new* `instance_id` produces a *different* fingerprint, and
the guard correctly admits it. A genuine retry of the *same* `instance_id` —
which the design forbids — would be refused.

## 3. Phased implementation

This fix lands in three commits to keep each diff reviewable and the bench
numbers separable:

| Phase | Files | Closes | Status |
|-------|-------|--------|--------|
| 1 | `docs/THRESHOLD_RETRY_FIX.md`, `pkg/crypto/threshold/retry_guard.go`, `pkg/crypto/threshold/retry_guard_test.go` | Design + standalone guard primitive | ✅ landed (`1dcd2c4`) |
| 2 | `threshold.go`: wire `RetryGuard` into `Participant`, hash ciphertext+pk into fingerprint, plumb `instanceID` through `ThresholdDecrypt` signature | Per-emission guard live in PCKS path | ✅ landed (`e03c5d5`) |
| 3a | `threshold.go`: per-round CRS derivation in `genCollectivePublicKey`, `genRelinearizationKey`, `genGaloisKeys` via `derivePerRoundCRS(epochSeed, roundLabel)`. Per-Galois-element sub-derivation. Distinct rounds within an epoch can never share a CRP. | Structural bound on retry-of-single-keygen-round attacks (Colin de Verdière 2026 family) | ✅ landed |
| 3b | `instance.go`: coordinator state machine (`unknown → reserved → running → committed \| aborted`). `BeginInstance()`, `AbortInstance()` public API. `ThresholdDecrypt` activates the instance, commits on success, aborts on any error. Reuse of a terminal `instanceID` is refused at the coordinator BEFORE any participant is contacted. | Defense-in-depth on top of the Phase 2 per-participant guard. Refuse instanceID reuse at the cheaper coordinator-side gate. | ✅ landed |
| 3c | `chaos_test.go`: concurrent same-ID hammering (one winner, all participants see exactly 1 fingerprint), N distinct IDs all succeed in parallel, abort-then-retry blocks, malicious-coordinator-bypassed RetryGuard still catches replay, concurrent guard.Admit one-winner stress. | Empirical confidence in the 1+2+3a+3b stack under adversarial conditions. | ✅ landed |

Phase 2 was the API-changing piece (`ThresholdDecrypt` gained an `instanceID`
argument). Phase 3a was internal. Phases 3b + 3c shipped 2026-05-13 with
two new files: `pkg/crypto/threshold/instance.go` (state machine) and
`pkg/crypto/threshold/chaos_test.go` (adversarial-pattern suite). The
public API gained `Committee.BeginInstance()` (mint + reserve fresh ID)
and `Committee.AbortInstance(id)` (coordinator-initiated abort).

Each phase preserves existing tests' hardcoded expected values (e.g.,
`TestThresholdDecryptScalar` expects 0.52 ± 0.01) — that is the strongest
correctness signal Opaque has on the threshold path.

## 4. Performance impact

| Phase | Estimated overhead | Notes |
|-------|--------------------|-------|
| Per-emission guard | <50 µs per call | One BLAKE2b-256 hash + map lookup |
| Per-round CRS | 0 (keygen is one-time) | Setup-time only |
| Abort/state-machine | 0 in the happy path | Cost only on failure |

Net expected impact on the headline 464 ms latency: **negligible** (sub-millisecond).
Verification post-Phase 3 against the SIFT1M m6i.2xlarge bench.

## 5. Security claim once Phase 3 lands

> Under the σ=2^45 + DecodePublic(logprec=10) parameters and the per-emission
> + per-instance CRS discipline, no adversary in the honest-but-curious
> coordinator + honest-but-curious committee model can extract any participant's
> secret key share by retrying or replaying a protocol instance, regardless of
> how many such retry attempts are made — because no second share is ever emitted
> for the same fingerprint.

That is the formal closure of the Mouchet/Okada/CdV retry-attack family. It
takes the named-attack count against Opaque's threshold layer from 1 → 0
(see `docs/SECURITY_MODEL.md` §8).

## 6. What this does NOT close

- **Simulation security** of Lattigo's threshold construction (different scheme;
  would require rewriting the threshold layer on top of GLM-style proofs).
- **Malicious** committee participants emitting bogus shares — mitigation
  belongs in Phase 4+ (verifiable secret sharing, share commit-and-prove). Not
  in current threat model.
- **Active server** (Compass-tier) tampering with the coordinator state — also
  out of scope; orthogonal to the retry attack.
