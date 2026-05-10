// Package threshold provides threshold CKKS homomorphic encryption for Opaque.
//
// In threshold CKKS, the decryption key is split across N independent participants
// (the key committee). Any t-of-N participants can jointly decrypt by producing
// partial key-switch shares that re-encrypt results under a querying client's
// ephemeral public key. No single party ever holds the full secret key.
//
// Usage flow:
//  1. Setup: NewCommittee creates N participants, collectively generates keys
//  2. Query: Client creates a session, encrypts query with collective pk
//  3. Server: Computes HE dot products using eval keys (no secret key needed)
//  4. Decrypt: Each participant produces a partial key-switch share
//  5. Aggregate: Shares are combined into a ciphertext under the client's pk
//  6. Client: Decrypts locally with their ephemeral sk
package threshold

import (
	"crypto/sha256"
	"fmt"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/mhe"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

// Participant represents a single key committee node.
// Each participant holds a secret key share and can produce partial decryption shares.
type Participant struct {
	// ID is the unique identifier for this participant (1-indexed, non-zero).
	ID mhe.ShamirPublicPoint

	// sk is this participant's secret key share.
	sk *rlwe.SecretKey

	// shamirShare is the t-of-N Shamir share (set when threshold < N).
	shamirShare mhe.ShamirSecretShare

	// params are the shared CKKS parameters.
	params hefloat.Parameters

	// guard prevents double-emission of partial-decryption shares for the same
	// (instanceID, ciphertext, clientPK) fingerprint — the load-bearing
	// component closing the Mouchet'24 / Okada'25 / Colin de Verdière 2026
	// retry-attack family. See docs/THRESHOLD_RETRY_FIX.md §2.1.
	guard *RetryGuard
}

// Committee orchestrates the threshold CKKS protocol across N participants.
type Committee struct {
	// Participants in the key committee.
	Participants []*Participant

	// N is the total number of participants.
	N int

	// Threshold is the minimum number of participants needed for decryption.
	Threshold int

	// CollectivePK is the jointly generated public key.
	CollectivePK *rlwe.PublicKey

	// RelinKey is the jointly generated relinearization key.
	RelinKey *rlwe.RelinearizationKey

	// GaloisKeys are the jointly generated rotation keys.
	GaloisKeys []*rlwe.GaloisKey

	// params are the shared CKKS parameters.
	params hefloat.Parameters

	// epochSeed is the fresh 64-byte entropy bound to this Committee's keygen.
	// All per-round CRS derivations (CPK, Relin, Galois) are derived from
	// H(epochSeed || roundLabel) so that distinct rounds use distinct CRPs and
	// any future "retry a single keygen round" path would naturally use the
	// same epoch's CRS for that round (and the per-emission RetryGuard would
	// then catch a duplicate share for the same fingerprint). Re-running the
	// full keygen from scratch must mint a new epochSeed — that's how Phase 3
	// abort-not-retry semantics close the Colin de Verdière 2026 attack on
	// keygen rounds. See docs/THRESHOLD_RETRY_FIX.md §2.2.
	epochSeed []byte

	// SimulatedRTT adds a simulated network round-trip delay per participant
	// in ThresholdDecrypt to model real distributed deployment. Zero means no delay.
	// In a real deployment, the coordinator sends ct+clientPK to each node (1 RTT)
	// and each node sends back its partial share (1 RTT), all in parallel.
	// Typical values: 1-2ms (same data center), 50-100ms (cross-region).
	SimulatedRTT time.Duration
}

// derivePerRoundCRS returns a fresh sampling.PRNG for a given keygen round,
// seeded by H(epochSeed || roundLabel). Each named round (CPK, Relin, Galois)
// gets a deterministic-from-epoch but distinct-across-rounds CRS so the SampleCRP
// outputs of one round can never collide with another's. This is the structural
// piece of the Phase 3 fix for the Colin de Verdière 2026 retry-attack: a future
// "retry round X" path within the same epoch will read the same CRS, causing
// the per-participant RetryGuard (Phase 2) to refuse the duplicate emission;
// a fresh-epoch retry derives a new CRS family entirely.
func derivePerRoundCRS(epochSeed []byte, roundLabel string) (sampling.PRNG, error) {
	h := sha256.New()
	// Length-prefix to defeat boundary-collision (analogous to writeLP in retry_guard.go).
	var lenBuf [8]byte
	for i, b := range [][]byte{epochSeed, []byte(roundLabel)} {
		_ = i
		n := uint64(len(b))
		for j := 0; j < 8; j++ {
			lenBuf[j] = byte(n >> (8 * j))
		}
		h.Write(lenBuf[:])
		h.Write(b)
	}
	return sampling.NewKeyedPRNG(h.Sum(nil))
}

// ClientSession represents a querying client's ephemeral session.
// The client generates a fresh keypair per session for receiving threshold-decrypted results.
type ClientSession struct {
	// SK is the client's ephemeral secret key (private, never shared).
	SK *rlwe.SecretKey

	// PK is the client's ephemeral public key (sent to the committee).
	PK *rlwe.PublicKey

	// params are the CKKS parameters.
	params hefloat.Parameters

	// decryptor decrypts ciphertexts under this session's key.
	decryptor *rlwe.Decryptor

	// encoder decodes CKKS plaintexts.
	encoder *hefloat.Encoder
}

// NewCommittee creates a new threshold CKKS committee with n participants and threshold t.
// This runs the full collective key generation protocol (simulated locally for POC).
func NewCommittee(n, threshold int) (*Committee, error) {
	if n < 1 {
		return nil, fmt.Errorf("n must be >= 1, got %d", n)
	}
	if threshold < 1 || threshold > n {
		return nil, fmt.Errorf("threshold must be in [1, %d], got %d", n, threshold)
	}

	params, err := newParameters()
	if err != nil {
		return nil, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// Mint a fresh 64-byte epoch seed. Every per-round CRS (CPK/Relin/Galois)
	// derives from this seed via H(epochSeed || roundLabel) — see
	// derivePerRoundCRS. In a distributed-coordinator deployment the seed is
	// agreed upon out-of-band; for the in-process simulator we draw fresh
	// entropy each time NewCommittee is called.
	epochSeed := make([]byte, 64)
	prng, err := sampling.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("failed to create PRNG: %w", err)
	}
	if _, err := prng.Read(epochSeed); err != nil {
		return nil, fmt.Errorf("failed to generate epoch seed: %w", err)
	}

	c := &Committee{
		N:         n,
		Threshold: threshold,
		params:    params,
		epochSeed: epochSeed,
	}

	// Step 1: Each participant generates a local secret key share.
	kgen := rlwe.NewKeyGenerator(params)
	c.Participants = make([]*Participant, n)
	for i := 0; i < n; i++ {
		c.Participants[i] = &Participant{
			ID:     mhe.ShamirPublicPoint(i + 1), // 1-indexed
			sk:     kgen.GenSecretKeyNew(),
			params: params,
			guard:  NewRetryGuard(),
		}
	}

	// Step 2: If threshold < N, distribute Shamir shares.
	if threshold < n {
		if err := c.setupThreshold(); err != nil {
			return nil, fmt.Errorf("threshold setup failed: %w", err)
		}
	}

	// Step 3: Collective public key generation.
	if err := c.genCollectivePublicKey(); err != nil {
		return nil, fmt.Errorf("collective public key generation failed: %w", err)
	}

	// Step 4: Collective relinearization key generation.
	if err := c.genRelinearizationKey(); err != nil {
		return nil, fmt.Errorf("relinearization key generation failed: %w", err)
	}

	// Step 5: Collective Galois key generation (for dot product rotations).
	if err := c.genGaloisKeys(); err != nil {
		return nil, fmt.Errorf("Galois key generation failed: %w", err)
	}

	return c, nil
}

// GetParams returns the CKKS parameters.
func (c *Committee) GetParams() hefloat.Parameters {
	return c.params
}

// GetEvalKeySet returns the evaluation key set for server-side HE operations.
// This contains only public keys — no secret key material.
func (c *Committee) GetEvalKeySet() *rlwe.MemEvaluationKeySet {
	return rlwe.NewMemEvaluationKeySet(c.RelinKey, c.GaloisKeys...)
}

// GetEncoder returns an encoder for the committee's parameters.
func (c *Committee) GetEncoder() *hefloat.Encoder {
	return hefloat.NewEncoder(c.params)
}

// NewClientSession creates a new ephemeral client session.
// The client generates a fresh keypair for receiving decrypted results.
func (c *Committee) NewClientSession() (*ClientSession, error) {
	kgen := rlwe.NewKeyGenerator(c.params)
	sk, pk := kgen.GenKeyPairNew()

	return &ClientSession{
		SK:        sk,
		PK:        pk,
		params:    c.params,
		decryptor: rlwe.NewDecryptor(c.params, sk),
		encoder:   hefloat.NewEncoder(c.params),
	}, nil
}

// EncryptVector encrypts a vector using the collective public key.
func (c *Committee) EncryptVector(vector []float64) (*rlwe.Ciphertext, error) {
	enc := rlwe.NewEncryptor(c.params, c.CollectivePK)
	encoder := hefloat.NewEncoder(c.params)

	maxSlots := c.params.MaxSlots()
	padded := make([]float64, maxSlots)
	copy(padded, vector)

	pt := hefloat.NewPlaintext(c.params, c.params.MaxLevel())
	if err := encoder.Encode(padded, pt); err != nil {
		return nil, fmt.Errorf("failed to encode vector: %w", err)
	}

	ct, err := enc.EncryptNew(pt)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt: %w", err)
	}

	return ct, nil
}

// ThresholdDecrypt performs threshold decryption of a ciphertext, re-encrypting
// the result under the client's public key. The client can then decrypt locally.
//
// This uses PublicKeySwitchProtocol: each participant produces a partial share,
// shares are aggregated, and the result is a ciphertext under clientPK.
//
// instanceID is a coordinator-supplied protocol-instance nonce — distinct
// invocations of the threshold protocol MUST use distinct instanceIDs. Each
// participant tracks the (instanceID, ct, clientPK) fingerprint via its
// RetryGuard and refuses to emit a second share for the same fingerprint,
// closing the Mouchet'24 / Okada'25 / Colin de Verdière 2026 retry-attack
// family. Use NewInstanceID() if the caller doesn't have a domain-specific
// nonce — see docs/THRESHOLD_RETRY_FIX.md §2 for the threat model.
func (c *Committee) ThresholdDecrypt(ct *rlwe.Ciphertext, clientPK *rlwe.PublicKey, instanceID []byte) (*rlwe.Ciphertext, error) {
	// No mutex needed: all state created per-call (pcks, shares, activeKeys).
	// Participant fields (shamirShare, sk) are read-only after setup.
	// Concurrent calls are safe and simulate real distributed deployment
	// where multiple ciphertexts are decrypted in parallel.

	// Noise flooding sigma=2^45 — provable 128-bit IND-CPA^D security per
	// Bergamaschi et al. PKC 2025 (ePrint 2024/424) for tau ≤ 2^20 decryptions
	// per key. Sigma stays below the CKKS scale (2^60 with the restructured
	// LogQ chain — see newParameters), leaving 2^45/2^60 = 2^-15 ≈ 3e-5 of
	// post-decode noise on [-1,1] scores, well below the 2^-10 DecodePublic
	// rounding precision so signal is preserved. Composes with DecodePublic
	// for Li-Micciancio defense in depth.
	noiseFlood := ring.DiscreteGaussian{Sigma: 1 << 45, Bound: 6 * (1 << 45)}

	activeParticipants := c.Participants[:c.Threshold]

	// Convert Shamir shares to additive form (each participant independently).
	activeKeys := make([]*rlwe.SecretKey, c.Threshold)
	if c.Threshold < c.N {
		activePoints := make([]mhe.ShamirPublicPoint, c.Threshold)
		for i, p := range activeParticipants {
			activePoints[i] = p.ID
		}

		// In a real deployment, each node does this locally and in parallel.
		var wg sync.WaitGroup
		errs := make([]error, c.Threshold)
		for i, p := range activeParticipants {
			wg.Add(1)
			go func(i int, p *Participant) {
				defer wg.Done()
				cmb := mhe.NewCombiner(*c.params.GetRLWEParameters(), p.ID, activePoints, c.Threshold)
				additiveSK := rlwe.NewSecretKey(c.params)
				errs[i] = cmb.GenAdditiveShare(activePoints, p.ID, p.shamirShare, additiveSK)
				activeKeys[i] = additiveSK
			}(i, p)
		}
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				return nil, fmt.Errorf("failed to generate additive share for participant %d: %w", activeParticipants[i].ID, err)
			}
		}
	} else {
		for i, p := range activeParticipants {
			activeKeys[i] = p.sk
		}
	}

	// Serialize ct + clientPK ONCE (outside the worker loop) for the
	// per-participant guard fingerprint. Lattigo's binary serialization is
	// deterministic, so the same ct will always produce the same bytes — that's
	// what makes the guard's "refuse second emission for same fingerprint"
	// property hold.
	ctBytes, err := ct.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize ciphertext for retry-guard fingerprint: %w", err)
	}
	pkBytes, err := clientPK.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize clientPK for retry-guard fingerprint: %w", err)
	}

	// Each participant generates their partial key-switch share in parallel.
	// In a real deployment, this is the network-parallel step: each node
	// computes its share independently and sends it to the aggregator.
	shares := make([]mhe.PublicKeySwitchShare, c.Threshold)
	{
		var wg sync.WaitGroup
		shareErrs := make([]error, c.Threshold)
		for i := range shares {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				// Simulate network: coordinator sends ct+clientPK to node (1 RTT),
				// node computes share, sends it back (1 RTT). Both happen per-node.
				if c.SimulatedRTT > 0 {
					time.Sleep(2 * c.SimulatedRTT)
				}
				// RETRY-GUARD CHECK: refuse to emit a second share for this
				// (instanceID, ct, clientPK) fingerprint. This is the fix for
				// Mouchet'24 / Okada'25 / Colin de Verdière 2026.
				if err := activeParticipants[i].guard.Admit(instanceID, ctBytes, pkBytes); err != nil {
					shareErrs[i] = fmt.Errorf("participant %d retry-guard refused emission: %w", activeParticipants[i].ID, err)
					return
				}
				// Each participant gets its own PCKS instance (internal PRNG state).
				pcks, err := mhe.NewPublicKeySwitchProtocol(c.params, noiseFlood)
				if err != nil {
					shareErrs[i] = err
					return
				}
				shares[i] = pcks.AllocateShare(ct.Level())
				pcks.GenShare(activeKeys[i], clientPK, ct, &shares[i])
			}(i)
		}
		wg.Wait()
		for i, err := range shareErrs {
			if err != nil {
				return nil, fmt.Errorf("participant %d share gen failed: %w", i, err)
			}
		}
	}

	// Aggregation is cheap and sequential (coordinator/DB server does this).
	pcksAgg, err := mhe.NewPublicKeySwitchProtocol(c.params, noiseFlood)
	if err != nil {
		return nil, fmt.Errorf("failed to create PCKS aggregator: %w", err)
	}
	aggregated := shares[0]
	for i := 1; i < c.Threshold; i++ {
		if err := pcksAgg.AggregateShares(aggregated, shares[i], &aggregated); err != nil {
			return nil, fmt.Errorf("failed to aggregate shares: %w", err)
		}
	}

	// Apply the key-switch: result is encrypted under clientPK.
	ctOut := rlwe.NewCiphertext(c.params, 1, ct.Level())
	pcksAgg.KeySwitch(ct, aggregated, ctOut)

	return ctOut, nil
}

// decodeLogPrec rounds decrypted values to 2^-decodeLogPrec precision per
// Lattigo SECURITY.md guidance for IND-CPA^D mitigation (Li-Micciancio attack).
// Centroid similarity scores in [-1, 1] need ~5-7 bits precision to separate
// top-K clusters; logprec=10 gives 2^-10 ≈ 1e-3 precision with safe margin.
const decodeLogPrec = 10

// DecryptScalar decrypts a scalar value from a ciphertext using the client's session key.
func (s *ClientSession) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	pt := s.decryptor.DecryptNew(ct)
	decoded := make([]float64, 1)
	if err := s.encoder.DecodePublic(pt, decoded, decodeLogPrec); err != nil {
		return 0, fmt.Errorf("failed to decode: %w", err)
	}
	return decoded[0], nil
}

// DecryptBatchScalars decrypts multiple dot product results from a batch ciphertext.
func (s *ClientSession) DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error) {
	pt := s.decryptor.DecryptNew(ct)
	maxSlots := s.params.MaxSlots()
	decoded := make([]float64, maxSlots)
	if err := s.encoder.DecodePublic(pt, decoded, decodeLogPrec); err != nil {
		return nil, fmt.Errorf("failed to decode: %w", err)
	}

	results := make([]float64, numCentroids)
	for i := 0; i < numCentroids; i++ {
		pos := i * dimension
		if pos < len(decoded) {
			results[i] = decoded[pos]
		}
	}
	return results, nil
}

// Close zeros the client session's secret key material.
func (s *ClientSession) Close() {
	if s.SK != nil {
		for i := range s.SK.Value.Q.Coeffs {
			for j := range s.SK.Value.Q.Coeffs[i] {
				s.SK.Value.Q.Coeffs[i][j] = 0
			}
		}
		s.SK = nil
	}
	s.decryptor = nil
}

// newParameters creates CKKS parameters matching Opaque's existing configuration.
// Mirrors crypto.NewParameters() — LogQ=[60×5] + LogDefaultScale=60 for σ=2^45
// noise-flooding headroom while preserving signal (Bergamaschi PKC 2025 bound).
func newParameters() (hefloat.Parameters, error) {
	return hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{60, 60, 60, 60, 60},
		LogP:            []int{61, 61},
		LogDefaultScale: 60,
	})
}

// setupThreshold distributes Shamir shares for t-of-N threshold.
func (c *Committee) setupThreshold() error {
	thr := mhe.NewThresholdizer(c.params)

	// Each participant creates a Shamir polynomial from their secret key.
	polynomials := make([]mhe.ShamirPolynomial, c.N)
	for i, p := range c.Participants {
		poly, err := thr.GenShamirPolynomial(c.Threshold, p.sk)
		if err != nil {
			return fmt.Errorf("participant %d: %w", p.ID, err)
		}
		polynomials[i] = poly
	}

	// Each participant evaluates their polynomial for every other participant
	// and aggregates received shares.
	for i, p := range c.Participants {
		aggShare := thr.AllocateThresholdSecretShare()

		for j := range c.Participants {
			share := thr.AllocateThresholdSecretShare()
			thr.GenShamirSecretShare(p.ID, polynomials[j], &share)

			if j == 0 {
				aggShare = share
			} else {
				if err := thr.AggregateShares(aggShare, share, &aggShare); err != nil {
					return fmt.Errorf("failed to aggregate Shamir shares for participant %d: %w", p.ID, err)
				}
			}
		}

		c.Participants[i].shamirShare = aggShare
	}

	return nil
}

// genCollectivePublicKey runs the collective public key generation protocol (1 round).
func (c *Committee) genCollectivePublicKey() error {
	cpk := mhe.NewPublicKeyGenProtocol(c.params)
	roundCRS, err := derivePerRoundCRS(c.epochSeed, "cpk")
	if err != nil {
		return fmt.Errorf("derive CPK round CRS: %w", err)
	}
	crp := cpk.SampleCRP(roundCRS)

	// Each participant generates their share.
	shares := make([]mhe.PublicKeyGenShare, c.N)
	for i, p := range c.Participants {
		shares[i] = cpk.AllocateShare()
		cpk.GenShare(p.sk, crp, &shares[i])
	}

	// Aggregate all shares.
	aggregated := shares[0]
	for i := 1; i < c.N; i++ {
		cpk.AggregateShares(aggregated, shares[i], &aggregated)
	}

	// Generate the collective public key.
	c.CollectivePK = rlwe.NewPublicKey(c.params)
	cpk.GenPublicKey(aggregated, crp, c.CollectivePK)

	return nil
}

// genRelinearizationKey runs the collective relin key generation (2 rounds).
func (c *Committee) genRelinearizationKey() error {
	rkg := mhe.NewRelinearizationKeyGenProtocol(c.params)

	// Allocate shares and ephemeral keys for each participant.
	type rkgState struct {
		ephSK      *rlwe.SecretKey
		round1     mhe.RelinearizationKeyGenShare
		round2     mhe.RelinearizationKeyGenShare
	}
	states := make([]rkgState, c.N)
	roundCRS, err := derivePerRoundCRS(c.epochSeed, "relin")
	if err != nil {
		return fmt.Errorf("derive Relin round CRS: %w", err)
	}
	crp := rkg.SampleCRP(roundCRS)

	// Round 1: each participant generates ephemeral key + round 1 share.
	for i, p := range c.Participants {
		ephSK, r1, r2 := rkg.AllocateShare()
		rkg.GenShareRoundOne(p.sk, crp, ephSK, &r1)
		states[i] = rkgState{ephSK: ephSK, round1: r1, round2: r2}
	}

	// Aggregate round 1 shares.
	aggRound1 := states[0].round1
	for i := 1; i < c.N; i++ {
		rkg.AggregateShares(aggRound1, states[i].round1, &aggRound1)
	}

	// Round 2: each participant uses aggregated round 1 + their ephemeral key.
	for i, p := range c.Participants {
		rkg.GenShareRoundTwo(states[i].ephSK, p.sk, aggRound1, &states[i].round2)
	}

	// Aggregate round 2 shares.
	aggRound2 := states[0].round2
	for i := 1; i < c.N; i++ {
		rkg.AggregateShares(aggRound2, states[i].round2, &aggRound2)
	}

	// Generate the relinearization key.
	c.RelinKey = rlwe.NewRelinearizationKey(c.params)
	rkg.GenRelinearizationKey(aggRound1, aggRound2, c.RelinKey)

	return nil
}

// genGaloisKeys runs collective Galois key generation for all rotations
// needed by the dot product (powers of 2).
func (c *Committee) genGaloisKeys() error {
	gkg := mhe.NewGaloisKeyGenProtocol(c.params)

	// Generate Galois elements for power-of-2 rotations (same as crypto.galoisElements).
	logN := c.params.LogN()
	galEls := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		galEls[i] = c.params.GaloisElement(1 << i)
	}

	c.GaloisKeys = make([]*rlwe.GaloisKey, len(galEls))

	for idx, galEl := range galEls {
		// Each Galois element gets its own sub-round CRS so distinct rotation
		// keys never share a CRP — defense-in-depth on top of the per-round
		// derivation already separating Galois from CPK + Relin.
		roundCRS, err := derivePerRoundCRS(c.epochSeed, fmt.Sprintf("galois/%d", galEl))
		if err != nil {
			return fmt.Errorf("derive Galois round CRS for element %d: %w", galEl, err)
		}
		crp := gkg.SampleCRP(roundCRS)

		// Each participant generates their share for this rotation.
		shares := make([]mhe.GaloisKeyGenShare, c.N)
		for i, p := range c.Participants {
			shares[i] = gkg.AllocateShare()
			if err := gkg.GenShare(p.sk, galEl, crp, &shares[i]); err != nil {
				return fmt.Errorf("Galois key share gen failed for element %d: %w", galEl, err)
			}
		}

		// Aggregate shares.
		aggregated := shares[0]
		for i := 1; i < c.N; i++ {
			if err := gkg.AggregateShares(aggregated, shares[i], &aggregated); err != nil {
				return fmt.Errorf("Galois key aggregation failed: %w", err)
			}
		}

		// Generate Galois key.
		gk := rlwe.NewGaloisKey(c.params)
		if err := gkg.GenGaloisKey(aggregated, crp, gk); err != nil {
			return fmt.Errorf("Galois key gen failed: %w", err)
		}
		c.GaloisKeys[idx] = gk
	}

	return nil
}
