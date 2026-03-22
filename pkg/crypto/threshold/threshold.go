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
	"fmt"
	"sync"

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

	// crs is the common reference string shared by all participants.
	crs mhe.CRS
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

	// Create CRS from a random seed (in production, this seed is agreed upon out-of-band).
	crsSeed := make([]byte, 64)
	prng, err := sampling.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("failed to create PRNG: %w", err)
	}
	if _, err := prng.Read(crsSeed); err != nil {
		return nil, fmt.Errorf("failed to generate CRS seed: %w", err)
	}
	crs, err := sampling.NewKeyedPRNG(crsSeed)
	if err != nil {
		return nil, fmt.Errorf("failed to create CRS: %w", err)
	}

	c := &Committee{
		N:         n,
		Threshold: threshold,
		params:    params,
		crs:       crs,
	}

	// Step 1: Each participant generates a local secret key share.
	kgen := rlwe.NewKeyGenerator(params)
	c.Participants = make([]*Participant, n)
	for i := 0; i < n; i++ {
		c.Participants[i] = &Participant{
			ID:     mhe.ShamirPublicPoint(i + 1), // 1-indexed
			sk:     kgen.GenSecretKeyNew(),
			params: params,
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
func (c *Committee) ThresholdDecrypt(ct *rlwe.Ciphertext, clientPK *rlwe.PublicKey) (*rlwe.Ciphertext, error) {
	// No mutex needed: all state created per-call (pcks, shares, activeKeys).
	// Participant fields (shamirShare, sk) are read-only after setup.
	// Concurrent calls are safe and simulate real distributed deployment
	// where multiple ciphertexts are decrypted in parallel.

	// Noise flooding: sigma must mask key share info but leave headroom
	// after deep HE circuits (multiply + rescale + up to log2(maxSlots) rotations).
	// 2^20 provides ~20 bits of masking while preserving precision after full inner-sum.
	noiseFlood := ring.DiscreteGaussian{Sigma: 1 << 20, Bound: 6 * (1 << 20)}

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

// DecryptScalar decrypts a scalar value from a ciphertext using the client's session key.
func (s *ClientSession) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	pt := s.decryptor.DecryptNew(ct)
	decoded := make([]float64, 1)
	if err := s.encoder.Decode(pt, decoded); err != nil {
		return 0, fmt.Errorf("failed to decode: %w", err)
	}
	return decoded[0], nil
}

// DecryptBatchScalars decrypts multiple dot product results from a batch ciphertext.
func (s *ClientSession) DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error) {
	pt := s.decryptor.DecryptNew(ct)
	maxSlots := s.params.MaxSlots()
	decoded := make([]float64, maxSlots)
	if err := s.encoder.Decode(pt, decoded); err != nil {
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
func newParameters() (hefloat.Parameters, error) {
	return hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{60, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61, 61},
		LogDefaultScale: 45,
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
	crp := cpk.SampleCRP(c.crs)

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
	crp := rkg.SampleCRP(c.crs)

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
		crp := gkg.SampleCRP(c.crs)

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
