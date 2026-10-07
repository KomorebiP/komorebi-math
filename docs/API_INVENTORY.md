# API inventory

This thematic inventory began with the `0.14.0a2` coordinate release and remains a human navigation aid. For the frozen `0.16.0a2` contract, use `komorebi_math.stable`, `komorebi_math.v1`, or `komorebi_math.api.v1` together with `public_api_stable_0_16.json`. The source and committed manifests remain the authoritative signature references.

## `komorebi_math.core`

### Grammars and realizations

* `Arrow`
* `OperatorGrammar`
* `OperatorRelation`
* `OperatorWord`
* `OperatorRealization`

### Algebras and seed systems

* `OperatorAlgebra`
* `AlgebraBasisResult`
* `SeedSystem`
* `cyclic_span`
* `matrix_units`
* `semisimple_seed_rank`
* `single_operator_seed_rank`
* `shared_direct_sum_seed_bounds`
* `independent_direct_sum_seed_rank`
* `tensor_seed_rank_upper_bound`

### Packets and transfers

* `PacketSystem`
* `PacketCoordinates`
* `PacketOperator`
* `TransferGraph`
* `SupportPreorder`
* `block_from_paths`
* `path_contributions`

### Morphisms

* `OperatorMorphism`
* `PacketMorphism`
* `SeedMorphism`

## `komorebi_math.refinement`

### Towers and packet addresses

* `PacketRefinement`
* `RefinementTower`
* `common_refinement`
* `ResidueCoordinates`
* `residue_analyze`
* `residue_packet_system`
* `crt_pair`
* `crt_many`

### Weighted transfer

* `WeightedTransferEdge`
* `WeightedTransferGraph`

### Lattices and Smith congruences

* `LatticePacketData`
* `SmithCongruenceProblem`

### Linear defects

* `LinearRefinementDefect`

## `komorebi_math.nonlinear`

### Coordinates, defects, obstruction, lifting, and holonomy

* `NonlinearPacketMap`
* `NonlinearRefinementDefect`
* `LinearObstructionProblem`
* `LinearLiftProblem`
* `PolynomialLiftProblem`
* `FiniteLiftTree`
* `TransportEdge`
* `TransportGraph`
* `build_lift_tree`
* `solve_mod_prime`

### Derivative providers and composition

* `DerivativeProvider`
* `CallableDerivativeProvider`
* `PolynomialDerivativeProvider`
* `FiniteDifferenceDerivativeProvider`
* `SympyDerivativeProvider`
* `ComposedDerivativeProvider`
* `faa_di_bruno`
* `packet_faa_di_bruno`
* `set_partitions`
* `stirling_second`

### Packet jets and trees

* `PacketJet`
* `PacketJetAddress`
* `PacketJetFamily`
* `PacketJetSupport`
* `packet_jet_support`
* `packet_taylor`
* `OrderedRootedTree`
* `PolynomialFixedPointProblem`
* `majorant_convergence`
* `packet_tree_reconstruction`
* `ordered_rooted_trees`
* `ordered_trees_with_leaves`
* `catalan_number`

### Graded nonlinear operations

* `FiniteAbelianGroup`
* `GradedPacketOperation`
* `maurer_cartan_packets`
* `higher_maurer_cartan_packets`
* `divided_square_packets`

### Correction and elimination

* `HilbertObstructionProjector`
* `RightInverseCorrectionProblem`
* `LyapunovSchmidtReduction`
* `NonlinearSchurElimination`

### Filtered lifting and spectral sequences

* `FilteredVectorSpace`
* `FilteredCorrectionProblem`
* `FiniteFilteredCochainComplex`
* `verify_finite_filtered_branch`
* `conditional_projective_limit_completion`

### Shared smooth transport

* `MovingPacketSystem`
* `KatoTransport`
* `direct_packet_rotation`

The transport objects are re-exported from `komorebi_math.quantum` for backward compatibility.

## `komorebi_math.algorithms`

### Layout and packet IR

* `PacketLayout`
* `PacketBlock`
* `PacketIR`

### Structured operators

* `StructuredOperator`
* `DenseOperator`
* `IdentityOperator`
* `DiagonalOperator`
* `BlockDiagonalOperator`
* `CirculantOperator`
* `ToeplitzOperator`
* `KroneckerOperator`
* `WalshHadamardOperator`
* `as_structured_operator`

### Low rank algebra

* `LowRankOperator`
* `RecompressionResult`
* `BlockDiagonalLowRankOperator`
* `recompress_core`
* `recompress_factors`
* `product_chain_error_bound`
* `accumulated_recompression_bound`

### Support, Krylov, precision, compilation, and scheduling

The namespace includes structural support propagation, Krylov and conjugate gradient routines, block Jacobi certificates, continuous and discrete precision allocation, exact finite tree compilers, Pareto frontiers, task scheduling, complexity estimates, restricted structure discovery, NumPy execution, optional CuPy execution, and matched benchmark contracts.

## `komorebi_math.signals`

### Production finite DSP

* `WindowMetrics`, `get_window`, `window_metrics`, `overlap_profile`, `overlap_add_certificate`, `canonical_dual_window`
* `FrameBatch`, `STFTResult`, `frame_signal`, `stft`, `istft`, `spectrogram`, `overlap_add`
* `PowerSpectrum`, `CrossSpectrum`, `periodogram`, `welch_psd`, `cross_spectrum`, `parseval_certificate`
* `FIRFilter`, `FrequencyResponse`, `StreamingFIR`, `design_fir`, `design_lowpass`, `design_highpass`, `design_bandpass`, `design_bandstop`, `fir_filter`, `stream_fir_filter`
* `RationalResamplingPlan`, `ResamplingResult`, `upsample`, `downsample`, `interpolate`, `decimate`, `design_resample_filter`, `resample_poly`, `resample_poly_result`
* `AdaptiveFilterResult`, `lms_filter`, `nlms_filter`, `rls_filter`
* `HaarDecomposition`, `WaveletDenoiseResult`, `haar_decompose`, `wavelet_denoise`, `threshold_coefficients`, `denoising_risk_certificate`
* `StreamingSTFT`, `StreamingSTFTBatch`, `stream_stft`
* `LinearFilter`, `FilterOutput`, `StreamingLinearFilter`, `lfilter`, `stream_lfilter`, `design_biquad`
* `OverlapAddConvolver`, `fft_convolve`, `overlap_add_convolve`, `certified_overlap_add_convolve`


### Polyphase coordinates

* `PolyphaseCoordinates`
* `LatticePolyphaseCoordinates`
* `PolyphaseKernel`
* `PolyphaseSystem`
* `polyphase_analyze`
* `polyphase_synthesize`
* `polyphase_roundtrip_certificate`
* `lattice_polyphase_analyze`
* `lattice_polyphase_synthesize`
* `polyphase_filter_kernels`
* `mixed_radix_address`
* `split_mixed_radix_address`
* `refinement_children`
* `residue_project`

### Convolution

* `ConvolutionMode`
* `PolyphaseConvolution`
* `PolyphaseBlockMatrix`
* `FinitePolyphaseKernel`
* `convolve_signal`
* `convolution_matrix`
* `convolution_output_length`
* `polyphase_convolution`
* `finite_polyphase_blocks`
* `finite_polyphase_filter_kernels`

### Filter banks and wavelets

* `PeriodicFilterBank`
* `FiniteFilterBank`
* `HaarFilterBank`
* `PeriodicLiftingScheme`
* `FilterBankReport`
* `HaarPacketTree`
* `PrunedPacketCoefficients`
* `BestBasisResult`
* `haar_analyze`
* `haar_synthesize`
* `haar_matrix`
* `haar_filter_bank`
* `periodic_five_three_lifting`
* `best_basis`
* `entropy_cost`
* `lp_cost`
* `packet_energy`

### Singular packets and recovery

* `TruncatedPowerTerm`
* `TruncatedPowerPacket`
* `SingularPacket1D`
* `RetainedPacketReconstruction`
* `RetainedPacketResult`
* `JumpDetection`
* `JumpRecoveryResult`
* `VectorJumpRecoveryResult`
* `BreakpointBandResult`
* `retained_packet_reconstruct`
* `retained_packet_reconstruction`
* `estimated_packet_reconstruct`
* `recover_separated_jumps`
* `recover_jumps_subgaussian`
* `recover_vector_jumps`
* `recover_vector_jumps_subgaussian`
* `detect_piecewise_polynomial_breaks`
* `finite_difference`
* `scalar_region_error`
* `scalar_region_lp_error`
* `overlap_matrix_error`
* `overlap_partition_lp_error`
* `step_location_lp_error`

### Secondary signal tools

* cyclic coefficient packet products
* shifted nonlinear filters
* Fourier low pass approximation
* Otsu thresholding
* the deliberately narrow positive contrast packet estimator
* Smith and Cayley phase coordinates

## `komorebi_math.dynamics`

### Semigroups and refinement

* `MatrixSemigroup`
* `PacketSemigroup`
* `SemigroupRefinement`
* `FiniteSemigroupTower`
* `ProjectiveSemigroupTower`
* `matrix_exponential`
* `semigroup_refinement_defect`

### Markov and martingale systems

* `FiniteMarkovChain`
* `ContinuousTimeMarkovChain`
* `AggregationMap`
* `MarkovLumping`
* `ConditionalExpectation`
* `ConditionalExpectationTower`
* `MartingaleTower`
* `aggregation_matrix`
* `coordinate_lumpability_residual`
* `normalize_partition`
* `partition_refines`

### Bayesian models

* `BayesianUpdate`
* `BayesianPartition`
* `BayesianFlow`
* `HiddenMarkovFilter`
* `coarsest_sufficient_partition`
* `approximate_likelihood_partition`
* `relative_entropy`
* `relative_entropy_decomposition`

### Fractional and Green operators

* `FractionalDefectResult`
* `fractional_power`
* `fractional_semigroup`
* `fractional_defect`
* `fractional_naturality_certificate`
* `DirichletGreen1D`

### Feynman Kac, Feshbach, memory, and survival

* `FiniteFeynmanKac`
* `FeshbachReduction`
* `MemoryKernel`
* `RationalMemoryKernel`
* `RationalMemoryRealization`
* `memory_kernel_error_bound`
* `kernel_l1_error`
* `volterra_trajectory_bound`
* `PhaseTypeSurvival`

## Infrastructure

### Certificates and configuration

* `Certificate`
* `ComputationResult`
* `ConditionalResult`
* `Exactness`
* `Settings`
* `configure`
* `get_settings`
* `settings_context`

### Registries

* `FeatureSpec`
* `FeatureStatus`
* `feature`
* `features`
* `ClaimStatus`
* `ProvenanceRecord`
* `describe`

### Unified runtime

* `UnifiedStage`
* `StageSpec`
* `UnifiedSystem`
* `STAGES`

## `komorebi_math.analytic`

### Cyclic coordinates and sections

* `CyclicSectionFamily`
* `cyclic_project_coefficients`
* `cyclic_packets`
* `cyclic_reconstruct`
* `cyclic_project_value`
* `cyclic_product_packets`
* `cyclic_projection_certificate`
* `transfer_section`
* `signed_section`
* `transfer_sections`
* `signed_sections`
* `denominator`
* `denominator_root_sum`
* `denominator_derivative`
* `transfer_shift_matrix`
* `signed_shift_matrix`
* `cyclic_addition`
* `section_certificate`

### Reciprocal and polynomial families

* `logarithmic_coefficients`
* `reciprocal_coefficients`
* `reciprocal_polynomial`
* `reciprocal_value`
* `reciprocal_semigroup_certificate`
* `transfer_tangent`
* `transfer_tangent_coefficients`
* `congruential_inverse`
* `congruential_reciprocal_coefficients`
* `appell_polynomial`
* `appell_coefficients`
* `appell_polynomial_symbolic`
* `theta_polynomial`
* `theta_polynomial_symbolic`

### Zero diagnostics

* `CyclicDenominator`
* `ZeroStringResult`
* `asymptotic_positive_zero`
* `positive_zero_string`
* `zero_count_asymptotic`
* `hadamard_partial_product`
* `reciprocal_coefficient_from_zeros`
* `reduced_zero_zeta`

## `komorebi_math.quantum`

### Seeds, symmetry, and circuits

* `QuantumSeedSystem`
* `schmidt_rank`
* `local_orbit_dimension`
* `observable_seed_rank`
* `computational_basis_generation`
* `finite_group_projector`
* `cyclic_symmetry_projectors`
* `charge_certificate`
* `charge_selection_certificate`
* `total_charge_packets`
* `symmetry_projector_certificate`
* `CircuitPacketExpansion`
* `CircuitPathAmplitude`
* `InterferenceReport`

### Moving packets and leakage

* `LeakageResult`
* `fixed_packet_leakage`
* `MovingPacketSystem`
* `KatoTransport`
* `DirectRotationResult`
* `direct_packet_rotation`
* `moving_geometric_coupling`
* `moving_packet_leakage_bound`

### Channels and error correction

* `ChannelPacketBlock`
* `QuantumChannel`
* `amplitude_damping`
* `operator_packets`
* `reconstruct_operator_packets`
* `dephase`
* `LindbladGenerator`
* `QuantumCode`
* `KnillLaflammeReport`

### Effective dynamics and finite geometry

* `QuantumFeshbachReduction`
* `EuclideanTransfer`
* `heat_trace`
* `spectral_dimension`
* `periodic_lattice_eigenvalues`
* `periodic_laplacian`
* `low_mode_dispersion_bound`
* `ConstraintClosureReport`
* `constraint_closure`
* `transported_constraint_defect`
* `BoundedHamiltonianTower`
* `HamiltonianTowerReport`

## Arithmetic and sequences

Block 6 adds 78 arithmetic public names and 36 sequence public names. The inventory covers exact local domains, valuations, Hurwitz convolution, standard windows, Euler coordinates, Mahler transfer, recurrence packets, lazy sequences, automatic models, holonomic recurrences, transforms, and compilation.

## Deformation

Block 7 adds 32 public deformation names covering lift forests, visibility, counting charts, finite cochain complexes, packet cohomology, Fitting fibers, nested occupancy, Frobenius spines, companion forests, and compatible carriers.

## Local Langlands finite models

Block 7 adds 24 public local names covering finite fields, tame Frobenius packets, monodromy chains, Gauss sums, Heisenberg models, alternating forms, quadratic refinements, projective envelopes, and finite bridges.

## Finite image, ramification, and completion

The finite image namespace exports explicit groups, subgroups, linear characters, induced representations, orbit packets, Mackey packets, Clifford towers, conditional induction plans, and virtual character resolutions. The ramification namespace exports finite ramification tower diagnostics. The global and completion namespaces export finite Hecke packets, support comparisons, patching reports, dyadic sign diagnostics, five gate completion reports, and compatible propagation reports.


## Portable exports and LaTeX

Block 8 adds public document blocks, serialization adapters, JSON and tabular writers, text and web renderers, LaTeX renderers, LaTeX compiler discovery, PDF compilation, manifested export bundles with normalized ZIP metadata, and artifact certificates. See `EXPORT_AND_LATEX.md` for complete examples.

## Block 10 additions

The Block 10 public inventory adds sparse derivative tensors and providers, lazy directional automatic-differentiation adapters, packet-jet execution plans, provider-independent snapshots, CuPy runtime diagnostics, versioned portable persistence, and structural API compatibility tools. The retained Block 10 manifest is `src/komorebi_math/data/public_api_0_10.json`. The retained Block 11 research manifest is `src/komorebi_math/data/public_api_0_11.json`.

Preferred public CuPy names use `CuPy`. Existing implementation spellings beginning with `Cupy` remain aliases inside the nonlinear modules and are not the preferred documented surface.

## Block 11 additions

Block 11 adds directory-backed packet-jet streams, resumable live capture, bounded snapshot chunking, adjacent portable-codec migrations, a closed behavioral golden fixture, exact finite complexity counts, environment-scoped runtime profiles, and the curated `komorebi_math.beta` compatibility façade.

The retained Block 11 research inventory is `src/komorebi_math/data/public_api_0_11.json`. The current broader inventory is `src/komorebi_math/data/public_api_0_14.json`. The strict beta inventory is `src/komorebi_math/data/public_api_beta_0_14.json`. Structural manifests are supplemented by the committed semantic fixture `src/komorebi_math/data/behavioral_golden_0_14.json`.

## `komorebi_math.generating`

### Bases and coefficient domains

* `WeightedMonomialBasis`, `BasisDescriptor`, `FiniteNormalizedSeries`, `FormalSeries`
* `IntegerDomain`, `RationalDomain`, `ResidueRingDomain`, `PrimeFieldDomain`, `TruncatedPadicDomain`
* `ordinary_basis`, `exponential_basis`, `factorial_power_basis`, `q_factorial_basis`, `fibonacci_factorial_basis`, `pochhammer_basis`

### Custom sources and recognition

* `CustomFunctionSpec`, `FunctionRegistry`
* recurrence, rational, hypergeometric, algebraic, Euler-product, composition, and factorial-power Bell constructors
* bounded recurrence, P-recursive, hypergeometric, algebraic, and quasipolynomial recognizers

### Arithmetic compilers

* `FiniteMaskCarryPlan`, `CartierLinearRepresentation`, `CartierAutomaton`
* `mahler_mask_product`, `mahler_mask_inverse`, `compile_finite_mask_carry_plan`, `compile_digit_cauchy_product`
* `compile_cartier_linear_representation`, `compile_rational_mask_automaton`
* `KummerLayerDistribution`, `PrimePowerUnitCompiler`, `PrimePowerKernelCompiler`
* `BellParityAutomaton`, `BellResidueAutomaton`
* `q_binomial_valuation_decomposition`, `q_binomial_support`
* cyclotomic collapse, trusted snapshots, and finite signal adapters

## `komorebi_math.coordinates`

Block 14 adds a general coordinate atlas with arbitrary finite bases and subspaces, path composition and commuting diagrams, normalized-series changes, combinatorial transforms, classical and custom polynomial families, recurrence and state-space coordinates, affine and formal charts, tensor and operator transport, prime-field maps, spectral and modal transforms, packet permutations, resource budgets, and trusted portable records. Euler-to-Fibonacci conversion remains one named bridge inside the general system.
