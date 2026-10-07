# Behavioral fixtures and finite profiling

## Closed behavioral fixtures

`run_behavioral_golden_fixture` executes a fixed registry of case kinds. A fixture cannot name a module, class, or callable to import. Unknown case kinds are rejected.

This closed evaluator prevents the regression fixture from becoming an arbitrary code-execution format. Inputs and expected values are explicit JSON data. Exact rational values use canonical numerator and denominator strings.

```python
from komorebi_math.golden import run_behavioral_golden_fixture

report = run_behavioral_golden_fixture()
report.certificate.require()
```

The report records the fixture path, fixture digest, observed values, expected values, and a certificate for every case.

Changing an expected value is not a routine test update. It is a semantic compatibility event and should be reviewed with the corresponding implementation and mathematical contract.

## Exact structural complexity counts

`estimate_packet_jet_complexity` reports deterministic counts for one finite family and support record:

* total family addresses
* active and pruned addresses
* naive and planned provider calls
* projected input vectors
* output projections
* dense standard-basis evaluations
* dense tensor elements
* derivative-call reduction ratio

These values are operation counts derived from the selected finite structure. They are not elapsed-time predictions.

```python
support = packet_jet_support(family)
estimate = estimate_packet_jet_complexity(family, support)
estimate.certificate.require()
```

## Observational runtime profiles

`profile_operation` and `profile_packet_jet_runtime` record local observations using `perf_counter_ns` and `tracemalloc`.

```python
profile = profile_packet_jet_runtime(
    family,
    directions,
    repetitions=5,
    include_snapshot=True,
    include_serialization=True,
)
```

A profile includes:

* Python and platform metadata
* operation scope
* repetition and warm-up counts
* minimum, median, and maximum duration
* maximum traced peak bytes
* exact structural counts beside measured stages

The timing is valid only for the recorded process and workload. It does not establish a portable speedup, a backend comparison, or a hardware-independent complexity theorem. `tracemalloc` observes Python-managed allocations and does not measure every native, framework, or device allocation.

## Device validation boundary

`verification/block11_device_validation.py` performs matched CPU and CuPy snapshot evaluations when a usable CUDA runtime exists. Without such a runtime it exits with a not-executed result unless `--allow-unavailable` is explicit.

A workflow definition is not execution evidence. Device claims require the generated validation record from an actual CUDA runner.
