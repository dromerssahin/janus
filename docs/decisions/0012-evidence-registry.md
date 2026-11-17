# 0012: Manual fictional evidence registry

Status: accepted, 2026-11-10.

The initial registry is a versioned, manually curated YAML document. Stable IDs
and explicit links connect records to study and generator assumptions; manifests
retain the referenced records and registry fingerprint. The included records are
fictional and do not cite literature or represent empirical estimates. A source
locator is optional and must not contain credentials, restricted identifiers or
sensitive paths. Fingerprints establish content identity only.

Evidence can explain why a model assumption was chosen. It does not establish a
causal effect, clinical validity, privacy, or external validity. Automated
retrieval, extraction, review and real data governance remain outside this
milestone. Changes to a bound registry require a new fit or an explicitly updated
configuration; reproducibility still depends on the execution environment.
An alternative was to infer sources automatically from parameter names; explicit
links were chosen so that an association is never silently presented as support.
