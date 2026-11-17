# 0039: In-memory external flat evaluation

`evaluate_flat_external` compares caller-provided real training, generated and
external reference rows using a common categorical schema. It reports marginal
and full joint total variation and smoothed binary utility on the same reference
patients. Missing values remain states in fidelity; missing and structural-absence
labels are rejected. Empty cohorts, invalid rows and invalid task declarations
are rejected. The caller is responsible for patient independence, timing,
source harmonization, data-use permissions and protecting inputs. The function
does not verify any of these or establish clinical, privacy or causal validity.

The example driver samples the existing fitted DAGs in memory and writes only
aggregate metrics and backoff counts; its caller must handle input and output
under the relevant policy. It provides no uncertainty intervals. A separate
study-specific preprocessing and validation step is needed for each dataset.
