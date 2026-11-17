# Decision 0050: validate protocol contracts before execution

The CLI exposes `validate-contract`, which validates a coherence contract and
optionally linked protocol and candidate-comparison manifests. Linked contract
IDs must agree. This is a preflight integrity check and does not process
patient data, approve source access, select a model, or establish validity.
