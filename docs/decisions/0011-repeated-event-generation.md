# Configured repeated-event generation

Generator schema version 4 adds an opt-in repeated-event configuration for selected follow-up variables. Each episode samples a categorical count and emits that many values from the variable's existing root categorical distribution, with event and recording day equal and uniform inclusive timing inside the configured index-relative bounds. Counts are capped at 100. Existing generator versions and their random streams retain their behavior.

The generated history is an explicit configured assumption. It does not fit trajectories, model dependence between events, missingness, censoring, or treatment effects. Complete follow-up is declared only when requested; otherwise coverage remains unknown. Zero observed events do not establish absence. The fictional workflow uses longitudinal evaluation version 4, whose denominators and coverage states remain distinct.

Configuration validation rejects index repetition, unsupported variable roles/types, conditional repeated distributions, malformed probabilities/counts, invalid follow-up bounds, and invalid value distributions before CLI publication. Generated IDs are unique within each episode, and emitted events are sorted by event day then ID. Manifests retain configuration, seed, RNG description, and limitations.
