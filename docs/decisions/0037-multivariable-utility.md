# 0037: Joint-state predictive utility

Binary flat utility tasks may add categorical predictors to the original
single predictor. The predictor state is the ordered tuple of raw tokens;
conditional binary frequencies are smoothed with the existing alpha. Unseen
validation tuples use the smoothed overall training rate and are counted, with
no fitted tables or individual predictions emitted. The existing configuration
without additional predictors retains its meaning. Predictors must be distinct
from each other and the categorical target. The existing timing declaration
applies to all predictors; flat records do not verify their availability.

Full joint-state tables can be sparse and do not generalize across similar
states. The synthetic and real training sizes may differ. Scores compare
predictive performance on held-out inner patients, not clinical, causal, privacy
or external validity. Overlapping development splits are not independent tests.
