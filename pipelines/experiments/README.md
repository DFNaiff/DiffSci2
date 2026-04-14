# pipelines/experiments/

Exploratory extensions that are adjacent to the main pipeline but not in
main.tex. Kept here instead of in `field_controlled/` to keep that
directory focused on the canonical paper pipeline.

| Script                          | Purpose                                                                 | Legacy                                                  |
|---------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------|
| `copula_field_estimator.py`     | Alternative field model (histogram / logit / beta warping + GP copula). | `scripts/legacy/0002-porosity-field-estimator-copula.py`|
| `enhanced_conditioning.py`      | FiLM + multi-scale conditioning post-training wrapper.                  | `scripts/legacy/0003b-porosity-field-training-enhanced.py`|
| `field_from_real.py`            | Generate conditioned on the real porosity field (not GP-sampled).       | `scripts/legacy/0006-porosity-field-generator-from-training.py`|
| `multistone_2d.py`              | 2D unconditional training on a mixture of stones (ergodicity study).    | `scripts/legacy/0008-unconditional-2d-multistone-training.py`|
| `unfolding.py`                  | Mirror/unfold generated volumes to study border behavior.               | `scripts/legacy/0012-unfolding.py`                      |
| `two_phase_flow.py`             | Buckley-Leverett solver + Corey fit + oil-water drainage simulation.    | `scripts/legacy/0005d-porosity-field-buckley-leverett.py` + `0011-oil-water-flow.py`|

All of these are currently thin delegators to the legacy scripts — they
exist to (a) give the new `pipelines/` layout a place for each piece of
exploratory work, and (b) keep the code findable. Port their CLIs to
match the rest of the pipelines when you need them for a new run.
