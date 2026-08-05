#!/usr/bin/env python
"""
Idempotent, defensive patches to the tutorial notebooks.

Run from this directory with the notebooks CLOSED in Jupyter (or reload them from
disk afterwards):

    python apply_fixes.py            # report what would change
    python apply_fixes.py --write    # apply

Each patch matches a distinctive anchor string. If the anchor is missing (already
patched, or the cell was edited by hand) the patch is skipped and reported, never
forced. Cells whose code changes have their stale outputs cleared, so it is obvious
they need re-running.

--------------------------------------------------------------------------------
FIX 1 -- 02, denoising_history: y of shape [1] collapses to a 0-dim tensor
--------------------------------------------------------------------------------
`denoising_history` calls `module.get_denoised_estimate` directly, which bypasses the
`dict_unsqueeze(y, 0)` that `SIModule.sample()` applies. `ScalarEmbedder.forward`
squeezes a trailing dimension of size 1, so a bare `[1]` becomes 0-dim and raises
`ValueError: Invalid dimensions: 0`. The conditioning must be passed as `[B, 1]`,
which is the form the embedder's own docstring documents.

--------------------------------------------------------------------------------
FIX 2 -- 01 section 9: the coarse-to-fine comparison proved nothing
--------------------------------------------------------------------------------
The original section compared an exact 64^2 draw against a 32^2-coarse draw, i.e. a
refinement ratio of only 2, and the printed numbers came out 92% vs 94% of the target
variance -- the coarse draw looking *better*, contradicting the text. The effect is
real but only appears at higher ratios, so the section is replaced by a sweep over
N/N_c with the exact draw (N_c = N) as a built-in reference:

    N_c   ratio   variance   % of theta^2
      4    16.0     0.0672      49%
      8     8.0     0.0938      68%
     16     4.0     0.1044      76%
     32     2.0     0.1435     104%
     64     1.0     0.1251      91%   <- exact; the residual gap is finite-sample noise

Notebook 04 generates at latent 144 with N_c = 32, a ratio of 4.5 -- so about a quarter
of the field variance is lost before the diffusion model sees the field.
"""
import argparse
import json
import sys

PATCHES = []
REPLACEMENTS = []


def replace_cell(notebook, anchor, new_source, description, unless=None):
    """Register a whole-cell replacement, matched by `anchor` substring."""
    REPLACEMENTS.append((notebook, anchor, new_source, description, unless))


def patch(notebook, anchor, description, unless=None):
    """Register a cell-source patch.

    anchor : substring that identifies the target cell
    unless : substring whose presence means the patch is already applied
             (needed when the anchor survives patching)
    """
    def register(fn):
        PATCHES.append((notebook, anchor, description, fn, unless))
        return fn
    return register


# ---------------------------------------------------------------- FIX 1: nb 02
@patch('02-scalar-porosity-pretraining.ipynb',
       "torch.tensor([porosity_value], device=DEVICE)",
       "denoising_history: pass conditioning as [1, 1], not [1]")
def fix_denoising_history(source):
    old = "    y = {'porosity': torch.tensor([porosity_value], device=DEVICE)}\n"
    new = ("    # Shape [1, 1], not [1]: we are calling the denoiser directly, so we bypass\n"
           "    # the dict_unsqueeze that SIModule.sample() applies. ScalarEmbedder squeezes\n"
           "    # a trailing dim of size 1, and a bare [1] would collapse to a 0-dim tensor.\n"
           "    y = {'porosity': torch.tensor([[porosity_value]], device=DEVICE)}\n")
    return [new if ln == old else ln for ln in source]


# ---------------------------------------------------------------- FIX 3: nb 00
@patch('00-vae-training-2d.ipynb',
       'fc.savefig(fig, "00_size_generalisation")',
       "size-generalisation figure: stop row-2 titles colliding with row-1 images",
       unless="room for the row-2 titles")
def fix_size_fig_spacing(source):
    old = "fig.tight_layout()\n"
    new = "fig.tight_layout()\nfig.subplots_adjust(hspace=0.22)   # room for the row-2 titles\n"
    out, done = [], False
    for ln in source:
        if ln == old and not done:
            out.append(new)
            done = True
        else:
            out.append(ln)
    return out


# ---------------------------------------------------------------- FIX 2: nb 01
SECTION9_MD = r"""---
## 9. What the coarse-to-fine shortcut costs

Linear interpolation cannot create variability at scales finer than the coarse grid
spacing, so a coarsely-sampled field is **too smooth** and its variance **too low**. The
thesis names this as one of two reasons the generated field variance is under-reproduced
(the other being that the diffusion model only partially follows the field).

How much it costs depends entirely on the **refinement ratio** $N/N_c$ — how much work
interpolation is being asked to do. So rather than assert it, we sweep it.

The sweep has a free reference built in: when $N_c = N$ the interpolant is evaluated at
its own sample points, so it reproduces the **exact** GP draw. Note that even the exact
draw does not hit $\theta^2$ on the nose, because a $64^2$ window spans only a handful of
correlation lengths — that residual gap is finite-sample noise, not approximation error,
and it sets the scale below which differences here mean nothing."""

SECTION9_CODE = r"""n_cmp = 16
N = LATENT_SHAPE[0]
coarse_values = [4, 8, 16, 32, N]        # N_c = N is the exact draw

draws = {cn: np.stack([sample_field(sampler, LATENT_SHAPE, coarse_n=cn)[1]
                       for _ in range(n_cmp)])
         for cn in coarse_values}

print(f"target grid {N}^2, target variance theta^2 = {var_logit:.4f}\n")
print(f"{'N_c':>5} {'ratio N/N_c':>12} {'variance':>10} {'% of target':>12}")
for cn in coarse_values:
    v = draws[cn].var()
    tag = '   <- interpolation is exact here' if cn == N else ''
    print(f"{cn:>5} {N / cn:>12.1f} {v:>10.4f} {100 * v / var_logit:>11.0f}%{tag}")

GEN_LATENT, GEN_COARSE = 144, fc.COARSE_N      # what notebook 04 actually uses
print(f"\nNotebook 04 generates a latent {GEN_LATENT}^2 field with N_c = {GEN_COARSE}, "
      f"a refinement ratio of {GEN_LATENT / GEN_COARSE:.1f}.")
print("Read the deficit off the table at that ratio, not at ratio 1 or 2 "
      "(where it is within sampling noise).")"""

SECTION9_FIG = r"""fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

vmin, vmax = draws[N][0].min(), draws[N][0].max()
for ax, cn in zip(axes[:2], [N, 8]):
    ax.imshow(draws[cn][0], cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f"$N_c$ = {cn}   (ratio {N // cn})" + ("  — exact" if cn == N else ""))
    ax.set_xticks([]); ax.set_yticks([])

colors = {4: 'C3', 8: 'C1', 16: 'C0', 32: 'C2', N: 'k'}
for cn in coarse_values:
    _, c_cn, _ = radial(draws[cn], mean_logit)
    axes[2].plot(rr, c_cn, color=colors[cn], lw=2, ls='-' if cn == N else '--',
                 label=f"$N_c$ = {cn}" + (" (exact)" if cn == N else ""))
axes[2].plot(rr, c_real, color='0.55', lw=3, alpha=0.8, zorder=0, label='real rock')
axes[2].set_xlabel("lag $r$ [latent units]"); axes[2].set_ylabel("$C(r)$")
axes[2].set_title("coarser grid $\\to$ smoother field")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

ratios = [N / cn for cn in coarse_values]
fracs = [100 * draws[cn].var() / var_logit for cn in coarse_values]
axes[3].plot(ratios, fracs, 'o-', color='C0', lw=2)
axes[3].axhline(100, color='k', ls='--', lw=1, label='target $\\theta^2$')
axes[3].axvline(GEN_LATENT / GEN_COARSE, color='C3', ls=':', lw=2,
                label=f'notebook 04 ({GEN_LATENT / GEN_COARSE:.1f})')
axes[3].set_xscale('log', base=2)
axes[3].set_xlabel("refinement ratio $N / N_c$"); axes[3].set_ylabel("variance [% of $\\theta^2$]")
axes[3].set_title("the deficit, as a function of how\nmuch interpolation must invent")
axes[3].legend(fontsize=9); axes[3].grid(alpha=0.3)

fig.tight_layout()
fc.savefig(fig, "01_coarse_to_fine_cost")
plt.show()"""

SECTION9_SUMMARY = r"""The honest summary, read off the table rather than assumed:

- at ratio 1–2 the shortcut costs **nothing measurable** — the differences there are
  sampling noise, so a comparison run at those ratios proves nothing either way;
- the deficit becomes real from ratio $\approx 4$ onward and grows steeply;
- at the ratio notebook 04 actually generates with ($144/32 \approx 4.5$) roughly a
  **quarter of the field variance is lost** before the diffusion model even sees it.

So the conditioning field we feed the model is genuinely smoother than the real thing, and
we know by how much. Notebook 04 shows the downstream consequence: the generated field
tracks the input field, and therefore inherits its excess smoothness. Fixing this —
circulant embedding to generate the fine-grid residual with the correct spectral content —
is listed as future work in the thesis and would not disturb anything else in the pipeline.

Note the practical corollary: the fix is cheap to approximate by **raising $N_c$**. The
exact-sampling cost is $\mathcal{O}(N_c^6)$ in 2D, so there is a real ceiling, but the
table says where the useful range starts."""


# ---------------------------------------------------------------- FIX 4: nb 03
# The realised porosity field of a generated image was computed as
# avg_pool2d(1 - binary, F) -- the pore fraction of each F x F block, i.e. a
# *window-8* field. The conditioning field is a *window-129* local average, then
# pooled by F. These are different quantities. Self-test on real rock, where the
# realised field of the real image must reproduce its own conditioning field:
#
#     avg_pool8(binary)       rho = +0.18   std = 0.333
#     realised_field(binary)  rho = +1.000  std = 0.0586  (= conditioning)
#
# So the old metric was dominated by block noise and capped rho at ~0.18 even for
# a perfect generator. The canonical definition is in
# scripts/old/0007-porosity-field-evaluation.py:
#     calculate_porosity_field_full(volume, kernel_size) -> average_volume(field, F)

_NB03_GENERATE = '''field_model = fc.load_flow_model(CKPT, conditional=True)
module = fc.make_si_module(field_model, autoencoder=fc.load_vae()).to(DEVICE).eval()


@torch.no_grad()
def generate_with_field(module, phi_latent, nsteps=fc.NSTEPS, seed=None):
    """phi_latent: [h, w] numpy in (0,1)  ->  binary image of shape [8h, 8w]."""
    if seed is not None:
        torch.manual_seed(seed)
    y = {'porosity': torch.tensor(phi_latent, dtype=torch.float32)}
    img = module.sample(1, shape=[fc.Z_DIM, *phi_latent.shape], y=y, nsteps=nsteps,
                        is_latent_shape=True, return_latents=False, guidance=1.0)
    arr = img[0, 0].cpu().numpy()
    return (arr > arr.mean()).astype(np.float32)


def latent_field_of(binary, window=fc.WINDOW, factor=fc.F):
    """Realised porosity field of a binary image, on the latent grid.

    This MUST use the same definition of "porosity field" as the conditioning:
    a local average over a window of `window` voxels, and only then pooled by F.

    Pooling the binary image straight down by F instead gives the pore fraction of
    each F x F block -- a window-8 field. That is a different quantity, and not a
    small difference: on real rock it correlates with the true window-129 field at
    only rho ~ 0.18, so it would cap the measured tracking of *any* generator.
    The self-test in the next cell checks this rather than assuming it.

    Mirrors scripts/old/0007-porosity-field-evaluation.py:
        calculate_porosity_field_full(volume, w) -> average_volume(field, F)
    """
    return fc.realised_field(binary, window=window, factor=factor)


# A real conditioning field taken from a held-out slice.
SIZE = 512
ls = SIZE // fc.F
v = 0
phi_real = val_phi[v][:SIZE, :SIZE]
phi_cond = fc.to_latent(phi_real)
ref_binary = val_bin[v][:SIZE, :SIZE].astype(np.float32)

t0 = time.time()
gen = generate_with_field(module, phi_cond, seed=11)
print(f"generated {gen.shape} in {time.time() - t0:.1f}s")

phi_gen = latent_field_of(gen)

# The 'same'-mode field averaged over fewer than w^2 real voxels within R of the
# border, so every comparison below uses the interior only.
phi_cond_i, phi_gen_i = fc.field_interior(phi_cond), fc.field_interior(phi_gen)
rho = np.corrcoef(phi_cond_i.ravel(), phi_gen_i.ravel())[0, 1]
print(f"requested phi : mean {phi_cond_i.mean():.4f}  std {phi_cond_i.std():.4f}")
print(f"realised  phi : mean {phi_gen_i.mean():.4f}  std {phi_gen_i.std():.4f}")
print(f"voxel-wise field correlation rho = {rho:.3f}")'''

_NB03_SELFTEST = '''# Self-test: feed the REAL image through latent_field_of. Since the stored field was
# computed from that same image, the result must reproduce it -- rho = 1. This is the
# check that catches a mismatched field definition.
_check = latent_field_of(ref_binary)
_a, _b = fc.field_interior(phi_cond), fc.field_interior(_check)
print(f"realised field of the REAL image vs its own conditioning field:")
print(f"  rho = {np.corrcoef(_a.ravel(), _b.ravel())[0, 1]:+.4f}   "
      f"max|diff| = {np.abs(_a - _b).max():.2e}      <- must be 1.0000")

# What the naive alternative would have given:
_wrong = F.avg_pool2d(torch.from_numpy(1.0 - ref_binary)[None, None], fc.F, fc.F)[0, 0].numpy()
_w = fc.field_interior(_wrong)
print(f"  avg_pool{fc.F}(binary) instead: rho = {np.corrcoef(_a.ravel(), _w.ravel())[0, 1]:+.4f}, "
      f"std {_w.std():.4f} vs {_a.std():.4f}  <- a window-{fc.F} field, not window-{fc.WINDOW}")'''

_NB03_FIG4 = '''fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
vmin, vmax = min(phi_cond_i.min(), phi_gen_i.min()), max(phi_cond_i.max(), phi_gen_i.max())

axes[0].imshow(ref_binary, cmap='gray_r', interpolation='nearest')
axes[0].set_title(f"real rock (held-out slice)\\n$\\\\phi$ = {1 - ref_binary.mean():.3f}")

im = axes[1].imshow(phi_cond_i, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[1].set_title("conditioning field (from the real rock)")
fig.colorbar(im, ax=axes[1], fraction=0.046)

axes[2].imshow(gen, cmap='gray_r', interpolation='nearest')
axes[2].set_title(f"generated\\n$\\\\phi$ = {1 - gen.mean():.3f}")

im = axes[3].imshow(phi_gen_i, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[3].set_title(f"realised field of the generated image\\n$\\\\rho$ = {rho:.3f}")
fig.colorbar(im, ax=axes[3], fraction=0.046)

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Conditioning on a real measured field "
             f"(fields shown on the interior, window $w$ = {fc.WINDOW})", y=1.03)
fig.tight_layout()
fc.savefig(fig, "03_conditioning_on_real_field")
plt.show()'''

_NB03_FIG_TRACK = '''fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].plot(phi_cond_i.ravel(), phi_gen_i.ravel(), '.', ms=2.5, alpha=0.35, color='C0')
lims = [vmin, vmax]
axes[0].plot(lims, lims, 'k--', lw=1.2)
axes[0].set_xlabel("requested $\\\\phi$"); axes[0].set_ylabel("realised $\\\\phi$")
axes[0].set_title(f"voxel-wise, $\\\\rho$ = {rho:.3f}"); axes[0].grid(alpha=0.3)

mid = phi_cond_i.shape[0] // 2
axes[1].plot(phi_cond_i[mid], 'k-', lw=2, label='requested')
axes[1].plot(phi_gen_i[mid], 'C3-', lw=1.6, label='realised')
axes[1].set_xlabel("position [latent units]"); axes[1].set_ylabel("$\\\\phi$")
axes[1].set_title("mid-line profile"); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
fc.savefig(fig, "03_field_tracking")
plt.show()'''

_NB03_RAMP = '''# A synthetic field: a smooth diagonal ramp between two porosity levels.
yy, xx = np.meshgrid(np.linspace(0, 1, ls), np.linspace(0, 1, ls), indexing='ij')
phi_ramp = (0.10 + 0.18 * (0.5 * (xx + yy))).astype(np.float32)

gen_ramp = generate_with_field(module, phi_ramp, seed=5)
phi_ramp_out = latent_field_of(gen_ramp)
ramp_in_i, ramp_out_i = fc.field_interior(phi_ramp), fc.field_interior(phi_ramp_out)
rho_ramp = np.corrcoef(ramp_in_i.ravel(), ramp_out_i.ravel())[0, 1]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
im = axes[0].imshow(ramp_in_i, cmap='viridis', interpolation='nearest')
axes[0].set_title("requested: a synthetic ramp"); fig.colorbar(im, ax=axes[0], fraction=0.046)
axes[1].imshow(gen_ramp, cmap='gray_r', interpolation='nearest')
axes[1].set_title(f"generated  $\\\\phi$ = {1 - gen_ramp.mean():.3f}")
im = axes[2].imshow(ramp_out_i, cmap='viridis', interpolation='nearest')
axes[2].set_title(f"realised field, $\\\\rho$ = {rho_ramp:.3f}"); fig.colorbar(im, ax=axes[2], fraction=0.046)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("The conditioning is genuinely being used", y=1.03)
fig.tight_layout()
fc.savefig(fig, "03_synthetic_ramp_control")
plt.show()'''

_NB03 = '03-field-controlled-training.ipynb'
replace_cell(_NB03, 'def latent_field_of', _NB03_GENERATE,
             'latent_field_of: window-w field, then pool (was pooling the binary)',
             unless='fc.realised_field')
replace_cell(_NB03, '03_conditioning_on_real_field', _NB03_FIG4,
             'conditioning figure: use interior fields', unless='phi_cond_i')
replace_cell(_NB03, '03_field_tracking', _NB03_FIG_TRACK,
             'tracking figure: use interior fields', unless='phi_cond_i')
replace_cell(_NB03, '03_synthetic_ramp_control', _NB03_RAMP,
             'ramp control: use interior fields', unless='ramp_in_i')


# ------------------------------------------------- ADDITION: nb 03, section 5b
# Pretrained (scalar-conditioned) vs post-trained (field-conditioned), both fed the
# SAME field with the SAME seed. Generation case 4 vs case 5 of 0004e.

_NB03_CMP_MD = r"""### Was the fine-tuning necessary? — pretrained vs post-trained

The architecture never changed, so the **pretrained** scalar-conditioned model from
notebook 02 accepts a $32^2$ field just as happily as the fine-tuned one does. What it
never saw is training data in which the conditioning *varied across the image*: every
example it was shown had a single porosity value covering the whole patch.

That lets us isolate the effect of the fine-tune exactly. Same conditioning field, same
random seed — hence the same initial noise — same sampler, same number of steps. **Only
the weights differ.**

This is the pair of generation cases the thesis compares in
`0004e-porosity-field-generator.py`: **case 4** (field conditioning, scalar-trained model)
against **case 5** (field conditioning, field-trained model)."""

_NB03_CMP_GEN = '''# The pretrained model, straight from notebook 02's checkpoint.
pre_model = fc.load_flow_model(SCALAR_CKPT, conditional=True)
pre_module = fc.make_si_module(pre_model, autoencoder=fc.load_vae()).to(DEVICE).eval()

# Same fields and same seeds as the post-trained runs above, so the only difference
# between the two rows of the figure below is the weights.
t0 = time.time()
gen_pre = generate_with_field(pre_module, phi_cond, seed=11)
gen_ramp_pre = generate_with_field(pre_module, phi_ramp, seed=5)
print(f"pretrained generations in {time.time() - t0:.1f}s")

phi_gen_pre_i = fc.field_interior(latent_field_of(gen_pre))
ramp_out_pre_i = fc.field_interior(latent_field_of(gen_ramp_pre))

rho_pre = np.corrcoef(phi_cond_i.ravel(), phi_gen_pre_i.ravel())[0, 1]
rho_ramp_pre = np.corrcoef(ramp_in_i.ravel(), ramp_out_pre_i.ravel())[0, 1]

print(f"\\n{'model':<24} {'rho (real field)':>17} {'rho (ramp)':>12} "
      f"{'realised std':>13} {'mean phi':>10}")
print(f"{'requested':<24} {'--':>17} {'--':>12} "
      f"{phi_cond_i.std():>13.4f} {phi_cond_i.mean():>10.4f}")
print(f"{'pretrained (scalar)':<24} {rho_pre:>17.3f} {rho_ramp_pre:>12.3f} "
      f"{phi_gen_pre_i.std():>13.4f} {phi_gen_pre_i.mean():>10.4f}")
print(f"{'post-trained (field)':<24} {rho:>17.3f} {rho_ramp:>12.3f} "
      f"{phi_gen_i.std():>13.4f} {phi_gen_i.mean():>10.4f}")'''

_NB03_CMP_FIG = '''cases = [
    ("measured field\\n(held-out rock)", phi_cond_i, gen_pre, phi_gen_pre_i, gen, phi_gen_i,
     rho_pre, rho),
    ("synthetic ramp", ramp_in_i, gen_ramp_pre, ramp_out_pre_i, gen_ramp, ramp_out_i,
     rho_ramp_pre, rho_ramp),
]

fig, axes = plt.subplots(2, 5, figsize=(20, 8.6))
for r, (name, phi_in, g_pre, f_pre, g_post, f_post, r_pre, r_post) in enumerate(cases):
    vmin = min(phi_in.min(), f_pre.min(), f_post.min())
    vmax = max(phi_in.max(), f_pre.max(), f_post.max())

    im = axes[r, 0].imshow(phi_in, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[r, 0].set_title(f"requested $\\\\phi$\\n{name}", fontsize=10)
    fig.colorbar(im, ax=axes[r, 0], fraction=0.046)

    axes[r, 1].imshow(g_pre, cmap='gray_r', interpolation='nearest')
    axes[r, 1].set_title(f"pretrained: generated\\n$\\\\phi$ = {1 - g_pre.mean():.3f}", fontsize=10)
    im = axes[r, 2].imshow(f_pre, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[r, 2].set_title(f"pretrained: realised $\\\\phi$\\n$\\\\rho$ = {r_pre:.3f}", fontsize=10)
    fig.colorbar(im, ax=axes[r, 2], fraction=0.046)

    axes[r, 3].imshow(g_post, cmap='gray_r', interpolation='nearest')
    axes[r, 3].set_title(f"post-trained: generated\\n$\\\\phi$ = {1 - g_post.mean():.3f}", fontsize=10)
    im = axes[r, 4].imshow(f_post, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[r, 4].set_title(f"post-trained: realised $\\\\phi$\\n$\\\\rho$ = {r_post:.3f}", fontsize=10)
    fig.colorbar(im, ax=axes[r, 4], fraction=0.046)

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Same field, same noise, same architecture — only the weights differ "
             "(case 4 vs case 5)", y=0.98)
fig.tight_layout()
fc.savefig(fig, "03_pretrained_vs_posttrained")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

axes[0].plot(phi_cond_i.ravel(), phi_gen_pre_i.ravel(), '.', ms=2, alpha=0.28, color='C7',
             label=f'pretrained  $\\\\rho$ = {rho_pre:.3f}')
axes[0].plot(phi_cond_i.ravel(), phi_gen_i.ravel(), '.', ms=2, alpha=0.28, color='C0',
             label=f'post-trained  $\\\\rho$ = {rho:.3f}')
lo = min(phi_cond_i.min(), phi_gen_pre_i.min(), phi_gen_i.min())
hi = max(phi_cond_i.max(), phi_gen_pre_i.max(), phi_gen_i.max())
axes[0].plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='identity')
axes[0].set_xlabel("requested $\\\\phi$"); axes[0].set_ylabel("realised $\\\\phi$")
axes[0].set_title("measured field"); axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, markerscale=5)

mid = ramp_in_i.shape[0] // 2
axes[1].plot(ramp_in_i[mid], 'k-', lw=2.4, label='requested ramp')
axes[1].plot(ramp_out_pre_i[mid], color='C7', lw=1.7,
             label=f'pretrained ($\\\\rho$ = {rho_ramp_pre:.3f})')
axes[1].plot(ramp_out_i[mid], color='C0', lw=1.7,
             label=f'post-trained ($\\\\rho$ = {rho_ramp:.3f})')
axes[1].set_xlabel("position [latent units]"); axes[1].set_ylabel("$\\\\phi$")
axes[1].set_title("mid-line profile across the ramp")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

fig.tight_layout()
fc.savefig(fig, "03_pretrained_vs_posttrained_tracking")
plt.show()'''

_NB03_CMP_NOTE = r"""Read the two $\rho$ columns printed above; they settle the question rather than the
prose doing it. The mechanism to expect: the pretrained model was only ever rewarded for
matching a *global* porosity, so a field is, to it, roughly a strangely-encoded scalar —
it should track the overall level but respond weakly to where the field is high or low.
The ramp is the sharper test of the two, because a model that ignores spatial structure
has nowhere to hide: it will produce a uniform image where a gradient was asked for.

Two caveats worth keeping in mind when reading these numbers:

- $\rho$ is not bounded by 1 in practice — the field is a *soft* constraint, and even a
  perfectly fine-tuned model produces microstructure whose realised field only resembles
  the requested one. Notebook 04 quantifies this at generation scale.
- a single sample per model per field. Treat the gap as indicative; the sample-to-sample
  spread is measured properly in notebook 04."""


# ------------------------------------------------- ADDITION: nb 03, section 5c
# Why the fine-tuning loss curve is nearly flat while behaviour changes a lot.
# Measured facts that motivate it:
#   scalar pretrain : val_loss 0.1024 -> 0.0522   (49% drop, 15.4k steps)
#   field finetune  : val_loss 0.0571 -> 0.0513   (10% drop); train 0.0564 -> 0.0549
#   ||theta_ft - theta_pre|| / ||theta_pre|| = 0.075, and the conditioning embedder
#   is the *least* changed module (0.017) -- most drift is in the bottleneck blocks.

_NB03_SENS_MD = r"""### Why did the loss barely move?

The fine-tune changes behaviour qualitatively, yet its loss curve is nearly flat:
train loss went $0.0564 \to 0.0549$ (2.6%), and its *first* validation already sat at
the pretrained model's converged level. The weights moved only 7.5% in relative $L_2$ —
and, against intuition, **least of all in the conditioning embedder** (1.7%); most of the
drift is in the UNet bottleneck and the resampling layers.

That combination is only confusing if you assume the denoising loss measures how well the
model follows the field. It does not necessarily, and this section measures how much it
can see at all.

The objective is $\lambda(\sigma)\lVert D_\theta(z_\sigma,\sigma,y) - z\rVert^2$. Almost
all of that error is irreducible — at a given $\sigma$ the noise simply cannot be removed.
Conditioning can only help predict the *large-scale, low-frequency* part of $z$, a small
share of its variance. So a perfect field-follower and a field-ignorer may differ by a
couple of percent of loss while their samples differ obviously.

The probe holds everything fixed — same latents, same noise, same $\sigma$, same frozen
VAE — and varies only the conditioning:

| variant | what it removes |
|---|---|
| `field (correct)` | nothing |
| `field, mismatched` | the spatial correspondence (another patch's field, same marginals) |
| `flat = own mean` | the spatial *structure*, keeping the correct overall level |
| `none` | the conditioning entirely |

`flat = own mean` is the one to watch: its gap to `field (correct)` is exactly the part of
the loss that **spatial** conditioning information can explain."""

_NB03_SENS_CODE = r'''# A fixed probe set: identical latents, noise and sigmas for every variant and both
# models, so nothing but the conditioning (and the weights) can move the loss.
N_PROBE_BATCHES, PROBE_BS = 12, 8
probe_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=PROBE_BS, shuffle=False, num_workers=0)

torch.manual_seed(0)
probe = []
with torch.no_grad():
    for k, b in enumerate(probe_loader):
        if k >= N_PROBE_BATCHES:
            break
        z, _ = module.encode(b['x'].to(DEVICE), None)   # frozen VAE, shared by both models
        z = module.initial_norm(z)
        probe.append((z, b['y']['porosity'].to(DEVICE), torch.randn_like(z)))
print(f"probe: {len(probe)} batches x {PROBE_BS}, latent {tuple(probe[0][0].shape[1:])}")

# In this config t IS sigma, so we can probe the noise levels the sampler visits.
SIGMAS = module.create_time_schedule(13).cpu().numpy()[:-1]
print(f"sigma grid: {np.array2string(SIGMAS, precision=3)}")


def conditioning_variants(phi):
    """Four conditionings with the same marginals but different spatial information."""
    return [
        ('field (correct)',   {'porosity': phi}),
        ('field, mismatched', {'porosity': phi.roll(1, 0)}),
        ('flat = own mean',   {'porosity': phi.mean(dim=(1, 2), keepdim=True).expand_as(phi)}),
        ('none',              None),
    ]


VARIANTS = [n for n, _ in conditioning_variants(probe[0][1])]


@torch.no_grad()
def loss_vs_sigma(mod):
    """EDM-weighted denoising loss per sigma, per conditioning variant."""
    mod.eval()
    out = {n: np.zeros(len(SIGMAS)) for n in VARIANTS}
    for z, phi, noise in probe:
        for si, s in enumerate(SIGMAS):
            t = torch.full((z.shape[0],), float(s), device=DEVICE)
            alpha = mod.config.alpha_fn(t).view(-1, 1, 1, 1)
            sigma = mod.config.sigma_fn(t).view(-1, 1, 1, 1)
            z_sig = alpha * z + sigma * noise
            w = mod.config.loss_weighting.weighting_function(t.view(-1, 1, 1, 1))
            for name, y in conditioning_variants(phi):
                d = mod.get_denoiser_output(z_sig, t, y=y)
                out[name][si] += float(
                    (mod.config.loss_metric_module(d, z) * w).mean()) / len(probe)
    return out


t0 = time.time()
sens = {'pretrained (scalar)': loss_vs_sigma(pre_module),
        'post-trained (field)': loss_vs_sigma(module)}
print(f"probed in {time.time() - t0:.0f}s")

for mname, r in sens.items():
    base = r['field (correct)'].mean()
    print(f"\n{mname}:  mean weighted loss with the correct field = {base:.5f}")
    for v in VARIANTS:
        print(f"    {v:<20} {r[v].mean():.5f}   {100 * (r[v].mean() - base) / base:+7.2f}%")'''

_NB03_SENS_FIG = r'''fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.6))

for ax, (mname, r) in zip(axes[:2], sens.items()):
    for v, c in zip(VARIANTS, ['C0', 'C3', 'C1', 'C7']):
        ax.plot(SIGMAS, r[v], 'o-', color=c, lw=1.8, ms=4, label=v)
    ax.set_xscale('log')
    ax.set_xlabel("$\\sigma$"); ax.set_ylabel("weighted denoising loss")
    ax.set_title(mname); ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

ax = axes[2]
for (mname, r), c, mk in zip(sens.items(), ['C7', 'C0'], ['s--', 'o-']):
    gap = 100 * (r['flat = own mean'] - r['field (correct)']) / r['field (correct)']
    ax.plot(SIGMAS, gap, mk, color=c, lw=2, ms=4, label=mname)
ax.axhline(0, color='k', lw=0.8)
ax.set_xscale('log'); ax.set_xlabel("$\\sigma$")
ax.set_ylabel("loss penalty for removing\nspatial information [%]")
ax.set_title("How much the loss can even see the field")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

fig.tight_layout()
fc.savefig(fig, "03_conditioning_loss_sensitivity")
plt.show()'''

_NB03_SENS_NOTE = r"""The numbers above distinguish two readings, so let them decide:

- **If the `flat = own mean` penalty is small (a few percent) for the post-trained model**,
  the loss essentially cannot see the conditioning. A flat fine-tuning curve then tells you
  nothing about *when* the model learned to follow the field — what converged was the loss,
  not the behaviour. This would also explain why the fine-tune's first validation already
  sat at the pretrained level.
- **If the penalty is large**, the loss does track conditioning quality, and fast
  convergence really does mean the field was learned quickly — consistent with the small
  weight change, i.e. the capability was largely present and only needed routing.

Also look at *where* in $\sigma$ the penalty lives. Conditioning should matter most at
large $\sigma$, where the large-scale structure of the sample is still undetermined; at
small $\sigma$ the answer is already contained in $z_\sigma$ and the field is redundant. A
penalty peaked at large $\sigma$ but averaging to nearly nothing is the signature of a loss
dominated by noise levels at which conditioning is irrelevant.

Two caveats before drawing conclusions:

- the post-trained checkpoint holds **EMA** weights while the pretrained one is raw, so a
  small part of the weight-space distance is not attributable to the field task;
- the pretrained model was still drifting when it stopped (its last validation, 0.0595, is
  above its own minimum, 0.0522), so some of the fine-tune's improvement is just more
  training. A control fine-tune on **mismatched** fields for the same number of steps would
  separate the two.

**The decisive follow-up**, if you want it: checkpoint every ~50 steps for the first few
hundred steps of the fine-tune and plot $\rho$(requested, realised) against step on the
same axes as the loss. If $\rho$ saturates within a couple hundred steps while the loss
stays flat throughout, the picture is complete."""


def insert_sensitivity(nb):
    """Insert the conditioning-sensitivity probe after the comparison note."""
    for c in nb['cells']:
        if '03_conditioning_loss_sensitivity' in ''.join(c['source']):
            return 0
    anchor = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and '03_pretrained_vs_posttrained' in ''.join(c['source']):
            anchor = i
    if anchor is None:
        return 0
    # place after the interpretive markdown that follows the comparison figure
    while (anchor + 1 < len(nb['cells'])
           and nb['cells'][anchor + 1]['cell_type'] == 'markdown'
           and 'What to carry forward' not in ''.join(nb['cells'][anchor + 1]['source'])):
        anchor += 1
    new = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': _lines(_NB03_SENS_MD)},
        {'cell_type': 'code', 'metadata': {}, 'source': _lines(_NB03_SENS_CODE),
         'execution_count': None, 'outputs': []},
        {'cell_type': 'code', 'metadata': {}, 'source': _lines(_NB03_SENS_FIG),
         'execution_count': None, 'outputs': []},
        {'cell_type': 'markdown', 'metadata': {}, 'source': _lines(_NB03_SENS_NOTE)},
    ]
    nb['cells'][anchor + 1:anchor + 1] = new
    return len(new)


def insert_comparison(nb):
    """Insert the pretrained-vs-post-trained comparison after the ramp-control cell."""
    for c in nb['cells']:
        if '03_pretrained_vs_posttrained' in ''.join(c['source']):
            return 0
    anchor = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and '03_synthetic_ramp_control' in ''.join(c['source']):
            anchor = i
    if anchor is None:
        return 0
    new = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': _lines(_NB03_CMP_MD)},
        {'cell_type': 'code', 'metadata': {}, 'source': _lines(_NB03_CMP_GEN),
         'execution_count': None, 'outputs': []},
        {'cell_type': 'code', 'metadata': {}, 'source': _lines(_NB03_CMP_FIG),
         'execution_count': None, 'outputs': []},
        {'cell_type': 'markdown', 'metadata': {}, 'source': _lines(_NB03_CMP_NOTE)},
    ]
    nb['cells'][anchor + 1:anchor + 1] = new
    return len(new)


def insert_selftest(nb):
    """Add the field-definition self-test right after the generation cell."""
    for c in nb['cells']:
        if 'realised field of the REAL image' in ''.join(c['source']):
            return 0
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and 'def latent_field_of' in ''.join(c['source']):
            nb['cells'].insert(i + 1, {
                'cell_type': 'code', 'metadata': {},
                'source': _lines(_NB03_SELFTEST),
                'execution_count': None, 'outputs': [],
            })
            return 1
    return 0


def _lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def fix_section9(nb):
    """Replace the four cells of section 9. Returns number of cells changed."""
    idx_md = idx_code = idx_fig = idx_sum = None
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])
        if c['cell_type'] == 'markdown' and 'coarse-to-fine shortcut' in src:
            idx_md = i
        elif c['cell_type'] == 'code' and 'exact_fields = np.stack' in src:
            idx_code = i
        elif c['cell_type'] == 'code' and '01_coarse_to_fine_cost' in src:
            idx_fig = i
        elif c['cell_type'] == 'markdown' and 'The honest summary' in src:
            idx_sum = i
    if None in (idx_md, idx_code, idx_fig, idx_sum):
        return 0

    for i, text in [(idx_md, SECTION9_MD), (idx_sum, SECTION9_SUMMARY)]:
        nb['cells'][i]['source'] = _lines(text)
    for i, text in [(idx_code, SECTION9_CODE), (idx_fig, SECTION9_FIG)]:
        nb['cells'][i]['source'] = _lines(text)
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='apply (default: dry run)')
    args = ap.parse_args()

    changed_any = False

    # --- source-level patches ---
    for path, anchor, desc, fn, unless in PATCHES:
        nb = json.load(open(path))
        hits = 0
        for c in nb['cells']:
            if c['cell_type'] != 'code':
                continue
            src = ''.join(c['source'])
            if unless is not None and unless in src:
                continue
            if anchor in src:
                c['source'] = fn(c['source'])
                c['outputs'] = []
                c['execution_count'] = None
                hits += 1
        if hits == 0:
            print(f"[skip ] {path}: {desc} (anchor absent -- already applied?)")
            continue
        print(f"[{'apply' if args.write else 'would'}] {path}: {desc} ({hits} cell)")
        changed_any = True
        if args.write:
            json.dump(nb, open(path, 'w'), indent=1)

    # --- whole-cell replacements ---
    by_nb = {}
    for path, anchor, new_src, desc, unless in REPLACEMENTS:
        by_nb.setdefault(path, []).append((anchor, new_src, desc, unless))
    for path, items in by_nb.items():
        nb = json.load(open(path))
        touched = False
        for anchor, new_src, desc, unless in items:
            hits = 0
            for c in nb['cells']:
                if c['cell_type'] != 'code':
                    continue
                src = ''.join(c['source'])
                if unless is not None and unless in src:
                    continue
                if anchor in src:
                    c['source'] = _lines(new_src)
                    c['outputs'] = []
                    c['execution_count'] = None
                    hits += 1
                    break                      # first match only
            if hits == 0:
                print(f"[skip ] {path}: {desc} (anchor absent -- already applied?)")
            else:
                print(f"[{'apply' if args.write else 'would'}] {path}: {desc}")
                touched = True
        for fn, label in [(insert_selftest, 'field-definition self-test cell'),
                          (insert_comparison, 'pretrained-vs-post-trained comparison (4 cells)'),
                          (insert_sensitivity, 'conditioning loss-sensitivity probe (4 cells)')]:
            if fn(nb):
                print(f"[{'apply' if args.write else 'would'}] {path}: insert {label}")
                touched = True
            else:
                print(f"[skip ] {path}: {label} (already present)")
        if touched:
            changed_any = True
            if args.write:
                json.dump(nb, open(path, 'w'), indent=1)

    # --- section 9 rewrite ---
    path = '01-porosity-field-and-gaussian-process.ipynb'
    nb = json.load(open(path))
    n = fix_section9(nb)
    if n == 0:
        print(f"[skip ] {path}: section 9 sweep (anchors absent -- already applied?)")
    else:
        print(f"[{'apply' if args.write else 'would'}] {path}: section 9 -> refinement-ratio sweep ({n} cells)")
        changed_any = True
        if args.write:
            json.dump(nb, open(path, 'w'), indent=1)

    if not args.write and changed_any:
        print("\nDry run. Re-run with --write to apply, then re-execute the cleared cells.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
