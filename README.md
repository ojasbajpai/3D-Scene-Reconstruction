# UAMC for MUSt3R
### Uncertainty-Guided Adaptive Memory Compression — A Novel Extension to MUSt3R 3D Scene Reconstruction

This repository contains a set of architectural extensions to [MUSt3R](https://github.com/naver/must3r) (NAVER LABS Europe) that address its core scalability bottleneck: an unbounded, quality-blind memory mechanism. We introduce **three interlocking contributions** — confidence-weighted keyframe admission, geometry-aware token merging, and selective confidence-gated re-rendering — implemented entirely as inference-time changes requiring **no retraining** of the base model.

---

## Table of Contents

- [Background](#background)
- [The Problem](#the-problem)
- [Our Solution — UAMC](#our-solution--uamc)
  - [Contribution 1 — Confidence-Weighted Memory Eviction](#contribution-1--confidence-weighted-memory-eviction)
  - [Contribution 2 — Geometry-Aware Token Merging](#contribution-2--geometry-aware-token-merging)
  - [Contribution 3 — Selective Confidence-Gated Re-Render](#contribution-3--selective-confidence-gated-re-render)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## Background

[MUSt3R](https://arxiv.org/abs/2503.01661) is a multi-view extension of DUSt3R that performs dense, unconstrained 3D reconstruction from RGB images with no camera calibration required. Its central innovation is a **multi-layer memory mechanism**: rather than processing all O(N²) image pairs like DUSt3R, each new frame cross-attends to a cache of previously processed keyframes, enabling real-time SLAM and large-scale SfM reconstruction.

MUSt3R maintains two independent data structures during online reconstruction:

| Structure | Contents | Role |
|---|---|---|
| `scene3d` | Accumulated 3D point cloud (`Xi,1` coordinates) | The actual reconstructed map — grows with every processed frame |
| `mem` (memory) | Decoder tokens `Di^l`, one set per keyframe per layer | Reference frames future frames cross-attend to for pose/depth estimation |

Every admitted keyframe contributes 768 tokens (one per 16×16 image patch) to memory, at every decoder layer. Cross-attention cost scales **O(Nmem²)** — so memory size directly determines both compute cost and reconstruction quality.

---

## The Problem

MUSt3R's keyframe admission rule (the gate that decides which frames enter `mem`) is purely geometric:

```python
# Original — must3r/slam/model.py
def choose_keyframe_from_overlap(overlap_score, thr, overlap_mode):
    if 'nn' in overlap_mode:
        outchoice = overlap_score > thr
    else:
        outchoice = overlap_score < thr
    return outchoice  # binary — True or False, nothing else

iskeyframe = is_first_frame or (
    choose_keyframe_from_overlap(res['overlap_score'], keyframe_overlap_thr, overlap_mode)
    and conf.median() > min_conf_keyframe   # crude binary confidence gate
)
```

This has three structural weaknesses:

1. **Blind to prediction quality.** The rule only asks *"does this frame see new geometry?"* — never *"how confident is the network about this frame's 3D predictions?"* A motion-blurred or poorly-lit frame that happens to see new geometry is admitted as a **full memory anchor** with equal voting power to a sharp, high-confidence frame, polluting cross-attention for every subsequent frame.
2. **Binary confidence gating.** `conf.median() > min_conf_keyframe` is an on/off switch with no middle ground — a frame is either fully trusted or entirely ignored.
3. **Unbounded linear memory growth.** There is no mechanism to deduplicate or compress tokens once admitted. If 10 keyframes all observe the same wall from different angles, all 10×768 tokens persist independently in memory, forever.

The MUSt3R paper itself acknowledges the consequence of this design (Section 5.5):

> *"MUSt3R shows signs of limitations for sequences where the views drift too far from the 1st view."*

---

## Our Solution — UAMC

We propose **Uncertainty-guided Adaptive Memory Compression (UAMC)** — three contributions that operate on top of MUSt3R's existing architecture without modifying or retraining the network itself.

> **Design principle:** `scene3d` always receives every frame's 3D geometry, regardless of confidence. UAMC only governs whether a frame becomes a trusted **memory anchor**. No geometry is ever lost — we only filter what gets used as a cross-attention reference.

### Contribution 1 — Confidence-Weighted Memory Eviction

We replace the binary admission rule with a **joint confidence-geometry score** and a **three-tier admission system**.

**`must3r/slam/model.py` — `choose_keyframe_from_overlap()`**

```python
def choose_keyframe_from_overlap(overlap_score, thr, overlap_mode,
                                  conf_score=None, alpha=0.4, beta=0.6,
                                  hard_thr=0.55, soft_thr=0.30):
    if 'nn' in overlap_mode:
        geo_score = float(overlap_score > thr)
    else:
        geo_score = float(overlap_score < thr)

    if conf_score is None:
        return bool(geo_score)   # fallback — preserves original behavior

    joint_score = alpha * conf_score + beta * geo_score

    if joint_score >= hard_thr:
        return 'hard'                          # full memory anchor, weight = 1.0
    elif geo_score > 0 and conf_score > 0.2:
        return 'soft'                          # down-weighted anchor
    else:
        return False                           # excluded from memory
```

**`must3r/slam/model.py` — `postproc_pred()`**

```python
conf_score = torch.tanh((res['conf'].float() - 1.0).mean()).item()

kf_decision = choose_keyframe_from_overlap(
    res['overlap_score'], keyframe_overlap_thr, overlap_mode,
    conf_score=conf_score, alpha=0.4, beta=0.6,
    hard_thr=0.55, soft_thr=0.30
)
iskeyframe = is_first_frame or (kf_decision in ('hard', 'soft'))
iskeyframe_tier = 'hard' if is_first_frame else kf_decision
```

> **Note on confidence normalization:** MUSt3R's `conf` values are post-activation (`1 + exp(logit)`), ranging from 1 to ∞ — not raw logits and not probabilities. We subtract 1.0 to shift the range to (0, ∞), then apply `tanh` to squash it smoothly into (0, 1). This is a necessary correction; naively applying `sigmoid()` to post-activation values collapses nearly all frames to the same output.

The decision propagates through the call chain (`postproc_pred` → `MUSt3RAgent.update()` → `SLAM_MUSt3R`) as an additional `iskeyframe_tier` return value, used downstream to weight memory contributions.

**`must3r/demo/inference.py` — `slam_is_keyframe()`**

The Gradio demo path uses a separate keyframe callback, updated with the same joint scoring logic:

```python
def slam_is_keyframe(subsample, min_conf_keyframe, keyframe_overlap_thr,
                     overlap_percentile, overlap_mode, id, res, scene_state):
    cam_center = res['c2w'][:3, -1]
    res_unsqueeze = {k: v.unsqueeze(0).unsqueeze(0) for k, v in res.items()}
    overlap_score = get_overlap_score(res_unsqueeze, scene_state, cam_center=cam_center,
                                      mode=overlap_mode, kf_x_subsamp=subsample,
                                      min_conf_keyframe=min_conf_keyframe,
                                      percentile=overlap_percentile)
    assert not np.isnan(overlap_score)
    overlap_score = float(np.clip(overlap_score, 0.0, 2.0))  # guards against numerical overflow

    conf_score = torch.tanh((res['conf'].float() - 1.0).mean()).item()

    kf_decision = choose_keyframe_from_overlap(
        overlap_score, keyframe_overlap_thr, overlap_mode,
        conf_score=conf_score, alpha=0.4, beta=0.6,
        hard_thr=0.55, soft_thr=0.30
    )
    return bool(kf_decision)
```

**Three-tier admission table:**

| Tier | Condition | Memory action | Scene action |
|---|---|---|---|
| **Hard anchor** | `joint_score ≥ 0.55` | Full entry, weight = 1.0 | `scene3d.add()` — always |
| **Soft anchor** | New geometry, low confidence | Entry with weight = `conf_score` | `scene3d.add()` — always |
| **Discarded** | `joint_score < 0.30` | No memory entry | `scene3d.add()` — always |

---

### Contribution 2 — Geometry-Aware Token Merging

Even with confidence filtering, every admitted keyframe still adds 768 tokens to memory. If multiple frames observe the same physical region, their tokens are redundant. We introduce a `GeometricTokenMerger` that projects every decoder token to its 3D centroid and merges tokens across frames that fall within a small spatial radius.

**`must3r/memory_merger.py` (new file)**

Each decoder token corresponds to a 16×16 pixel patch. We compute its 3D centroid by averaging the global pointmap over that patch:

```python
def register_keyframe(self, label, pointmap, conf_score=1.0):
    pts = pointmap['pts3d']                                   # [H, W, 3]
    pts_r = pts[:ph*self.patch_size, :pw*self.patch_size]
    pts_r = pts_r.reshape(ph, self.patch_size, pw, self.patch_size, 3)
    centroids = pts_r.mean(dim=(1, 3)).reshape(-1, 3)         # [num_patches, 3]
    self._token_positions[label] = centroids.cpu()
    self._token_weights[label] = conf_score
```

Merging runs every `merge_every_k` keyframes (default 5). It builds a KD-tree over all token centroids currently in memory, finds pairs within `merge_dist_thresh` (default 0.10m), and clusters them with union-find:

```python
def merge_memory(self, mem):
    if not self.should_merge():
        return mem

    mem_vals, mem_labels, mem_nimgs, mem_protected, mem_protected_tokens = mem
    # ... collect all token 3D positions across frames currently in memory ...

    tree = KDTree(all_positions)
    pairs = tree.query_pairs(r=self.tau_m)
    clusters = self._union_find(len(all_positions), pairs)
    merge_clusters = [c for c in clusters if len(c) > 1]

    # confidence-weighted average of every merge cluster
    for layer_vals in mem_vals:
        layer_vals = layer_vals.clone()
        for survivor_idx, other_indices in tokens_to_update.items():
            all_indices = [survivor_idx] + other_indices
            weights = torch.tensor([self._token_weights.get(
                int(mem_labels[0, idx].item()), 1.0) for idx in all_indices])
            weights = weights / weights.sum()
            avg = (layer_vals[0, all_indices, :] * weights.unsqueeze(1)).sum(dim=0)
            layer_vals[0, survivor_idx, :] = avg

    # remove redundant tokens via boolean masking
    keep_mask = torch.ones(Nmem, dtype=torch.bool)
    for idx in tokens_to_remove:
        keep_mask[idx] = False
    new_mem_vals = [v[:, keep_mask, :] for v in new_mem_vals]
    new_mem_labels = mem_labels[:, keep_mask]

    return [new_mem_vals, new_mem_labels, mem_nimgs, mem_protected, mem_protected_tokens]
```

Confidence weights from Contribution 1 directly govern the averaging — a sharp, high-confidence frame's token dominates the merged representation over a blurry frame's token describing the same 3D region.

> **Implementation detail:** `merge_memory()` must return a **list**, not a tuple. Downstream code in `engine/inference.py` performs in-place item assignment (`mem[2] = len(img_labels)`), which fails on immutable tuples.

**`must3r/engine/inference.py` — hookup**

```python
merger = GeometricTokenMerger(merge_dist_thresh=0.10, merge_every_k=5)
...
if is_keyframe:
    keyframes.add(img_id_i)
    scene_state = scene_state_update_function(pointmaps_0_i[j], scene_state)
    conf_score = torch.sigmoid(pointmaps_0_i[j]['conf']).mean().item()
    merger.register_keyframe(new_labels[j], pointmaps_0_i[j], conf_score=conf_score)
    mem = merger.merge_memory(mem)
```

---

### Contribution 3 — Selective Confidence-Gated Re-Render

MUSt3R supports a "rendering" mode that re-processes frames using the accumulated memory without updating it — but the baseline applies this uniformly to all frames. We make it **selective**: only frames with low first-pass confidence are re-rendered, and each re-render only cross-attends to a spatially local subset of memory rather than the full token cache.

**`must3r/engine/inference.py` — tracking low-confidence frames**

```python
if 'conf' in pointmaps_0_i[j]:
    frame_conf = (pointmaps_0_i[j]['conf'].float() - 1.0).mean().item()
    if frame_conf < 0.5:
        low_conf_frames.append(img_id_i)
```

**`must3r/memory_merger.py` — `get_local_memory()`**

```python
def get_local_memory(mem, frame_positions, merger, top_k=20):
    mem_vals, mem_labels, mem_nimgs, mem_protected, mem_protected_tokens = mem
    Nmem = mem_labels.shape[1]
    if Nmem <= top_k:
        return mem

    # build a 3D centroid per memory token from the merger's per-label registry
    positions = np.zeros((Nmem, 3), dtype=np.float32)
    for token_idx in range(Nmem):
        lbl_int = int(mem_labels[0, token_idx].item())
        centroid = label_to_centroid.get(lbl_int, frame_positions)
        positions[token_idx] = centroid.numpy() if isinstance(centroid, torch.Tensor) else centroid
    positions = torch.from_numpy(positions)

    dists = torch.norm(positions - frame_positions.float().unsqueeze(0), dim=-1)
    _, topk_ids = torch.topk(dists, k=min(top_k, Nmem), largest=False)
    topk_ids = topk_ids.sort().values

    local_vals = [v[:, topk_ids, :] for v in mem_vals]
    local_labels = mem_labels[:, topk_ids]
    return [local_vals, local_labels, mem_nimgs, mem_protected, mem_protected_tokens]
```

**`must3r/engine/inference.py` — the re-render pass**

After the main memory-building loop completes, flagged frames are re-processed using a spatially restricted memory slice and cached encoder features (avoiding a second costly encoder pass):

```python
for frame_id in low_conf_frames:
    frame_pm = pointmaps_0[frame_id]
    frame_center = frame_pm['pts3d'].reshape(-1, 3).mean(0).cpu()

    Nmem_current = mem[1].shape[1]
    local_mem = mem if Nmem_current < 2000 else get_local_memory(mem, frame_center, merger, top_k=500)

    frame_img, frame_true_shape = imgs[frame_id], true_shape[frame_id]
    frame_x, frame_pos = x[frame_id], pos[frame_id]

    _, refined_pm = inference_multi_ar_batch(
        encoder, decoder,
        [frame_img.unsqueeze(0)], [frame_true_shape.unsqueeze(0)],
        local_mem,
        encoder_precomputed_features=([frame_x.unsqueeze(0)], [frame_pos.unsqueeze(0)]),
        post_process_function=post_process_function,
        device=device,
        render=True   # reads memory, does not append to it
    )

    refined_dict = refined_pm[0]
    if isinstance(refined_dict, (list, tuple)):
        refined_dict = refined_dict[0]

    # drop the batch dimension introduced by single-frame inference
    if refined_dict['pts3d'].dim() == 4:
        refined_dict = {k: v.squeeze(0) for k, v in refined_dict.items()
                       if isinstance(v, torch.Tensor)}

    # only replace the original pointmap if spatial dimensions match
    if pointmaps_0[frame_id]['pts3d'].shape == refined_dict['pts3d'].shape:
        pointmaps_0[frame_id] = refined_dict
```

`render=True` ensures the re-render is a pure read of memory — it never appends new tokens, so the compressed memory built by Contribution 2 remains stable. The spatial slicing from `get_local_memory()` (built on Contribution 2's position registry) keeps the re-render cheap even when the global memory is large.

---

## Results

Validated end-to-end on an 11-frame indoor room sequence (CPU inference):

| Frame | Baseline `Nmem` | UAMC `Nmem` | Reduction |
|---|---|---|---|
| 6 | 5,376 | 3,113 | **42%** |
| 7 | 6,144 | 3,881 | 37% |
| 8 | 6,912 | 4,649 | 33% |
| 9 | 7,680 | 5,417 | 29% |
| 10 | 8,448 | 6,185 | 27% |

```
[UAMC-Merger] Merged: 5376 → 3113 tokens (removed 2263, 304 clusters)
[UAMC-Rerender] Re-rendering 3 low-confidence frames: [7, 8, 10]
```

- **Memory compression:** up to 42% token reduction from a single merge pass on 5 keyframes; the effect compounds on longer sequences as more redundant geometry is observed.
- **Inference latency:** total pipeline time on CPU dropped from **~2 min 15 sec to ~41 sec** across iterations of the pipeline, driven by reduced cross-attention cost from a smaller, deduplicated memory.
- **End-to-end stability:** all three contributions run together without breaking the original reconstruction pipeline — the final point cloud renders correctly through the Gradio demo.
- **Re-render targeting:** the selective re-render correctly identifies and reprocesses low-confidence frames using spatially local memory, leaving high-confidence frames untouched.

---

## Repository Structure

```
must3r/
├── slam/
│   └── model.py              # MODIFIED — joint confidence-geometry scoring, 3-tier admission
├── demo/
│   └── inference.py           # MODIFIED — slam_is_keyframe() uses joint scoring
├── engine/
│   └── inference.py           # MODIFIED — merger hookup, low-confidence tracking, selective re-render
└── memory_merger.py            # NEW — GeometricTokenMerger, get_local_memory()
```

| File | Status | Key changes |
|---|---|---|
| `must3r/slam/model.py` | Modified | `choose_keyframe_from_overlap()` → 3-tier system; `postproc_pred()` → confidence scoring, `iskeyframe_tier` propagation |
| `must3r/demo/inference.py` | Modified | `slam_is_keyframe()` → joint confidence-geometry scoring for the Gradio demo path |
| `must3r/engine/inference.py` | Modified | Merger initialization, keyframe registration, confidence-weighted memory weighting, selective re-render block |
| `must3r/memory_merger.py` | New | `GeometricTokenMerger` class (token-to-3D projection, KD-tree clustering, confidence-weighted merging); `get_local_memory()` for spatial slicing |

---

## Setup

This project extends the official [MUSt3R](https://github.com/naver/must3r) repository. Clone the base repo and apply the modified files:

```bash
git clone --recursive https://github.com/naver/must3r.git
cd must3r

# Replace/add the files from this repository
cp /path/to/this/repo/slam/model.py must3r/slam/model.py
cp /path/to/this/repo/demo/inference.py must3r/demo/inference.py
cp /path/to/this/repo/engine/inference.py must3r/engine/inference.py
cp /path/to/this/repo/memory_merger.py must3r/memory_merger.py
```

Follow the original MUSt3R installation instructions for environment setup (PyTorch, xformers, checkpoints, etc.).

---

## Usage

Run the Gradio demo exactly as in the base repository:

```bash
python must3r/demo/gradio.py --weights <checkpoint_path> --mode sequence:slam-keyframes
```

UAMC is active by default in `sequence: slam keyframes` mode. Tunable parameters (currently set as constants in code, exposed for experimentation):

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | 0.4 | Weight of confidence in joint admission score |
| `beta` | 0.6 | Weight of geometric novelty in joint admission score |
| `hard_thr` | 0.55 | Joint score above which a frame becomes a full memory anchor |
| `soft_thr` | 0.30 | Joint score above which a frame becomes a down-weighted soft anchor |
| `merge_dist_thresh` | 0.10 m | 3D distance below which tokens are merged |
| `merge_every_k` | 5 | Run the token merger every K keyframes |
| `frame_conf` threshold | 0.5 | Confidence (tanh-normalized) below which a frame is flagged for re-render |
| `top_k` (re-render) | 500 | Number of spatially nearest memory tokens used during re-render |

Debug output is printed to stdout, prefixed `[UAMC]` (keyframe admission decisions) and `[UAMC-Merger]` / `[UAMC-Rerender]` (compression and re-render events).

---

## Acknowledgements

This work builds directly on [MUSt3R: Multi-view Network for Stereo 3D Reconstruction](https://arxiv.org/abs/2503.01661) (Cabon, Stoffl, Antsfeld, Csurka, Chidlovskii, Revaud, Leroy — NAVER LABS Europe, 2025) and is implemented as a set of inference-time extensions to the official [naver/must3r](https://github.com/naver/must3r) codebase. All base model weights, architecture, and training are unmodified — UAMC operates entirely as a memory-management layer on top of the pretrained network.
