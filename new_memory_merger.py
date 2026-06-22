# must3r/memory_merger.py
import torch
import numpy as np

class GeometricTokenMerger:
    """
    Merges memory tokens that describe the same 3D region.
    Operates on mem_vals and mem_labels directly.
    Runs every K keyframes to keep Nmem bounded.
    """
    def __init__(self, merge_dist_thresh=0.05, patch_size=16, merge_every_k=5):
        self.tau_m = merge_dist_thresh   # meters — tokens closer than this get merged
        self.patch_size = patch_size
        self.merge_every_k = merge_every_k
        self._keyframe_count = 0
        self._token_positions = {}       # label -> 3D centroid tensor [num_patches, 3]
        self._token_weights = {}

    def register_keyframe(self, label, pointmap, conf_score=1.0):
        """
        Call this when a new keyframe is accepted.
        pointmap: dict with 'pts3d' key, shape [H, W, 3] — the global pointmap Xi,1
        label: int, the mem_label assigned to this frame
        """
        pts = pointmap['pts3d']          # [H, W, 3]
        H, W = pts.shape[:2]
        ph = H // self.patch_size
        pw = W // self.patch_size
        # reshape to get one 3D centroid per patch
        pts_r = pts[:ph*self.patch_size, :pw*self.patch_size]   # crop to fit
        pts_r = pts_r.reshape(ph, self.patch_size, pw, self.patch_size, 3)
        centroids = pts_r.mean(dim=(1, 3))       # [ph, pw, 3]
        centroids = centroids.reshape(-1, 3)      # [ph*pw, 3]
        self._token_positions[label] = centroids.cpu()
        self._token_weights[label] = conf_score
        self._keyframe_count += 1

    def should_merge(self):
        return self._keyframe_count % self.merge_every_k == 0

    def merge_memory(self, mem):
        """
        Main entry point. Takes mem tuple, returns compressed mem tuple.
        Only runs every merge_every_k keyframes.
        """
        if not self.should_merge():
            return mem

        mem_vals, mem_labels, mem_nimgs, mem_protected, mem_protected_tokens = mem

        # get all unique labels currently in memory
        unique_labels = torch.unique(mem_labels).tolist()
        unique_labels = [int(l) for l in unique_labels if int(l) in self._token_positions]

        if len(unique_labels) < 2:
            return mem

        # collect all token positions with their label and token index in mem
        all_positions = []
        all_labels_list = []
        all_token_indices = []   # index into the Nmem dimension

        for lbl in unique_labels:
            pos = self._token_positions[lbl]        # [num_tokens, 3]
            # find where this label's tokens sit in mem_labels
            label_mask = (mem_labels[0] == lbl)     # [Nmem] boolean
            token_indices = label_mask.nonzero(as_tuple=True)[0].tolist()

            # pos and token_indices must have same length
            n = min(len(token_indices), pos.shape[0])
            for k in range(n):
                all_positions.append(pos[k].numpy())
                all_labels_list.append(lbl)
                all_token_indices.append(token_indices[k])

        if len(all_positions) < 2:
            return mem

        all_positions = np.array(all_positions)     # [Ntotal, 3]

        # find mergeable pairs
        from scipy.spatial import KDTree
        tree = KDTree(all_positions)
        pairs = tree.query_pairs(r=self.tau_m)

        if len(pairs) == 0:
            return mem

        # union-find clustering
        clusters = self._union_find(len(all_positions), pairs)

        # only keep clusters with more than 1 token (actual merges)
        merge_clusters = [c for c in clusters if len(c) > 1]
        tokens_before = mem_labels.shape[1]

        if len(merge_clusters) == 0:
            return mem

        # build set of token indices to REMOVE (all but the first in each cluster)
        tokens_to_remove = set()
        tokens_to_update = {}   # surviving_token_idx -> list of indices to average with

        for cluster in merge_clusters:
            # token indices in mem for this cluster
            mem_indices = [all_token_indices[i] for i in cluster]
            survivor = mem_indices[0]    # keep the first token
            others = mem_indices[1:]     # remove these

            tokens_to_update[survivor] = others
            tokens_to_remove.update(others)

        # average the token vectors for survivors
        # mem_vals is a list of tensors, one per decoder layer, shape [1, Nmem, D]
        new_mem_vals = []
        for layer_vals in mem_vals:
            # layer_vals: [1, Nmem, D]
            layer_vals = layer_vals.clone()
            for survivor_idx, other_indices in tokens_to_update.items():
                all_indices = [survivor_idx] + other_indices
                # weighted average (equal weights for now)
                weights = torch.tensor([self._token_weights.get(
                    int(mem_labels[0, idx].item()), 1.0) 
                    for idx in all_indices])
                weights = weights / weights.sum()
                avg = (layer_vals[0, all_indices, :] * weights.unsqueeze(1)).sum(dim=0)
                layer_vals[0, survivor_idx, :] = avg
            new_mem_vals.append(layer_vals)

        # build keep mask — keep all tokens NOT in tokens_to_remove
        Nmem = mem_labels.shape[1]
        keep_mask = torch.ones(Nmem, dtype=torch.bool)
        for idx in tokens_to_remove:
            keep_mask[idx] = False

        # apply mask to vals and labels
        new_mem_vals = [
            v[:, keep_mask, :]   # [1, Nmem_new, D]
            for v in new_mem_vals
        ]
        new_mem_labels = mem_labels[:, keep_mask]   # [1, Nmem_new]

        tokens_after = new_mem_labels.shape[1]
        tokens_removed = tokens_before - tokens_after
        print(f"[UAMC-Merger] Merged: {tokens_before} → {tokens_after} tokens "
            f"(removed {tokens_removed}, {len(merge_clusters)} clusters)")

        # also clean up our position registry for removed labels
        # (labels that no longer appear in mem_labels)
        surviving_labels = set(torch.unique(new_mem_labels).tolist())
        self._token_positions = {
            k: v for k, v in self._token_positions.items()
            if k in surviving_labels
        }

        return [new_mem_vals, new_mem_labels, mem_nimgs,
                mem_protected, mem_protected_tokens]

    def _union_find(self, n, pairs):
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in pairs:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb
        from collections import defaultdict
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        return list(clusters.values())
    
def get_local_memory(mem, frame_positions, merger, top_k=20):
    """
    Slice memory to top_k tokens nearest to frame center.
    Uses per-label centroids from merger registry.
    """
    mem_vals, mem_labels, mem_nimgs, mem_protected, mem_protected_tokens = mem
    Nmem = mem_labels.shape[1]

    if Nmem <= top_k:
        return mem

    # build one centroid per unique label, then assign to each token
    unique_labels = torch.unique(mem_labels[0])
    label_to_centroid = {}
    for lbl in unique_labels:
        lbl_int = int(lbl.item())
        if lbl_int in merger._token_positions:
            label_to_centroid[lbl_int] = merger._token_positions[lbl_int].mean(dim=0)
        else:
            label_to_centroid[lbl_int] = frame_positions  # fallback: treat as nearby

    # assign centroid to every token based on its label
    positions = np.zeros((Nmem, 3), dtype=np.float32)
    for token_idx in range(Nmem):
        lbl_int = int(mem_labels[0, token_idx].item())
        centroid = label_to_centroid.get(lbl_int, frame_positions)
        if isinstance(centroid, torch.Tensor):
            positions[token_idx] = centroid.numpy()
        else:
            positions[token_idx] = centroid

    positions = torch.from_numpy(positions)  # [Nmem, 3] — fast conversion

    # distances from frame center to each token
    fp = frame_positions.float()
    if fp.shape != positions[0].shape:
        return mem  # shape mismatch safety

    dists = torch.norm(positions - fp.unsqueeze(0), dim=-1)  # [Nmem]

    k = min(top_k, Nmem)
    _, topk_ids = torch.topk(dists, k=k, largest=False)
    topk_ids = topk_ids.sort().values

    local_vals = [v[:, topk_ids, :] for v in mem_vals]
    local_labels = mem_labels[:, topk_ids]

    return [local_vals, local_labels, mem_nimgs,
            mem_protected, mem_protected_tokens]
