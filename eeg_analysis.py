"""Reproducible sensor-network TDA for BNCI 001-2014, training sessions A01-A09.

Refactors the supplied A01 notebook. Spectra are cached per trial, never raw-averaged.
All decisions and random selections are exported. No trial-level classification.
"""
# %% Imports and fixed protocol
from pathlib import Path
import hashlib
import json
import urllib.request
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch, csd, get_window
from scipy.stats import rankdata, friedmanchisquare
from scipy.sparse.csgraph import minimum_spanning_tree
from ripser import ripser

CLASS_NAMES = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}
MOVEMENTS = list(CLASS_NAMES.values())
BANDS = ["mu", "beta"]
NAMES = [f"{m}_{b}" for m in MOVEMENTS for b in BANDS]
CHANNELS = [f"EEG{i:02d}" for i in range(1, 23)]
COLORS = ["#007C91", "#C28E00", "#7C5BA7", "#D34B40"]
BASE_URL = "https://lampx.tugraz.at/~bci/database/001-2014/"
GRID = np.linspace(0, 1, 2001)
MAIN_SEED = 42
SELECTION_SEEDS = np.arange(100, 120)


def download_files(data_dir, subjects=range(1, 10)):
    """Download missing official training files, leaving existing files untouched."""
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for subject in subjects:
        path = data_dir / f"A{subject:02d}T.mat"
        if not path.exists():
            print(f"Downloading {path.name}", flush=True)
            temporary = path.with_suffix(".mat.part")
            with urllib.request.urlopen(BASE_URL + path.name, timeout=120) as response:
                with temporary.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            temporary.replace(path)
        paths.append(path)
    return paths


def prepare_subject(path):
    """Keep the original artifact/nonfinite policy; stop on incomplete/flat epochs.

    Original block and trial IDs remain zero-based. Amplitude flags are review
    candidates, NOT additional exclusions in the main analysis.
    """
    path = Path(path)
    runs = np.atleast_1d(loadmat(path, squeeze_me=True,
                               struct_as_record=False)["data"]).ravel()
    epochs, eyes, labels, ids, rows = [], [], [], [], []
    task_count = 0
    for block_id, run in enumerate(runs):
        markers = np.atleast_1d(run.trial).astype(int).ravel()
        if markers.size == 0:
            continue  # A04 has a different number of calibration blocks.
        task_count += 1
        fs = float(run.fs)
        if fs != 250 or run.X.shape[1] != 25:
            raise ValueError(f"Unexpected sampling rate/channel count: {path.name}")
        classes = [str(x).replace(" ", "_") for x in np.atleast_1d(run.classes)]
        if classes != MOVEMENTS:
            raise ValueError(f"Unexpected class order: {classes}")
        y = np.atleast_1d(run.y).astype(int).ravel()
        flags = np.atleast_1d(run.artifacts).astype(int).ravel()
        if not (len(markers) == len(y) == len(flags)):
            raise ValueError("Mismatched trial markers, labels or artifact flags")
        for trial_id, (marker, label, flag) in enumerate(zip(markers, y, flags)):
            start = int(marker) - 1 + 500
            stop = start + 1000
            if start < 0 or stop > len(run.X):
                raise ValueError(f"Incomplete epoch: {path.name}, {block_id}, {trial_id}")
            x = np.asarray(run.X[start:stop, :22], dtype=float)
            finite = bool(np.isfinite(x).all())
            keep = flag == 0 and finite
            reasons = []
            if flag != 0:
                reasons.append("dataset_artifact")
            if not finite:
                reasons.append("nonfinite_eeg")
            rows.append(dict(subject=path.stem, block=block_id, trial=trial_id,
                             label=int(label), movement=CLASS_NAMES[int(label)],
                             artifact_flag=int(flag), finite=finite, retained=keep,
                             exclusion_reason=";".join(reasons), review=False,
                             max_ptp_uv=float(np.ptp(x, axis=0).max()) if finite else np.nan))
            if keep:
                epochs.append(x)
                eyes.append(np.asarray(run.X[start:stop, 22:25], dtype=float))
                labels.append(label)
                ids.append((block_id, trial_id))
    X, y, ids = np.stack(epochs), np.asarray(labels), np.asarray(ids)
    if (np.ptp(X, axis=1) <= 1e-12).any():
        raise ValueError(f"Flat retained EEG channel in {path.name}; inspect before continuing")
    amp = np.ptp(X, axis=1).max(axis=1)
    med = np.median(amp)
    mad = np.median(np.abs(amp - med))
    cutoff = med + 6 * 1.4826 * mad
    review = amp > cutoff if mad > 0 else np.zeros(len(X), dtype=bool)
    for row, flag in zip([r for r in rows if r["retained"]], review):
        row["review"] = bool(flag)
    counts = [int((y == label).sum()) for label in CLASS_NAMES]
    if min(counts) < 2:
        raise ValueError(f"Too few usable trials: {path.name}")
    return dict(subject=path.stem, path=path, X=X, eog=np.stack(eyes), y=y, ids=ids,
                fs=250., audit=pd.DataFrame(rows), review=review, cutoff=float(cutoff),
                counts=counts, n_blocks=len(runs), n_task_blocks=task_count,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


# %% Coherence and dissimilarities
def spectral_options(fs):
    n = int(fs)
    return dict(fs=fs, window=get_window("hann", n, fftbins=True), nperseg=n,
                noverlap=n // 2, nfft=n, detrend="constant", scaling="density",
                average="mean", axis=1)


def spectral_cache(X, fs):
    """One complex spectral matrix per trial and frequency, using original Welch settings.

    Retain only 8-30 Hz after estimation; this is frequency selection, not an
    added time-domain filter. Seven 1-s windows per 4-s trial, with 50% overlap.
    """
    options = spectral_options(fs)
    frequencies, auto = welch(X, **options)
    mask = (frequencies >= 8) & (frequencies <= 30)
    auto = auto[:, mask]
    if not np.isfinite(auto).all() or np.any(auto <= 0):
        raise ValueError("Invalid or zero spectral power")
    S = np.empty((len(X), mask.sum(), 22, 22), dtype=np.complex128)
    for i in range(22):
        S[:, :, i, i] = auto[:, :, i]
        for j in range(i + 1, 22):
            _, cross = csd(X[:, :, i], X[:, :, j], **options)
            S[:, :, i, j] = cross[:, mask]
            S[:, :, j, i] = cross[:, mask].conj()
    return frequencies[mask], S


def matrices_from_cache(frequencies, spectra, labels, indices):
    results = {}
    for label, movement in CLASS_NAMES.items():
        chosen = indices[labels[indices] == label]
        if len(chosen) < 2:
            raise ValueError(f"Too few trials for {movement}")
        pooled = spectra[chosen].mean(axis=0)  # complex average BEFORE magnitude
        power = np.diagonal(pooled, axis1=1, axis2=2).real
        coherence = np.abs(pooled) ** 2 / (power[:, :, None] * power[:, None, :])
        if not np.isfinite(coherence).all() or coherence.max() > 1 + 1e-10:
            raise ValueError("Invalid coherence")
        coherence = np.clip(coherence, 0, 1)
        for band, mask in {"mu": (frequencies >= 8) & (frequencies < 13),
                           "beta": (frequencies >= 13) & (frequencies <= 30)}.items():
            C = coherence[mask].mean(axis=0)
            C = (C + C.T) / 2
            np.fill_diagonal(C, 1)
            D = 1 - C
            np.fill_diagonal(D, 0)
            assert D.shape == (22, 22) and np.isfinite(D).all()
            assert np.allclose(D, D.T) and np.all((D >= 0) & (D <= 1))
            results[f"{movement}_{band}"] = dict(coherence=C, distance=D, n_trials=len(chosen))
    return results


def balanced_indices(labels, n_each, seed):
    rng = np.random.default_rng(int(seed))
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == c), n_each, replace=False)
        for c in CLASS_NAMES
    ]))


# %% Persistence, stable ranks and exact curve distances
def persistence(matrices):
    diagrams = {}
    for name, item in matrices.items():
        D = item["distance"]
        dgms = ripser(D, distance_matrix=True, maxdim=1, thresh=1.0, coeff=2)["dgms"]
        assert len(dgms[0]) == 22 and np.isinf(dgms[0][:, 1]).sum() == 1
        assert np.isfinite(dgms[1]).all()
        # Independent H0 check: finite deaths equal minimum-spanning-tree weights.
        deaths = np.sort(dgms[0][np.isfinite(dgms[0][:, 1]), 1])
        assert np.allclose(deaths, np.sort(minimum_spanning_tree(D).data), atol=1e-7)
        diagrams[name] = dgms
    return diagrams


def stable_rank(diagram, grid):
    lengths = diagram[:, 1] - diagram[:, 0]
    return (lengths[:, None] > np.asarray(grid)[None, :]).sum(axis=0)


def finite_lifetimes(diagram):
    return (diagram[:, 1] - diagram[:, 0])[np.isfinite(diagram[:, 1])]


def exact_rank_l1(a, b):
    """Exact integral of |R_a - R_b|; equal infinite tails cancel."""
    if np.isinf(a[:, 1]).sum() != np.isinf(b[:, 1]).sum():
        return np.inf
    lengths = np.r_[finite_lifetimes(a), finite_lifetimes(b)]
    grid = np.unique(np.r_[0., lengths, 1.])
    area = np.sum(np.abs(stable_rank(a, grid)[:-1] - stable_rank(b, grid)[:-1]) * np.diff(grid))
    return float(area)


def pairwise_rank_distances(ph):
    result = {}
    for band in BANDS:
        for dim in [0, 1]:
            M = np.zeros((4, 4))
            for i in range(4):
                for j in range(i + 1, 4):
                    M[i, j] = M[j, i] = exact_rank_l1(ph[f"{MOVEMENTS[i]}_{band}"][dim],
                                                    ph[f"{MOVEMENTS[j]}_{band}"][dim])
            result[f"{band}_H{dim}"] = M
    return result


def feature_rows(ph, subject, analysis):
    rows = []
    for name, dgms in ph.items():
        movement, band = name.rsplit("_", 1)
        for dim, dgm in enumerate(dgms):
            lengths = finite_lifetimes(dgm)
            rows.append(dict(subject=subject, analysis=analysis, movement=movement, band=band,
                             homology=f"H{dim}", finite_bars=len(lengths),
                             total_persistence=float(lengths.sum()),
                             longest_lifespan=float(lengths.max(initial=0)),
                             stable_rank_005=int(stable_rank(dgm, [0.05])[0]),
                             stable_rank_010=int(stable_rank(dgm, [0.10])[0])))
    return rows


def save_topology(folder, prefix, matrices, ph, ids, labels, indices, n_each, seed):
    arrays = dict(selected_indices=indices, trial_ids=ids[indices], labels=labels[indices],
                  trials_per_movement=n_each, seed=seed, lifespan_grid=GRID)
    for name in NAMES:
        arrays[f"C_{name}"] = matrices[name]["coherence"]
        arrays[f"D_{name}"] = matrices[name]["distance"]
        for dim in [0, 1]:
            arrays[f"H{dim}_{name}"] = ph[name][dim]
            arrays[f"SR_H{dim}_{name}"] = stable_rank(ph[name][dim], GRID)
    arrays.update({f"L1_{key}": value for key, value in pairwise_rank_distances(ph).items()})
    np.savez_compressed(folder / f"{prefix}.npz", **arrays)


# %% Per-participant figures
def save_figure(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_quality(p, folder):
    amp = np.ptp(p["X"], axis=1).max(axis=1)
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    ax.plot(amp, ".", color=COLORS[0])
    ax.axhline(p["cutoff"], ls="--", color=COLORS[3], label="Review threshold")
    ax.set(xlabel="Retained trial index (0-based)", ylabel="Maximum channel peak-to-peak (microvolts)",
           title=f"{p['subject']}: amplitude review; flagged trials remain in main analysis")
    ax.legend()
    save_figure(fig, folder / "quality_review.png")
    # Save every flagged trial and the largest trial for inspection.
    for k in np.unique(np.r_[np.flatnonzero(p["review"]), amp.argmax()]):
        x = p["X"][k] - p["X"][k].mean(axis=0)
        t = np.arange(1000) / p["fs"]
        spacing = max(60., amp[k] * 1.2)
        fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
        offsets = np.arange(22)[::-1] * spacing
        axes[0].plot(t, x + offsets, lw=.5)
        axes[0].set_yticks(offsets, CHANNELS, fontsize=7)
        axes[0].set(xlabel="Seconds after cue", title="EEG, centered and offset for display")
        axes[1].plot(t, p["eog"][k], lw=.6)
        axes[1].legend(["EOG1", "EOG2", "EOG3"])
        axes[1].set(xlabel="Seconds after cue", ylabel="Microvolts", title="Eye signals from the same trial")
        fig.suptitle(f"{p['subject']}: original (block, trial) = {tuple(p['ids'][k])}")
        save_figure(fig, folder / f"review_trial_{k:03d}.png")


def plot_matrices(matrices, subject, folder):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    for r, band in enumerate(BANDS):
        for c, movement in enumerate(MOVEMENTS):
            ax = axes[r, c]
            im = ax.imshow(matrices[f"{movement}_{band}"]["distance"], vmin=0, vmax=1, cmap="viridis")
            ax.set(title=f"{movement.replace('_', ' ')} | {band}", xlabel="EEG index", ylabel="EEG index")
    fig.colorbar(im, ax=axes, shrink=.8, label="D = 1 - coherence")
    fig.suptitle(f"{subject}: balanced channel dissimilarities")
    save_figure(fig, folder / "balanced_distances.png")


def plot_barcodes(ph, subject, folder):
    finite_deaths = [d[np.isfinite(d[:, 1]), 1].max(initial=0) for ds in ph.values() for d in ds]
    limit = min(1., max(.1, np.ceil(max(finite_deaths) * 1.1 * 20) / 20))
    for band in BANDS:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
        for c, movement in enumerate(MOVEMENTS):
            for dim in [0, 1]:
                dgm = ph[f"{movement}_{band}"][dim]
                ax = axes[dim, c]
                for row, (birth, death) in enumerate(dgm[np.argsort(dgm[:, 0], kind="stable")]):
                    ax.hlines(row, birth, limit if np.isinf(death) else death,
                              color=COLORS[dim], lw=2)
                    if np.isinf(death):
                        ax.plot(limit, row, ">", color=COLORS[dim], clip_on=False)
                ax.set(xlim=(0, limit), ylim=(-1, max(len(dgm), 1)), yticks=[],
                       xlabel="Distance threshold", title=f"{movement.replace('_', ' ')}: H{dim}")
        fig.suptitle(f"{subject}: {band} barcodes; arrow = infinite H0 interval")
        save_figure(fig, folder / f"barcodes_{band}.png")


def plot_ranks(ph, subject, folder, selection=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    limit = max(.1, 1.1 * max(finite_lifetimes(d).max(initial=0) for ds in ph.values() for d in ds))
    for col, band in enumerate(BANDS):
        for dim in [0, 1]:
            ax = axes[dim, col]
            for movement, color in zip(MOVEMENTS, COLORS):
                name = f"{movement}_{band}"
                if selection is None:
                    grid = np.unique(np.r_[0, finite_lifetimes(ph[name][dim]), 1])
                    ax.step(grid, stable_rank(ph[name][dim], grid), where="post", color=color,
                            label=movement.replace("_", " "), lw=1.7)
                else:
                    v = selection[name][:, dim]
                    ax.step(GRID, np.median(v, axis=0), where="post", color=color,
                            label=movement.replace("_", " "), lw=1.7)
                    ax.fill_between(GRID, v.min(axis=0), v.max(axis=0), step="post", color=color, alpha=.18)
            ax.set(xlim=(0, limit), ylim=(0, None), title=f"{band} band: H{dim}",
                   xlabel="Minimum bar lifespan", ylabel="Stable rank")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    title = "standard stable ranks; infinite H0 retained" if selection is None else "20 selections: median and observed min-max (not a CI)"
    fig.suptitle(f"{subject}: {title}")
    save_figure(fig, folder / ("stable_ranks.png" if selection is None else "selection_sensitivity.png"))


# %% Complete one participant
def analyse_subject(path, output_root, common_n, save_epochs=False):
    p = prepare_subject(path)
    sid = p["subject"]
    folder = Path(output_root) / sid
    folder.mkdir(parents=True, exist_ok=True)
    p["audit"].to_csv(folder / "trial_audit.csv", index=False)
    if save_epochs:
        np.savez_compressed(folder / "prepared_epochs.npz", epochs=p["X"], eog=p["eog"],
                            labels=p["y"], trial_ids=p["ids"], fs=p["fs"])
    f, S = spectral_cache(p["X"], p["fs"])
    all_indices = np.arange(len(p["y"]))
    all_matrices = matrices_from_cache(f, S, p["y"], all_indices)
    np.savez_compressed(folder / "all_retained_distances.npz",
                        **{f"{key}_{name}": item[field] for name, item in all_matrices.items()
                           for key, field in [("C", "coherence"), ("D", "distance")]})
    sensitivity_rows = []
    without = all_matrices
    if p["review"].any():
        without = matrices_from_cache(f, S, p["y"], all_indices[~p["review"]])
    for name in NAMES:
        delta = np.abs(all_matrices[name]["distance"] - without[name]["distance"])[np.triu_indices(22, 1)]
        sensitivity_rows.append(dict(subject=sid, matrix=name, n_review=int(p["review"].sum()),
                                     mean_change=float(delta.mean()), max_change=float(delta.max())))
    pd.DataFrame(sensitivity_rows).to_csv(folder / "amplitude_sensitivity.csv", index=False)
    n_each = min(p["counts"])
    idx = balanced_indices(p["y"], n_each, MAIN_SEED)
    matrices = matrices_from_cache(f, S, p["y"], idx)
    ph = persistence(matrices)
    save_topology(folder, "balanced_topology", matrices, ph, p["ids"], p["y"], idx, n_each, MAIN_SEED)
    for name, item in matrices.items():
        pd.DataFrame(item["distance"], index=CHANNELS, columns=CHANNELS).to_csv(folder / f"D_{name}.csv")
    features = feature_rows(ph, sid, "main")
    selection = {name: np.empty((len(SELECTION_SEEDS), 2, len(GRID)), dtype=np.int16) for name in NAMES}
    selection_ids, selection_features, selection_l1 = [], [], []
    for repeat, seed in enumerate(SELECTION_SEEDS):
        chosen = balanced_indices(p["y"], n_each, seed)
        selection_ids.append(chosen)
        repeat_ph = persistence(matrices_from_cache(f, S, p["y"], chosen))
        for name in NAMES:
            for dim in [0, 1]:
                selection[name][repeat, dim] = stable_rank(repeat_ph[name][dim], GRID)
                selection_l1.append(dict(subject=sid, seed=int(seed), matrix=name, homology=f"H{dim}",
                                         l1_to_main=exact_rank_l1(ph[name][dim], repeat_ph[name][dim])))
        for row in feature_rows(repeat_ph, sid, "selection"):
            row["seed"] = int(seed)
            selection_features.append(row)
    np.savez_compressed(folder / "selection_sensitivity.npz", seeds=SELECTION_SEEDS,
                        selected_indices=np.stack(selection_ids), trial_ids=p["ids"][np.stack(selection_ids)],
                        lifespan_grid=GRID, trials_per_movement=n_each,
                        **{f"SR_{name}": v for name, v in selection.items()})
    pd.DataFrame(selection_features).to_csv(folder / "selection_features.csv", index=False)
    pd.DataFrame(selection_l1).to_csv(folder / "selection_l1_to_main.csv", index=False)
    common_idx = balanced_indices(p["y"], common_n, MAIN_SEED)
    common_matrices = matrices_from_cache(f, S, p["y"], common_idx)
    common_ph = persistence(common_matrices)
    save_topology(folder, "common_count_topology", common_matrices, common_ph, p["ids"], p["y"],
                  common_idx, common_n, MAIN_SEED)
    features += feature_rows(common_ph, sid, "common_count")
    pd.DataFrame(features).to_csv(folder / "topology_features.csv", index=False)
    settings = dict(source_url=BASE_URL + path.name, sha256=p["sha256"], fs=250,
                    sample_window_after_trial_marker=[500, 1500], seconds_after_trial_marker=[2, 6],
                    eeg_columns="0-21", eog_columns="22-24 (review only)",
                    reference="original left mastoid", added_filter="none", normalization="none",
                    detrending="subtract each Welch window mean", window="periodic Hann",
                    nperseg=250, noverlap=125, nfft=250, windows_per_trial=7,
                    bands={"mu": "8 <= f < 13", "beta": "13 <= f <= 30"},
                    spectral_pooling="complex cross spectra and auto spectra averaged over trials before coherence",
                    main_seed=MAIN_SEED, selection_seeds=SELECTION_SEEDS.tolist(),
                    primary_n_each=n_each, common_n_each=common_n,
                    review_cutoff_uv=p["cutoff"], review_original_ids=p["ids"][p["review"]].tolist(),
                    review_rule="median + 6 * 1.4826 * MAD; review only",
                    filtration="weighted flag complex; D=1-bandmean magnitude-squared coherence",
                    maxdim=1, coeff=2, threshold=1.0,
                    stable_rank="count death-birth > t; infinite H0 retained",
                    total_persistence="sum of finite lifetimes (infinite H0 omitted only from scalar area)",
                    versions={pkg: importlib.metadata.version(pkg) for pkg in ["numpy", "scipy", "ripser", "matplotlib", "pandas"]})
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    plot_quality(p, folder)
    plot_matrices(matrices, sid, folder)
    plot_barcodes(ph, sid, folder)
    plot_ranks(ph, sid, folder)
    plot_ranks(ph, sid, folder, selection)
    summary = dict(subject=sid, blocks=p["n_blocks"], task_blocks=p["n_task_blocks"],
                   original=len(p["audit"]), excluded=len(p["audit"]) - len(p["y"]), retained=len(p["y"]),
                   **dict(zip(MOVEMENTS, p["counts"])), balanced_each=n_each,
                   balanced_total=4*n_each, review_candidates=int(p["review"].sum()))
    print(f"{sid}: retained {summary['retained']}; {n_each}/movement; 20 selections finished", flush=True)
    return summary


# %% Group comparisons: nine participants are the independent units
def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    return adjusted


def permutation_friedman(values, n_permutations=99999, seed=2026):
    """Monte Carlo within-participant label permutations; no chi-square approximation.

    Tests exchangeability of the four movement summaries within participants.
    Uses one finite total-persistence summary per participant/movement.
    """
    values = np.asarray(values, float)
    n, k = values.shape
    ranks = rankdata(values, axis=1)
    ties = sum(sum(c**3-c for c in np.unique(row, return_counts=True)[1]) for row in values)
    correction = 1 - ties / (n * (k**3-k))
    if correction <= 0:
        return 0., 1., 0.
    def statistic(rank_sums):
        return (12 / (n*k*(k+1)) * np.square(rank_sums).sum(axis=-1) - 3*n*(k+1)) / correction
    observed = float(statistic(ranks.sum(axis=0)))
    assert np.isclose(observed, friedmanchisquare(*values.T).statistic)
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_permutations, 1000):
        b = min(1000, n_permutations-start)
        permutations = np.argsort(rng.random((b, n, k)), axis=-1)
        shuffled = np.take_along_axis(np.broadcast_to(ranks, (b,n,k)), permutations, axis=2)
        exceed += int((statistic(shuffled.sum(axis=1)) >= observed - 1e-10).sum())
    p = (exceed + 1) / (n_permutations + 1)
    return observed, float(p), float(np.sqrt(p*(1-p)/(n_permutations+1)))


def group_analysis(output_root, subjects):
    root = Path(output_root)
    folders = [root / sid for sid in subjects]
    features = pd.concat([pd.read_csv(f / "topology_features.csv") for f in folders], ignore_index=True)
    features.to_csv(root / "all_topology_features.csv", index=False)
    amplitude = pd.concat([pd.read_csv(f / "amplitude_sensitivity.csv") for f in folders], ignore_index=True)
    amplitude.to_csv(root / "all_amplitude_sensitivity.csv", index=False)
    selections = pd.concat([pd.read_csv(f / "selection_features.csv") for f in folders], ignore_index=True)
    selections.to_csv(root / "all_selection_features.csv", index=False)
    l1_variation = pd.concat([pd.read_csv(f / "selection_l1_to_main.csv") for f in folders], ignore_index=True)
    l1_variation.to_csv(root / "all_selection_l1.csv", index=False)
    tests = []
    for analysis in ["main", "common_count"]:
        these = []
        for band in BANDS:
            for homology in ["H0", "H1"]:
                frame = features.query("analysis == @analysis and band == @band and homology == @homology")
                values = frame.pivot(index="subject", columns="movement", values="total_persistence").loc[subjects, MOVEMENTS].values
                stat, p, mcse = permutation_friedman(values)
                these.append(dict(analysis=analysis, band=band, homology=homology, n_subjects=len(subjects),
                                  friedman_Q=stat, kendall_W=stat/(len(subjects)*3), permutation_p=p,
                                  monte_carlo_se=mcse, permutations=99999, seed=2026))
        for row, adjusted in zip(these, holm_adjust([r["permutation_p"] for r in these])):
            row["holm_p"] = float(adjusted)
        tests.extend(these)
    tests = pd.DataFrame(tests)
    tests.to_csv(root / "exploratory_omnibus_tests.csv", index=False)
    main = features.query("analysis == 'main'")
    group_summary = main.groupby(["band", "homology", "movement"]).agg(
        median_total=("total_persistence", "median"), min_total=("total_persistence", "min"),
        max_total=("total_persistence", "max"), median_longest=("longest_lifespan", "median"),
        median_finite_bars=("finite_bars", "median")).reset_index()
    group_summary.to_csv(root / "group_feature_summary.csv", index=False)
    pair_rows = []
    loaded = {sid: np.load(root / sid / "balanced_topology.npz") for sid in subjects}
    for sid in subjects:
        for band in BANDS:
            for dim in [0, 1]:
                M = loaded[sid][f"L1_{band}_H{dim}"]
                for i in range(4):
                    for j in range(i+1, 4):
                        pair_rows.append(dict(subject=sid, band=band, homology=f"H{dim}",
                                              movement_1=MOVEMENTS[i], movement_2=MOVEMENTS[j], l1=M[i,j]))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(root / "within_subject_pairwise_l1.csv", index=False)
    pairs.groupby(["band","homology","movement_1","movement_2"]).l1.agg(["median","min","max"]).reset_index().to_csv(root / "group_pairwise_l1_summary.csv", index=False)
    # A01-generated descriptive pattern, not a pre-specified confirmatory hypothesis.
    contrasts = []
    for analysis in ["main", "common_count"]:
        for band in BANDS:
            for homology in ["H0", "H1"]:
                x = features.query("analysis == @analysis and band == @band and homology == @homology")
                v = x.pivot(index="subject", columns="movement", values="total_persistence").loc[subjects]
                difference = v[["feet","tongue"]].mean(axis=1)-v[["left_hand","right_hand"]].mean(axis=1)
                for sid, value in difference.items():
                    contrasts.append(dict(subject=sid, analysis=analysis, band=band, homology=homology,
                                          feet_tongue_minus_hands=float(value)))
    pd.DataFrame(contrasts).to_csv(root / "exploratory_direction_contrasts.csv", index=False)
    # Paired features: each grey line belongs to one participant.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for col, band in enumerate(BANDS):
        for dim in [0,1]:
            h=f"H{dim}"
            v=main.query("band == @band and homology == @h").pivot(index="subject", columns="movement", values="total_persistence").loc[subjects,MOVEMENTS]
            ax=axes[dim,col]
            ax.plot(np.arange(4), v.values.T, color="#9AA4B0", alpha=.75, lw=.9, marker=".")
            ax.plot(np.arange(4), v.median(axis=0), color="#172C49", lw=2.5, marker="o", label="Across-participant median")
            ax.set_xticks(range(4),["Left hand","Right hand","Feet","Tongue"])
            ax.set(title=f"{band}: {h}", ylabel="Sum of finite bar lifespans")
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Paired total persistence: each grey line is one participant (n = 9)")
    save_figure(fig, root / "group_total_persistence.png")
    # Median of within-participant distances, not distance between averaged EEG.
    fig, axes=plt.subplots(2,2,figsize=(9,8),layout="constrained")
    for col,band in enumerate(BANDS):
        for dim in [0,1]:
            ax=axes[dim,col]
            M=np.median(np.stack([loaded[s][f"L1_{band}_H{dim}"] for s in subjects]),axis=0)
            im=ax.imshow(M,cmap="Blues",vmin=0)
            for i in range(4):
                for j in range(4):ax.text(j,i,f"{M[i,j]:.3f}",ha="center",va="center",color="white" if M[i,j]>M.max()*.65 else "#152238")
            ax.set_xticks(range(4),["Left","Right","Feet","Tongue"])
            ax.set_yticks(range(4),["Left","Right","Feet","Tongue"])
            ax.set_title(f"{band}: H{dim}")
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle("Median within-participant L1 distance between stable-rank curves")
    save_figure(fig,root / "group_pairwise_distances.png")
    # All participants on identical scales, by band and homology.
    for band in BANDS:
        for dim in [0,1]:
            xmax=max(.1,1.05*max(finite_lifetimes(loaded[s][f"H{dim}_{m}_{band}"]).max(initial=0) for s in subjects for m in MOVEMENTS))
            ymax=max(stable_rank(loaded[s][f"H{dim}_{m}_{band}"],[0])[0] for s in subjects for m in MOVEMENTS)
            fig,axes=plt.subplots(3,3,figsize=(12,8),layout="constrained",sharex=True,sharey=True)
            for ax,sid in zip(axes.ravel(),subjects):
                for m,color in zip(MOVEMENTS,COLORS):
                    dgm=loaded[sid][f"H{dim}_{m}_{band}"]
                    g=np.unique(np.r_[0,finite_lifetimes(dgm),1])
                    ax.step(g,stable_rank(dgm,g),where="post",color=color,label=m.replace('_',' '),lw=1.4)
                ax.set(title=sid, xlim=(0,xmax),ylim=(0,ymax+1));ax.grid(alpha=.18)
            axes[0,0].legend(fontsize=7)
            fig.supxlabel("Minimum bar lifespan");fig.supylabel("Stable rank")
            fig.suptitle(f"{band} band, H{dim}: all nine participants")
            save_figure(fig,root/f"all_subjects_{band}_H{dim}.png")
    for z in loaded.values():z.close()
    return features, tests, pairs


def run_all(data_dir, output_root, subjects=range(1,10), download_missing=True, save_epochs=False):
    output_root=Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    paths=download_files(data_dir,subjects) if download_missing else [Path(data_dir)/f"A{i:02d}T.mat" for i in subjects]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:raise FileNotFoundError("Missing files: "+", ".join(missing))
    # First pass determines the common sample size using quality exclusions only.
    min_counts=[]
    for path in paths:
        prepared=prepare_subject(path)
        min_counts.append(min(prepared["counts"]))
        del prepared
    common_n=min(min_counts)
    print(f"Common-count sensitivity: {common_n} trials per movement for every participant",flush=True)
    summaries=[]
    for path in paths:
        summaries.append(analyse_subject(path,output_root,common_n,save_epochs))
        pd.DataFrame(summaries).to_csv(output_root/"participant_summary.csv",index=False)
    features,tests,pairs=group_analysis(output_root,[p.stem for p in paths])
    return pd.DataFrame(summaries),features,tests,pairs
