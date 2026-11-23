"""Helper functions for the health dataset, called by the notebook 03_report_health.

It holds the loading of the cleaned files and the computations that are too long for the notebook.
"""
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

FOLDER = Path("../data/health_clean")

# repeated measurements kept for the clustering
KEPT_REPEATED_MEASURES = ["PPULSE", "PULSE_GAP", "RPULSE", "AGE_GAP", "BALANCE", "CARDOUT", "COGNITIVE",
                          "INFLAM", "LATSWAY", "MOBILITY", "O2SAT", "THORPEAK", "THORVIB"]


def load(folder=FOLDER):
    """Loads the cleaned files into a dictionary."""
    d = Path(folder)
    t = {}
    t["P"] = pd.read_csv(d / "patient_administration.csv", parse_dates=["date_of_birth"])
    t["E"] = pd.read_csv(d / "clinical_encounters.csv", parse_dates=["encounter_start", "encounter_end"])
    t["O"] = pd.read_csv(d / "clinical_observations.csv")
    t["PR"] = pd.read_csv(d / "clinical_procedures.csv", parse_dates=["performed_time"])
    t["W"] = pd.read_csv(d / "observations_wide.csv", parse_dates=["encounter_start"])
    t["CAT"] = pd.read_csv(d / "observation_catalog.csv").set_index("observation_code")
    t["CR"] = pd.read_csv(d / "cardiac_recordings.csv", parse_dates=["recording_date"])
    t["SIG"] = pd.read_csv(d / "cardiac_signal_samples.csv")
    t["N"] = pd.read_csv(d / "clinical_notes.csv")
    t["LOG"] = pd.read_csv(d / "cleaning_log.csv")

    # size of the 7 original files (before adding columns)
    names = {"patient_administration": "P", "clinical_encounters": "E", "clinical_observations": "O",
             "clinical_procedures": "PR", "cardiac_recordings": "CR", "cardiac_signal_samples": "SIG",
             "clinical_notes": "N"}
    t["sizes"] = pd.DataFrame({name: {"rows": len(t[k]), "columns": t[k].shape[1]} for name, k in names.items()}).T

    cat = t["CAT"]
    numeric = cat["type"] == "number"
    t["REPEATED"] = cat.index[(cat["n_measures"] > 241) & numeric].tolist()   # measured at every encounter
    t["SINGLE"] = cat.index[(cat["n_measures"] <= 241) & numeric].tolist()    # measured once per patient
    t["MEDICAL"] = KEPT_REPEATED_MEASURES + t["SINGLE"]

    E = t["E"]
    E["duration"] = (E["encounter_end"] - E["encounter_start"]).dt.total_seconds() / 60
    E["day"] = E["encounter_start"].dt.date.astype(str)
    return t


def cleaning_summary(log):
    """Cleaning log grouped by type of operation."""
    log = log.copy()
    log["operation"] = log["action"].str.split(": ").str[0]
    summary = log.groupby("operation", sort=False).agg(rows_affected=("n", "sum"))
    summary["files"] = log.groupby("operation", sort=False)["table"].unique().str.join(", ")
    return summary


def _slope(d, col):
    d = d[["T", col]].dropna()
    if len(d) < 3:
        return np.nan
    return np.polyfit(d["T"], d[col], 1)[0]


def patient_features(t):
    """Builds a table per encounter (W2) and ~90 variables per patient (feat)."""
    P, E, O, PR, W = t["P"], t["E"], t["O"], t["PR"], t["W"]
    REPEATED, SINGLE = t["REPEATED"], t["SINGLE"]
    visits = W.merge(P[["patient_number", "date_of_birth"]], on="patient_number")
    visits["AGE"] = (visits["encounter_start"] - visits["date_of_birth"]).dt.days / 365.25
    W2 = visits[["encounter_number", "patient_number", "encounter_start"] + REPEATED + ["AGE"]].copy()
    W2["AGE_GAP"] = W2["BIOAGE"] - W2["AGE"]
    W2["PULSE_GAP"] = W2["APICAL"] - W2["PPULSE"]
    W2["T"] = (W2["encounter_start"] - pd.Timestamp("2014-01-01")).dt.days / 365.25
    W2 = W2.sort_values(["patient_number", "encounter_start"])
    g = W2.groupby("patient_number")

    # measurements taken only once, and the two scales none < mild < moderate < severe
    singles = O[O["observation_code"].isin(SINGLE)].pivot(index="patient_number", columns="observation_code",
                                                           values="numeric_value")
    scales = O[O["observation_code"].isin(["MUSC", "NEURO"])].pivot(index="patient_number", columns="observation_code",
                                                                    values="text_value")
    order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    for c in scales.columns:
        scales[c] = scales[c].map(order)
    singles = singles.join(scales)

    # median, variability and slope of the repeated measurements
    feat = g[REPEATED + ["AGE", "AGE_GAP", "PULSE_GAP"]].median()
    feat = feat.join(g[REPEATED + ["PULSE_GAP"]].std().add_suffix("_STD"))
    for c in ["BIOAGE", "INFLAM", "MOBILITY", "COGNITIVE", "LATSWAY", "THORPEAK", "PULSE_GAP", "CARDOUT"]:
        feat[c + "_SLOPE"] = g.apply(lambda d: _slope(d, c))
    # largest drop of the age gap between two visits
    with_age = W2.dropna(subset=["AGE_GAP"]).copy()
    with_age["change"] = with_age.groupby("patient_number")["AGE_GAP"].diff()
    feat["MAX_AGE_DROP"] = with_age.groupby("patient_number")["change"].min()
    feat = feat.join(singles)

    # anomaly counts
    def count_where(mask):
        return O[mask].groupby("patient_number").size()

    feat["N_OCULAR_ATYP"] = count_where((O["observation_code"] == "OCULAR") & (O["text_value"] == "atypical"))
    feat["BINOCOORD_ATYP"] = count_where((O["observation_code"] == "BINOCOORD") & (O["text_value"] == "atypical"))
    feat["VESTIB_ABNORMAL"] = count_where((O["observation_code"] == "VESTIB") & (O["text_value"] == "abnormal"))
    feat["N_DISCORDANT"] = count_where(O["interpretation_code"] == "discordant")
    feat["N_REVIEW"] = count_where(O["interpretation_code"] == "review")
    feat["N_OUT_OF_RANGE"] = count_where(O["out_of_ref_range"])

    # encounters, procedures, ECGs, notes
    ge = E.groupby("patient_number")
    feat["N_VISITS"] = ge.size()
    feat["MEAN_DURATION"] = ge["duration"].mean()
    feat["CONFIDENTIAL_SHARE"] = ge["confidentiality_class"].apply(lambda s: (s != "standard").mean())
    for typ in E["encounter_type"].unique():
        feat["VISIT_" + typ] = E[E["encounter_type"] == typ].groupby("patient_number").size()
    for r in E["referral_source"].dropna().unique():
        feat["REFERRED_" + r.replace(" ", "_")] = E[E["referral_source"] == r].groupby("patient_number").size()
    for c in PR["procedure_code"].unique():
        feat["PROC_" + c] = PR[PR["procedure_code"] == c].groupby("patient_number").size()
    feat["N_ECG"] = t["CR"].groupby("patient_number").size()
    feat["N_NOTES"] = t["N"].groupby("patient_number").size()
    counts = [c for c in feat.columns if c.startswith(("N_", "VISIT_", "REFERRED_", "PROC_", "BINOCOORD_", "VESTIB_"))]
    feat[counts] = feat[counts].fillna(0)
    feat["N_MRP42"] = feat["PROC_MRP-42"]
    return W2, feat


def correlation_sweep(feat):
    """Pearson and Spearman on all pairs of variables (excluding pairs of the same measurement)."""
    s, p = feat.corr("spearman"), feat.corr("pearson")
    cols = list(feat.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, c = cols[i], cols[j]
            if a.split("_")[0] == c.split("_")[0]:   # e.g. BIOAGE and BIOAGE_STD: obvious pair
                continue
            rows.append((a, c, p.loc[a, c], s.loc[a, c]))
    b = pd.DataFrame(rows, columns=["variable 1", "variable 2", "pearson", "spearman"]).dropna()
    b["gap"] = b["pearson"] - b["spearman"]
    return b


def abod(Z):
    """ABOD: for each point, variance of the angles towards all pairs of other points.
    An isolated point sees the others under similar angles, so its variance is low."""
    result = np.empty(len(Z))
    for i in range(len(Z)):
        V = np.delete(Z, i, axis=0) - Z[i]        # vectors from point i to each of the others
        n2 = (V ** 2).sum(axis=1)                 # square of their length
        M = (V @ V.T) / np.outer(n2, n2)          # dot product of each pair, weighted by the distances
        upper = np.triu_indices(len(V), 1)        # each pair only once
        result[i] = M[upper].var()
    return result


def individual_breaks(W2, E, col="AGE_GAP", threshold=6, min_visits=4):
    """Visits that are abnormal relative to the patient's history (the patient's modified z-score)."""
    rows = []
    for pid, d in W2.groupby("patient_number"):
        s = d.set_index("encounter_number")[col].dropna()
        if len(s) < min_visits:
            continue
        abs_dev = max((s - s.median()).abs().median(), 0.5)
        z = 0.6745 * (s - s.median()) / abs_dev
        for enc, v in z[z.abs() > threshold].items():
            rows.append((pid, enc, s[enc], s.median(), v))
    r = pd.DataFrame(rows, columns=["patient", "encounter", "age gap", "patient median", "modified z"])
    info = E.set_index("encounter_number")
    r["date"] = r["encounter"].map(info["day"])
    r["type"] = r["encounter"].map(info["encounter_type"])
    return r


def simultaneous_encounters(E):
    """Pairs of encounters of the same doctor that overlap in time."""
    rows = []
    for doctor, d in E.sort_values("encounter_start").groupby("attending_license_number"):
        d = d.reset_index(drop=True)
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if d.loc[j, "encounter_start"] >= d.loc[i, "encounter_end"]:
                    break
                rows.append((doctor, d.loc[i, "day"], d.loc[i, "facility_license_number"] != d.loc[j, "facility_license_number"]))
    return pd.DataFrame(rows, columns=["doctor", "day", "two clinics at once"])


def doctor_similarity(E):
    """Profile of each patient (share of their encounters per doctor) and similarity between patients."""
    profile = E.groupby(["patient_number", "attending_license_number"]).size().unstack(fill_value=0)
    profile = profile.div(profile.sum(axis=1), axis=0)
    similarity = (profile @ profile.T).to_numpy(copy=True)
    np.fill_diagonal(similarity, 0)
    return profile, similarity


def patient_graph(profile, similarity, quantile):
    """Patient-patient graph keeping the pairs more similar than the given quantile."""
    import networkx as nx
    threshold = np.quantile(similarity[np.triu_indices(len(similarity), 1)], quantile)
    G = nx.Graph()
    G.add_nodes_from(profile.index)
    ii, jj = np.where(np.triu(similarity > threshold, 1))
    G.add_weighted_edges_from((profile.index[i], profile.index[j], similarity[i, j]) for i, j in zip(ii, jj))
    return G


def event_log(E, PR, CR):
    """Event log (encounters, procedures, ECGs) and one trace per patient."""
    ev = pd.concat([
        pd.DataFrame({"patient": E["patient_number"], "t": E["encounter_start"], "step": E["encounter_type"]}),
        pd.DataFrame({"patient": PR["patient_number"], "t": PR["performed_time"], "step": "procedure " + PR["procedure_code"]}),
        pd.DataFrame({"patient": CR["patient_number"], "t": CR["recording_date"], "step": "ECG " + CR["recording_context"]}),
    ]).sort_values(["patient", "t"])
    traces = ev.groupby("patient")["step"].apply(lambda s: ("START",) + tuple(s) + ("END",))
    return ev, traces


def transitions(traces):
    """Count of the direct successions A -> B (directly-follows graph)."""
    c = Counter()
    for t in traces:
        c.update(zip(t, t[1:]))
    return c


def dependencies(traces):
    """Heuristic Miner dependency degree: (|A>B| - |B>A|) / (|A>B| + |B>A| + 1)."""
    c = transitions(traces)
    dep = pd.Series({(a, b): (n - c[(b, a)]) / (n + c[(b, a)] + 1) for (a, b), n in c.items()
                     if a != b and a != "START" and b != "END"})
    return dep.sort_values(ascending=False)


def mrp42_lots(PR, occ):
    """Lot numbers of the MRP-42 injections and lot block of each executive."""
    lots = PR[PR["procedure_code"] == "MRP-42"].copy()
    lots["series"] = np.where(lots["material_reference"].str.startswith("MTL-R"), "research (MTL-R)", "executives (MTL)")
    lots["number"] = lots["material_reference"].str[-4:].astype(int)
    blocks = (lots[lots["series"] == "executives (MTL)"].groupby("patient_number")
              .agg(lots=("number", lambda s: f"{s.min()} to {s.max()}"), first_lot=("number", "min"),
                   first_injection=("performed_time", lambda s: s.min().date()))
              .sort_values("first_lot").join(occ))
    return lots, blocks


def dose_effects(PR, W2):
    """Change of the biological age gap between the visit just before and just after each dose."""
    with_age = W2.dropna(subset=["AGE_GAP"])
    doses = PR[PR["procedure_code"] == "MRP-42"].sort_values("performed_time")
    doses = doses.assign(rank=doses.groupby("patient_number").cumcount() + 1)
    rows = []
    for r in doses.itertuples():
        v = with_age[with_age["patient_number"] == r.patient_number]
        before = v[v["encounter_start"] < r.performed_time].tail(1)
        after = v[v["encounter_start"] > r.performed_time].head(1)
        if len(before) and len(after):
            rows.append((r.patient_number, r.authorization_class, r.rank, after["AGE_GAP"].iloc[0] - before["AGE_GAP"].iloc[0]))
    return pd.DataFrame(rows, columns=["patient", "class", "dose rank", "effect (years)"])


def dtw(a, b):
    """Dynamic Time Warping distance between two series (cost = absolute difference)."""
    D = np.full((len(a) + 1, len(b) + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[-1, -1]


def trajectories(W2, col="AGE_GAP"):
    """Trajectory of each patient, brought back to 0 at the first visit."""
    t = {p: d.sort_values("encounter_start")[col].dropna().values for p, d in W2.groupby("patient_number")}
    return {p: s - s[0] for p, s in t.items() if len(s) >= 2}


def dtw_matrix(traj):
    ids = list(traj)
    dist = np.zeros((len(ids), len(ids)))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dist[i, j] = dist[j, i] = dtw(traj[ids[i]], traj[ids[j]])
    return ids, dist


def ecg_synchrony(CR, SIG, height=.9, min_gap_s=.3, tolerance_s=.05):
    """Detects the beats on the 2 channels of each ECG and measures the share of synchronous beats."""
    rows = []
    for r in CR.itertuples():
        s = SIG[SIG["recording_id"] == r.recording_id].sort_values("time_ms")
        times = s["time_ms"].values / 1000
        min_gap = int(min_gap_s * r.sampling_rate_hz)   # in number of samples
        peaks_a, _ = find_peaks(s["channel_a_normalized"].values, height=height, distance=min_gap)
        peaks_b, _ = find_peaks(s["channel_b_normalized"].values, height=height, distance=min_gap)
        a, b = times[peaks_a], times[peaks_b]
        # a beat of channel A is synchronous if there is a beat of channel B within tolerance_s
        synchronous = [np.min(np.abs(b - x)) < tolerance_s for x in a]
        rows.append({"recording": r.recording_id, "patient": r.patient_number, "context": r.recording_context,
                     "bpm channel A": round(60 / np.median(np.diff(a))), "bpm channel B": round(60 / np.median(np.diff(b))),
                     "synchronous beats": round(np.mean(synchronous), 2)})
    return pd.DataFrame(rows)


def prepare_text(text):
    """Lowercase, removal of identifiers, punctuation, digits and stop words."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    t = re.sub(r"(pat|enc)-\d+", " ", text.lower())
    t = re.sub(r"[^a-z\s-]", " ", t)
    return " ".join(w for w in t.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2)


def unique_sentences(notes, excluded_notes=()):
    """Sentences present in a single note, excluding some notes and the first line of the templates."""
    rows = []
    for r in notes.itertuples():
        for sent in re.split(r"(?<=[.])\s+|\n", r.text):
            sent = re.sub(r"(PAT|ENC)-\d+", "#", sent.strip())
            if len(sent) > 25 and not sent.isupper():
                rows.append((r.document_id, r.patient_number, r.group, sent))
    sent = pd.DataFrame(rows, columns=["note", "patient", "group", "sentence"])
    sent["n notes"] = sent["sentence"].map(sent.groupby("sentence")["note"].nunique())
    intro = r"^(Clinical summary for|Occupational assessment for|Follow-up for|Specialist review for|Imaging for)"
    return sent[(sent["n notes"] == 1) & ~sent["note"].isin(excluded_notes) & ~sent["sentence"].str.match(intro)]
