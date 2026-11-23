"""Fonctions du dossier health, appelées par le notebook 03_rapport_health.

On y met le chargement des fichiers nettoyés et les calculs trop longs pour le notebook.
"""
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

DOSSIER = Path("../data/health_clean")

# mesures répétées gardées pour le regroupement
MESURES_REPETEES_GARDEES = ["PPULSE", "ECART_POULS", "RPULSE", "ECART_AGE", "BALANCE", "CARDOUT", "COGNITIVE",
                            "INFLAM", "LATSWAY", "MOBILITY", "O2SAT", "THORPEAK", "THORVIB"]


def charger(dossier=DOSSIER):
    """Charge les fichiers nettoyés dans un dictionnaire."""
    d = Path(dossier)
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
    t["JOURNAL"] = pd.read_csv(d / "journal_nettoyage.csv")

    # taille des 7 fichiers d'origine (avant d'ajouter des colonnes)
    noms = {"patient_administration": "P", "clinical_encounters": "E", "clinical_observations": "O",
            "clinical_procedures": "PR", "cardiac_recordings": "CR", "cardiac_signal_samples": "SIG",
            "clinical_notes": "N"}
    t["tailles"] = pd.DataFrame({nom: {"lignes": len(t[k]), "colonnes": t[k].shape[1]} for nom, k in noms.items()}).T

    cat = t["CAT"]
    nombre = cat["type"] == "nombre"
    t["REPETEES"] = cat.index[(cat["n_mesures"] > 241) & nombre].tolist()   # mesurées à chaque consultation
    t["UNIQUES"] = cat.index[(cat["n_mesures"] <= 241) & nombre].tolist()   # mesurées une fois par patient
    t["MEDICAL"] = MESURES_REPETEES_GARDEES + t["UNIQUES"]

    E = t["E"]
    E["duree"] = (E["encounter_end"] - E["encounter_start"]).dt.total_seconds() / 60
    E["jour"] = E["encounter_start"].dt.date.astype(str)
    return t


def resume_nettoyage(journal):
    """Journal de nettoyage regroupé par type d'opération."""
    journal = journal.copy()
    journal["operation"] = journal["action"].str.split(" : ").str[0]
    resume = journal.groupby("operation", sort=False).agg(lignes_concernées=("n", "sum"))
    resume["fichiers"] = journal.groupby("operation", sort=False)["table"].unique().str.join(", ")
    return resume


def _pente(d, col):
    d = d[["T", col]].dropna()
    if len(d) < 3:
        return np.nan
    return np.polyfit(d["T"], d[col], 1)[0]


def variables_patient(t):
    """Construit un tableau par consultation (W2) et ~90 variables par patient (feat)."""
    P, E, O, PR, W = t["P"], t["E"], t["O"], t["PR"], t["W"]
    REPETEES, UNIQUES = t["REPETEES"], t["UNIQUES"]
    conso = W.merge(P[["patient_number", "date_of_birth"]], on="patient_number")
    conso["AGE"] = (conso["encounter_start"] - conso["date_of_birth"]).dt.days / 365.25
    W2 = conso[["encounter_number", "patient_number", "encounter_start"] + REPETEES + ["AGE"]].copy()
    W2["ECART_AGE"] = W2["BIOAGE"] - W2["AGE"]
    W2["ECART_POULS"] = W2["APICAL"] - W2["PPULSE"]
    W2["T"] = (W2["encounter_start"] - pd.Timestamp("2014-01-01")).dt.days / 365.25
    W2 = W2.sort_values(["patient_number", "encounter_start"])
    g = W2.groupby("patient_number")

    # mesures prises une seule fois, et les deux échelles none < mild < moderate < severe
    uniques = O[O["observation_code"].isin(UNIQUES)].pivot(index="patient_number", columns="observation_code",
                                                           values="numeric_value")
    echelles = O[O["observation_code"].isin(["MUSC", "NEURO"])].pivot(index="patient_number", columns="observation_code",
                                                                      values="text_value")
    ordre = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    for c in echelles.columns:
        echelles[c] = echelles[c].map(ordre)
    uniques = uniques.join(echelles)

    # médiane, variabilité et pente des mesures répétées
    feat = g[REPETEES + ["AGE", "ECART_AGE", "ECART_POULS"]].median()
    feat = feat.join(g[REPETEES + ["ECART_POULS"]].std().add_suffix("_STD"))
    for c in ["BIOAGE", "INFLAM", "MOBILITY", "COGNITIVE", "LATSWAY", "THORPEAK", "ECART_POULS", "CARDOUT"]:
        feat[c + "_PENTE"] = g.apply(lambda d: _pente(d, c))
    # plus forte baisse de l'écart d'âge entre deux visites
    avec_age = W2.dropna(subset=["ECART_AGE"]).copy()
    avec_age["variation"] = avec_age.groupby("patient_number")["ECART_AGE"].diff()
    feat["CHUTE_MAX_AGE"] = avec_age.groupby("patient_number")["variation"].min()
    feat = feat.join(uniques)

    # comptages d'anomalies
    def compte(masque):
        return O[masque].groupby("patient_number").size()

    feat["N_OCULAR_ATYP"] = compte((O["observation_code"] == "OCULAR") & (O["text_value"] == "atypical"))
    feat["BINOCOORD_ATYP"] = compte((O["observation_code"] == "BINOCOORD") & (O["text_value"] == "atypical"))
    feat["VESTIB_ANORMAL"] = compte((O["observation_code"] == "VESTIB") & (O["text_value"] == "abnormal"))
    feat["N_DISCORDANT"] = compte(O["interpretation_code"] == "discordant")
    feat["N_REVIEW"] = compte(O["interpretation_code"] == "review")
    feat["N_HORS_PLAGE"] = compte(O["hors_plage_ref"])

    # consultations, actes, ECG, notes
    ge = E.groupby("patient_number")
    feat["NB_CONSULT"] = ge.size()
    feat["DUREE_MOY"] = ge["duree"].mean()
    feat["PART_CONFID"] = ge["confidentiality_class"].apply(lambda s: (s != "standard").mean())
    for typ in E["encounter_type"].unique():
        feat["CONSULT_" + typ] = E[E["encounter_type"] == typ].groupby("patient_number").size()
    for r in E["referral_source"].dropna().unique():
        feat["ORIENTE_" + r.replace(" ", "_")] = E[E["referral_source"] == r].groupby("patient_number").size()
    for c in PR["procedure_code"].unique():
        feat["ACTE_" + c] = PR[PR["procedure_code"] == c].groupby("patient_number").size()
    feat["N_ENREG_CARDIO"] = t["CR"].groupby("patient_number").size()
    feat["N_NOTES"] = t["N"].groupby("patient_number").size()
    comptes = [c for c in feat.columns if c.startswith(("N_", "CONSULT_", "ORIENTE_", "ACTE_", "BINOCOORD_", "VESTIB_"))]
    feat[comptes] = feat[comptes].fillna(0)
    feat["NB_MRP42"] = feat["ACTE_MRP-42"]
    return W2, feat


def balayage_correlations(feat):
    """Pearson et Spearman sur toutes les paires de variables (hors paires d'une même mesure)."""
    s, p = feat.corr("spearman"), feat.corr("pearson")
    cols = list(feat.columns)
    lignes = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, c = cols[i], cols[j]
            if a.split("_")[0] == c.split("_")[0]:   # ex. BIOAGE et BIOAGE_STD : paire évidente
                continue
            lignes.append((a, c, p.loc[a, c], s.loc[a, c]))
    b = pd.DataFrame(lignes, columns=["variable 1", "variable 2", "pearson", "spearman"]).dropna()
    b["écart"] = b["pearson"] - b["spearman"]
    return b


def abod(Z):
    """ABOD : pour chaque point, variance des angles vers toutes les paires d'autres points.
    Un point isolé voit les autres sous des angles proches, donc sa variance est faible."""
    sortie = np.empty(len(Z))
    for i in range(len(Z)):
        V = np.delete(Z, i, axis=0) - Z[i]        # vecteurs du point i vers chacun des autres
        n2 = (V ** 2).sum(axis=1)                 # carré de leur longueur
        M = (V @ V.T) / np.outer(n2, n2)          # produit scalaire de chaque paire, pondéré par les distances
        haut = np.triu_indices(len(V), 1)         # chaque paire une seule fois
        sortie[i] = M[haut].var()
    return sortie


def ruptures_individuelles(W2, E, col="ECART_AGE", seuil=6, min_visites=4):
    """Visites anormales par rapport à l'historique du patient (z-score modifié du patient)."""
    lignes = []
    for pid, d in W2.groupby("patient_number"):
        s = d.set_index("encounter_number")[col].dropna()
        if len(s) < min_visites:
            continue
        ecart_abs = max((s - s.median()).abs().median(), 0.5)
        z = 0.6745 * (s - s.median()) / ecart_abs
        for enc, v in z[z.abs() > seuil].items():
            lignes.append((pid, enc, s[enc], s.median(), v))
    r = pd.DataFrame(lignes, columns=["patient", "consultation", "écart âge", "médiane du patient", "z modifié"])
    infos = E.set_index("encounter_number")
    r["date"] = r["consultation"].map(infos["jour"])
    r["type"] = r["consultation"].map(infos["encounter_type"])
    return r


def consultations_simultanees(E):
    """Paires de consultations d'un même médecin qui se chevauchent dans le temps."""
    lignes = []
    for medecin, d in E.sort_values("encounter_start").groupby("attending_license_number"):
        d = d.reset_index(drop=True)
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if d.loc[j, "encounter_start"] >= d.loc[i, "encounter_end"]:
                    break
                lignes.append((medecin, d.loc[i, "jour"], d.loc[i, "facility_license_number"] != d.loc[j, "facility_license_number"]))
    return pd.DataFrame(lignes, columns=["médecin", "jour", "deux cliniques à la fois"])


def similarite_medecins(E):
    """Profil de chaque patient (part de ses consultations par médecin) et similarité entre patients."""
    profil = E.groupby(["patient_number", "attending_license_number"]).size().unstack(fill_value=0)
    profil = profil.div(profil.sum(axis=1), axis=0)
    simil = (profil @ profil.T).to_numpy(copy=True)
    np.fill_diagonal(simil, 0)
    return profil, simil


def graphe_patients(profil, simil, quantile):
    """Graphe patient-patient en gardant les paires plus similaires que le quantile donné."""
    import networkx as nx
    seuil = np.quantile(simil[np.triu_indices(len(simil), 1)], quantile)
    G = nx.Graph()
    G.add_nodes_from(profil.index)
    ii, jj = np.where(np.triu(simil > seuil, 1))
    G.add_weighted_edges_from((profil.index[i], profil.index[j], simil[i, j]) for i, j in zip(ii, jj))
    return G


def journal_evenements(E, PR, CR):
    """Journal d'événements (consultations, actes, ECG) et une trace par patient."""
    ev = pd.concat([
        pd.DataFrame({"patient": E["patient_number"], "t": E["encounter_start"], "etape": E["encounter_type"]}),
        pd.DataFrame({"patient": PR["patient_number"], "t": PR["performed_time"], "etape": "acte " + PR["procedure_code"]}),
        pd.DataFrame({"patient": CR["patient_number"], "t": CR["recording_date"], "etape": "ECG " + CR["recording_context"]}),
    ]).sort_values(["patient", "t"])
    traces = ev.groupby("patient")["etape"].apply(lambda s: ("DEBUT",) + tuple(s) + ("FIN",))
    return ev, traces


def enchainements(traces):
    """Comptage des enchaînements directs A -> B (directly-follows graph)."""
    c = Counter()
    for t in traces:
        c.update(zip(t, t[1:]))
    return c


def dependances(traces):
    """Degré de dépendance de l'Heuristic Miner : (|A>B| - |B>A|) / (|A>B| + |B>A| + 1)."""
    c = enchainements(traces)
    dep = pd.Series({(a, b): (n - c[(b, a)]) / (n + c[(b, a)] + 1) for (a, b), n in c.items()
                     if a != b and a != "DEBUT" and b != "FIN"})
    return dep.sort_values(ascending=False)


def lots_mrp42(PR, occ):
    """Numéros de lot des injections MRP-42 et bloc de lots de chaque dirigeant."""
    lots = PR[PR["procedure_code"] == "MRP-42"].copy()
    lots["série"] = np.where(lots["material_reference"].str.startswith("MTL-R"), "recherche (MTL-R)", "dirigeants (MTL)")
    lots["numéro"] = lots["material_reference"].str[-4:].astype(int)
    blocs = (lots[lots["série"] == "dirigeants (MTL)"].groupby("patient_number")
             .agg(lots=("numéro", lambda s: f"{s.min()} à {s.max()}"), premier_lot=("numéro", "min"),
                  première_injection=("performed_time", lambda s: s.min().date()))
             .sort_values("premier_lot").join(occ))
    return lots, blocs


def effet_doses(PR, W2):
    """Variation de l'écart d'âge biologique entre la visite juste avant et juste après chaque dose."""
    avec_age = W2.dropna(subset=["ECART_AGE"])
    doses = PR[PR["procedure_code"] == "MRP-42"].sort_values("performed_time")
    doses = doses.assign(rang=doses.groupby("patient_number").cumcount() + 1)
    lignes = []
    for r in doses.itertuples():
        v = avec_age[avec_age["patient_number"] == r.patient_number]
        avant = v[v["encounter_start"] < r.performed_time].tail(1)
        apres = v[v["encounter_start"] > r.performed_time].head(1)
        if len(avant) and len(apres):
            lignes.append((r.patient_number, r.authorization_class, r.rang, apres["ECART_AGE"].iloc[0] - avant["ECART_AGE"].iloc[0]))
    return pd.DataFrame(lignes, columns=["patient", "classe", "rang de la dose", "effet (années)"])


def dtw(a, b):
    """Distance Dynamic Time Warping entre deux séries (coût = écart absolu)."""
    D = np.full((len(a) + 1, len(b) + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[-1, -1]


def trajectoires(W2, col="ECART_AGE"):
    """Trajectoire de chaque patient, ramenée à 0 à la première visite."""
    t = {p: d.sort_values("encounter_start")[col].dropna().values for p, d in W2.groupby("patient_number")}
    return {p: s - s[0] for p, s in t.items() if len(s) >= 2}


def matrice_dtw(traj):
    ids = list(traj)
    dist = np.zeros((len(ids), len(ids)))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dist[i, j] = dist[j, i] = dtw(traj[ids[i]], traj[ids[j]])
    return ids, dist


def synchronie_ecg(CR, SIG, hauteur=.9, ecart_min_s=.3, tolerance_s=.05):
    """Détecte les battements sur les 2 canaux de chaque ECG et mesure la part de battements synchrones."""
    lignes = []
    for r in CR.itertuples():
        s = SIG[SIG["recording_id"] == r.recording_id].sort_values("time_ms")
        temps = s["time_ms"].values / 1000
        ecart_min = int(ecart_min_s * r.sampling_rate_hz)   # en nombre d'échantillons
        pics_a, _ = find_peaks(s["channel_a_normalized"].values, height=hauteur, distance=ecart_min)
        pics_b, _ = find_peaks(s["channel_b_normalized"].values, height=hauteur, distance=ecart_min)
        a, b = temps[pics_a], temps[pics_b]
        # un battement du canal A est synchrone s'il y a un battement du canal B à moins de tolerance_s
        synchrones = [np.min(np.abs(b - x)) < tolerance_s for x in a]
        lignes.append({"enregistrement": r.recording_id, "patient": r.patient_number, "contexte": r.recording_context,
                       "bpm canal A": round(60 / np.median(np.diff(a))), "bpm canal B": round(60 / np.median(np.diff(b))),
                       "battements synchrones": round(np.mean(synchrones), 2)})
    return pd.DataFrame(lignes)


def preparer_texte(texte):
    """Minuscules, retrait des identifiants, de la ponctuation, des chiffres et des mots vides."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    t = re.sub(r"(pat|enc)-\d+", " ", texte.lower())
    t = re.sub(r"[^a-z\s-]", " ", t)
    return " ".join(w for w in t.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2)


def phrases_uniques(notes, exclure_notes=()):
    """Phrases présentes dans une seule note, hors notes exclues et hors première ligne des gabarits."""
    lignes = []
    for r in notes.itertuples():
        for ph in re.split(r"(?<=[.])\s+|\n", r.text):
            ph = re.sub(r"(PAT|ENC)-\d+", "#", ph.strip())
            if len(ph) > 25 and not ph.isupper():
                lignes.append((r.document_id, r.patient_number, r.groupe, ph))
    ph = pd.DataFrame(lignes, columns=["note", "patient", "groupe", "phrase"])
    ph["nb notes"] = ph["phrase"].map(ph.groupby("phrase")["note"].nunique())
    intro = r"^(Clinical summary for|Occupational assessment for|Follow-up for|Specialist review for|Imaging for)"
    return ph[(ph["nb notes"] == 1) & ~ph["note"].isin(exclure_notes) & ~ph["phrase"].str.match(intro)]
