"""
Practitioner Wellbeing Index — Full Pipeline
=============================================
Reads a Qualtrics CSV export, applies indicator weights, computes
weighted theme scores per role, aggregates to a composite org score,
and generates three publication-ready figures:
  1. Global composite radar chart
  2. Multi-layer by-role radar chart
  3. Gap / misalignment bar chart

Usage:
    python wellbeing_pipeline.py

Output files (written to same directory as script):
    scores_by_role.csv
    composite_scores.csv
    fig1_global_radar.png
    fig2_role_radar.png
    fig3_gaps.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# ── 0. CONFIG ──────────────────────────────────────────────────────────────────

CSV_FILE = "Trial_April 13, 2026_08.52.csv"   # <-- update if filename differs
ROLE_COLUMN = "QRole"

# Role weights for the composite (must sum to 1.0)
ROLE_WEIGHTS = {
    "Leadership":      0.50,
    "Middle Manager":  0.30,
    "Frontline Staff": 0.20,
}

# Minimum respondents per role to report (otherwise flagged)
MIN_N_REPORT   = 1   # fully reportable
MIN_N_CAUTION  = 1   # reportable with caveat

# ── 1. DATA MAPS ───────────────────────────────────────────────────────────────

ROLE_THEME_MAP = {
    "Leadership": {
        "Emotional & Psychological Wellbeing":  ["Q1_1","Q1_2","Q1_3","Q1_4","Q2_1","Q2_2","Q2_3"],
        "Community & Connection":               ["Q3_1","Q3_2","Q3_3","Q3_4"],
        "Financial & Structural Supports":      ["Q4_1","Q4_2","Q4_3","Q4_4"],
        "Leadership & Organizational Culture":  ["Q5_1","Q5_2","Q5_3","Q5_4"],
        "Professional Growth & Development":    ["Q6_1","Q6_2","Q6_3","Q6_4"],
        "Physical Health & Security":           ["Q7_1","Q7_2","Q7_3","Q8_1","Q8_2","Q8_3"],
        "Workload & Balance":                   ["Q9_1","Q9_2","Q9_3"],
    },
    "Middle Manager": {
        "Emotional & Psychological Wellbeing":  ["Q10_1","Q10_2","Q10_3","Q10_4","Q11_1","Q11_2","Q11_3"],
        "Community & Connection":               ["Q12_1","Q12_2","Q12_3","Q12_4"],
        "Financial & Structural Supports":      ["Q13_1","Q13_2","Q13_3","Q13_4"],
        "Leadership & Organizational Culture":  ["Q14_1","Q14_2","Q14_3","Q14_4"],
        "Professional Growth & Development":    ["Q15_1","Q15_2","Q15_3","Q15_4"],
        "Physical Health & Security":           ["Q16_1","Q16_2","Q16_3","Q17_1","Q17_2","Q17_3"],
        "Workload & Balance":                   ["Q18_1","Q18_2","Q18_3"],
    },
    "Frontline Staff": {
        "Emotional & Psychological Wellbeing":  ["Q19_1","Q19_2","Q19_3","Q19_4","Q20_1","Q20_2","Q20_3"],
        "Community & Connection":               ["Q21_1","Q21_2","Q21_3","Q21_4"],
        "Financial & Structural Supports":      ["Q22_1","Q22_2","Q22_3","Q22_4"],
        "Leadership & Organizational Culture":  ["Q23_1","Q23_2","Q23_3","Q23_4"],
        "Professional Growth & Development":    ["Q24_1","Q24_2","Q24_3","Q24_4"],
        "Physical Health & Security":           ["Q25_1","Q25_2","Q25_3","Q26_1","Q26_2","Q26_3"],
        "Workload & Balance":                   ["Q27_1","Q27_2","Q27_3"],
    },
}

QUESTION_TO_INDICATOR = {
    "Q1_1":"Emotional exhaustion","Q1_2":"Overwhelm; awareness of burnout symptoms",
    "Q1_3":"Vicarious trauma","Q1_4":"Ability to disconnect; Work boundaries",
    "Q2_1":"Access to counseling","Q2_2":"Self-care activities",
    "Q2_3":"Technology-based physical and mental wellbeing tracking program",
    "Q3_1":"Coworker support","Q3_2":"Management connection",
    "Q3_3":"Peer-to-peer discussion","Q3_4":"Culturally safe spaces",
    "Q4_1":"Fair compensation","Q4_2":"Paid overtime",
    "Q4_3":"RRSP / retirement benefits","Q4_4":"Financial support for dependents or childcare",
    "Q5_1":"Safety to speak up","Q5_2":"Supervisor check-ins",
    "Q5_3":"Inclusion of wellbeing in employee performance reviews","Q5_4":"Feeling appreciated",
    "Q6_1":"Orientation and training process for new staff","Q6_2":"Event participation",
    "Q6_3":"Reflection & growth","Q6_4":"Staff mentorship programs",
    "Q7_1":"Physical and psychological safety policies",
    "Q7_2":"Training and incident response policies","Q7_3":"Sick leave / bereavement policies",
    "Q8_1":"Health insurance","Q8_2":"Paramedical / wellness supports","Q8_3":"Eating well & exercise",
    "Q9_1":"Manageable workload; caseload caps","Q9_2":"Work-life balance",
    "Q9_3":"Flexibility in work hours and processes",
    # Middle Manager (same indicators, different Q numbers)
    "Q10_1":"Emotional exhaustion","Q10_2":"Overwhelm; awareness of burnout symptoms",
    "Q10_3":"Vicarious trauma","Q10_4":"Ability to disconnect; Work boundaries",
    "Q11_1":"Access to counseling","Q11_2":"Self-care activities",
    "Q11_3":"Technology-based physical and mental wellbeing tracking program",
    "Q12_1":"Coworker support","Q12_2":"Management connection",
    "Q12_3":"Peer-to-peer discussion","Q12_4":"Culturally safe spaces",
    "Q13_1":"Fair compensation","Q13_2":"Paid overtime",
    "Q13_3":"RRSP / retirement benefits","Q13_4":"Financial support for dependents or childcare",
    "Q14_1":"Safety to speak up","Q14_2":"Supervisor check-ins",
    "Q14_3":"Inclusion of wellbeing in employee performance reviews","Q14_4":"Feeling appreciated",
    "Q15_1":"Orientation and training process for new staff","Q15_2":"Event participation",
    "Q15_3":"Reflection & growth","Q15_4":"Staff mentorship programs",
    "Q16_1":"Physical and psychological safety policies",
    "Q16_2":"Training and incident response policies","Q16_3":"Sick leave / bereavement policies",
    "Q17_1":"Health insurance","Q17_2":"Paramedical / wellness supports","Q17_3":"Eating well & exercise",
    "Q18_1":"Manageable workload; caseload caps","Q18_2":"Work-life balance",
    "Q18_3":"Flexibility in work hours and processes",
    # Frontline Staff
    "Q19_1":"Emotional exhaustion","Q19_2":"Overwhelm; awareness of burnout symptoms",
    "Q19_3":"Vicarious trauma","Q19_4":"Ability to disconnect; Work boundaries",
    "Q20_1":"Access to counseling","Q20_2":"Self-care activities",
    "Q20_3":"Technology-based physical and mental wellbeing tracking program",
    "Q21_1":"Coworker support","Q21_2":"Management connection",
    "Q21_3":"Peer-to-peer discussion","Q21_4":"Culturally safe spaces",
    "Q22_1":"Fair compensation","Q22_2":"Paid overtime",
    "Q22_3":"RRSP / retirement benefits","Q22_4":"Financial support for dependents or childcare",
    "Q23_1":"Safety to speak up","Q23_2":"Supervisor check-ins",
    "Q23_3":"Inclusion of wellbeing in employee performance reviews","Q23_4":"Feeling appreciated",
    "Q24_1":"Orientation and training process for new staff","Q24_2":"Event participation",
    "Q24_3":"Reflection & growth","Q24_4":"Staff mentorship programs",
    "Q25_1":"Physical and psychological safety policies",
    "Q25_2":"Training and incident response policies","Q25_3":"Sick leave / bereavement policies",
    "Q26_1":"Health insurance","Q26_2":"Paramedical / wellness supports","Q26_3":"Eating well & exercise",
    "Q27_1":"Manageable workload; caseload caps","Q27_2":"Work-life balance",
    "Q27_3":"Flexibility in work hours and processes",
}

INDICATOR_WEIGHTS = {
    "Emotional & Psychological Wellbeing": {
        "Emotional exhaustion": 0.20,
        "Overwhelm; awareness of burnout symptoms": 0.18,
        "Vicarious trauma": 0.18,
        "Ability to disconnect; Work boundaries": 0.14,
        "Access to counseling": 0.12,
        "Self-care activities": 0.10,
        "Technology-based physical and mental wellbeing tracking program": 0.08,
    },
    "Community & Connection": {
        "Coworker support": 0.30,
        "Management connection": 0.25,
        "Peer-to-peer discussion": 0.25,
        "Culturally safe spaces": 0.20,
    },
    "Leadership & Organizational Culture": {
        "Safety to speak up": 0.30,
        "Supervisor check-ins": 0.25,
        "Inclusion of wellbeing in employee performance reviews": 0.25,
        "Feeling appreciated": 0.20,
    },
    "Workload & Balance": {
        "Manageable workload; caseload caps": 0.40,
        "Work-life balance": 0.35,
        "Flexibility in work hours and processes": 0.25,
    },
    "Professional Growth & Development": {
        "Orientation and training process for new staff": 0.25,
        "Event participation": 0.25,
        "Reflection & growth": 0.25,
        "Staff mentorship programs": 0.25,
    },
    "Physical Health & Security": {
        "Physical and psychological safety policies": 0.25,
        "Training and incident response policies": 0.20,
        "Sick leave / bereavement policies": 0.20,
        "Health insurance": 0.15,
        "Paramedical / wellness supports": 0.10,
        "Eating well & exercise": 0.10,
    },
    "Financial & Structural Supports": {
        "Fair compensation": 0.35,
        "Paid overtime": 0.25,
        "RRSP / retirement benefits": 0.20,
        "Financial support for dependents or childcare": 0.20,
    },
}

# Items to reverse-code before scoring (negatively worded items)
# Applies BEFORE normalization: reversed_score = 6 - raw_score (for 1-5 scale)
REVERSE_CODE = {
    "Frontline Staff": ["Q19_1", "Q19_4"],
    # Add more as needed: "Leadership": ["Q1_x"], etc.
}

THEMES = list(INDICATOR_WEIGHTS.keys())

# Short display labels for radar axes
SHORT_LABELS = {
    "Emotional & Psychological Wellbeing": "Emotional\nWellbeing",
    "Community & Connection":              "Community\n& Connection",
    "Financial & Structural Supports":     "Financial\nSupports",
    "Leadership & Organizational Culture": "Culture &\nLeadership",
    "Professional Growth & Development":   "Growth &\nDevelopment",
    "Physical Health & Security":          "Physical\nHealth",
    "Workload & Balance":                  "Workload\n& Balance",
}


# ── 2. DATA LOADING & CLEANING ─────────────────────────────────────────────────

def load_and_clean(filepath):
    """Load Qualtrics CSV, strip metadata rows, keep Q-columns + role."""
    print(f"\n[1/5] Loading data from: {filepath}")
    if not os.path.exists(filepath):
        sys.exit(f"  ERROR: File not found — {filepath}\n"
                 f"  Update CSV_FILE at the top of this script.")

    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Qualtrics adds 2 header rows after the column names row
    df = df.iloc[2:].copy()
    df = df.reset_index(drop=True)

    # Keep only Q-columns plus the role column
    q_cols = [c for c in df.columns if c.startswith("Q")]
    if ROLE_COLUMN not in df.columns:
        sys.exit(f"  ERROR: Role column '{ROLE_COLUMN}' not found in CSV.\n"
                 f"  Available columns: {list(df.columns[:20])}")

    keep = list(set(q_cols + [ROLE_COLUMN]))
    df = df[keep].copy()

    print(f"  After stripping metadata rows: {len(df)} respondents, "
          f"{len(q_cols)} Q-columns")
    return df


def parse_likert(x):
    """Convert '4 - Agree' style strings to int. Returns None if unparseable."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if "-" in s:
        try:
            return int(s.split("-")[0].strip())
        except ValueError:
            return None
    try:
        v = int(float(s))
        return v if 1 <= v <= 5 else None
    except (ValueError, TypeError):
        return None


def clean_responses(df):
    """Parse Likert strings; validate role; report N per role."""
    print("\n[2/5] Parsing responses and validating roles")

    # Parse all Q-columns
    q_cols = [c for c in df.columns if c.startswith("Q") and c != ROLE_COLUMN]
    for col in q_cols:
        df[col] = df[col].apply(parse_likert)

    # Clean role
    df["Role"] = df[ROLE_COLUMN].astype(str).str.strip()

    known_roles = set(ROLE_THEME_MAP.keys())
    unknown = df[~df["Role"].isin(known_roles)]
    if len(unknown) > 0:
        print(f"  WARNING: {len(unknown)} respondents have unrecognised roles "
              f"and will be excluded: {unknown['Role'].unique().tolist()}")
    df = df[df["Role"].isin(known_roles)].copy()

    # Report N per role
    n_by_role = df["Role"].value_counts()
    print("  Respondents per role:")
    flags = {}
    for role in ROLE_WEIGHTS:
        n = n_by_role.get(role, 0)
        if n >= MIN_N_REPORT:
            status = "OK"
        elif n >= MIN_N_CAUTION:
            status = "CAUTION (low N)"
        else:
            status = "WARNING (very low N — interpret with care)"
        flags[role] = status
        print(f"    {role:<20} n={n:>3}  [{status}]")

    return df, flags


# ── 3. SCORING ─────────────────────────────────────────────────────────────────

def apply_reverse_coding(df):
    """
    Reverse-code specified items IN PLACE on the DataFrame
    (fixes the original bug where row copies were modified instead).
    reverse = 6 - raw  (valid for 1–5 scale)
    """
    print("\n[3/5] Applying reverse coding")
    for role, cols in REVERSE_CODE.items():
        mask = df["Role"] == role
        for col in cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col].apply(
                    lambda x: 6 - x if pd.notna(x) else x
                )
                print(f"  Reverse-coded {col} for {role}")
    return df


def compute_theme_score(row, role, theme):
    """
    For one respondent row, compute the weighted theme score (0–100).
    Steps:
      1. Get raw score for each indicator question
      2. Normalize to 0–100: (raw - 1) / 4 * 100
      3. Multiply by indicator weight
      4. Sum and divide by total weight of responded items
    Returns None if no valid responses exist for the theme.
    """
    questions = ROLE_THEME_MAP[role].get(theme, [])
    weights   = INDICATOR_WEIGHTS[theme]

    weighted_sum  = 0.0
    total_weight  = 0.0

    for q in questions:
        indicator = QUESTION_TO_INDICATOR.get(q)
        if indicator is None:
            continue
        w = weights.get(indicator)
        if w is None:
            continue
        val = row.get(q)
        if val is None or pd.isna(val):
            continue

        norm = (float(val) - 1.0) / 4.0 * 100.0
        weighted_sum += norm * w
        total_weight += w

    if total_weight == 0:
        return None
    # Re-weight so partial response doesn't deflate the score
    return weighted_sum / total_weight


def compute_all_scores(df):
    """Compute per-respondent theme scores, then average by role."""
    records = []
    for _, row in df.iterrows():
        role = row["Role"]
        rec  = {"Role": role}
        for theme in THEMES:
            rec[theme] = compute_theme_score(row, role, theme)
        records.append(rec)

    respondent_df = pd.DataFrame(records)

    # Average across respondents within each role
    role_scores = (
        respondent_df
        .groupby("Role")[THEMES]
        .mean()
        .round(1)
    )
    return role_scores


def compute_composite(role_scores):
    """
    Composite = weighted average of role-level theme scores.
    Uses ROLE_WEIGHTS (equal role-unit weighting, not headcount weighting).
    """
    composite = {}
    for theme in THEMES:
        s = 0.0
        for role, w in ROLE_WEIGHTS.items():
            if role in role_scores.index:
                s += role_scores.loc[role, theme] * w
        composite[theme] = round(s, 1)
    return pd.Series(composite, name="Composite")


# ── 4. FIGURES ─────────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "Leadership":      "#185FA5",
    "Middle Manager":  "#0F6E56",
    "Frontline Staff": "#993C1D",
}
COMPOSITE_COLOR = "#534AB7"

def _radar_axes(n):
    """Return evenly-spaced angles for n spokes, closing the loop."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    return angles


def _style_radar_ax(ax, angles, labels):
    """Apply shared radar styling."""
    ax.set_facecolor("#FCFDFF")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color="#304255")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=7.5, color="#91A0AE")
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=4)
    ax.spines["polar"].set_color("#D7E3F0")
    ax.spines["polar"].set_linewidth(1.1)
    ax.grid(color="#DDE6F0", linestyle="-", linewidth=0.8)


def fig1_global_radar(composite, n_flags):
    """Single composite radar chart."""
    print("\n  Generating Figure 1 — Global composite radar")

    labels  = [SHORT_LABELS[t] for t in THEMES]
    values  = [composite[t] for t in THEMES]
    angles  = _radar_axes(len(THEMES))
    vals_c  = values + values[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.2),
                           subplot_kw=dict(polar=True),
                           facecolor="#FCFDFF")
    _style_radar_ax(ax, angles, labels)

    ax.plot(angles, vals_c, color=COMPOSITE_COLOR, linewidth=2.8, zorder=3)
    ax.fill(angles, vals_c, color=COMPOSITE_COLOR, alpha=0.16, zorder=2)
    ax.scatter(angles[:-1], values, color=COMPOSITE_COLOR, s=32, zorder=4)

    # Score annotations on each vertex
    for angle, val in zip(angles[:-1], values):
        ax.annotate(
            f"{val:.1f}",
            xy=(angle, val),
            xytext=(angle, min(val + 8, 102)),
            ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=COMPOSITE_COLOR,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.85),
        )

    ax.set_title("Organizational Wellbeing Index\nComposite Score",
                 pad=24, fontsize=13.5, fontweight="bold", color="#1D2F40")

    # N-flag footnote
    caution_roles = [r for r, s in n_flags.items() if "CAUTION" in s or "WARNING" in s]
    if caution_roles:
        fig.text(0.5, 0.01,
                 f"Note: low respondent N for {', '.join(caution_roles)} — interpret with care.",
                 ha="center", fontsize=7.5, color="#888")

    plt.tight_layout()
    return fig
    print("  Saved: fig1_global_radar.png")


def fig2_role_radar(role_scores, n_flags):
    """Multi-layer radar with one polygon per role."""
    print("\n  Generating Figure 2 — By-role radar")

    labels = [SHORT_LABELS[t] for t in THEMES]
    angles = _radar_axes(len(THEMES))

    fig, ax = plt.subplots(figsize=(7.7, 7.7),
                           subplot_kw=dict(polar=True),
                           facecolor="#FCFDFF")
    _style_radar_ax(ax, angles, labels)

    for role in ROLE_WEIGHTS:
        if role not in role_scores.index:
            continue
        values = [role_scores.loc[role, t] for t in THEMES]
        vals_c = values + values[:1]
        color  = ROLE_COLORS[role]
        ax.plot(angles, vals_c, color=color, linewidth=2.4, label=role, zorder=3)
        ax.fill(angles, vals_c, color=color, alpha=0.09, zorder=2)
        ax.scatter(angles[:-1], values, color=color, s=20, zorder=4)

    ax.set_title("Practitioner Wellbeing by Role",
                 pad=24, fontsize=13.5, fontweight="bold", color="#1D2F40")

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.10),
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor="#D7E3F0",
    )

    caution_roles = [r for r, s in n_flags.items() if "CAUTION" in s or "WARNING" in s]
    if caution_roles:
        fig.text(0.5, 0.01,
                 f"Note: low respondent N for {', '.join(caution_roles)} — interpret with care.",
                 ha="center", fontsize=7.5, color="#888")

    plt.tight_layout()
    return fig
    print("  Saved: fig2_role_radar.png")


def fig3_gaps(role_scores):
    """
    Horizontal grouped bar chart showing theme scores per role,
    plus a gap indicator (max − min across roles).
    """
    print("\n  Generating Figure 3 — Gaps & misalignment")

    roles_present = [r for r in ROLE_WEIGHTS if r in role_scores.index]
    n_themes = len(THEMES)
    bar_h    = 0.22
    offsets  = np.linspace(-(len(roles_present)-1)/2 * bar_h,
                            (len(roles_present)-1)/2 * bar_h,
                            len(roles_present))
    y_pos    = np.arange(n_themes)

    fig, (ax_main, ax_gap) = plt.subplots(
        1, 2,
        figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [3, 1]},
        facecolor="white"
    )

    # — Main panel: grouped bars —
    for i, role in enumerate(roles_present):
        vals = [role_scores.loc[role, t] for t in THEMES]
        ax_main.barh(
            y_pos + offsets[i], vals,
            height=bar_h,
            color=ROLE_COLORS[role],
            alpha=0.85,
            label=role,
        )

    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(
        [SHORT_LABELS[t].replace("\n", " ") for t in THEMES],
        fontsize=9
    )
    ax_main.set_xlim(0, 105)
    ax_main.set_xlabel("Score (0 – 100)", fontsize=9)
    ax_main.axvline(50, color="#ccc", linewidth=0.8, linestyle="--")
    ax_main.set_title("Theme Scores by Role", fontsize=11, fontweight="bold",
                       color="#222", pad=10)
    ax_main.spines[["top","right"]].set_visible(False)
    ax_main.tick_params(axis="x", labelsize=8)

    ax_main.legend(fontsize=8.5, frameon=False, loc="lower right")

    # — Gap panel: gap magnitude bars —
    gaps = []
    gap_colors = []
    for theme in THEMES:
        vals = [role_scores.loc[r, theme] for r in roles_present]
        g = max(vals) - min(vals)
        gaps.append(g)
        if g < 10:
            gap_colors.append("#1D9E75")   # teal — low gap
        elif g < 20:
            gap_colors.append("#BA7517")   # amber — moderate
        else:
            gap_colors.append("#D85A30")   # coral — high gap

    ax_gap.barh(y_pos, gaps, height=0.45, color=gap_colors, alpha=0.85)
    ax_gap.set_yticks(y_pos)
    ax_gap.set_yticklabels([])
    ax_gap.set_xlim(0, max(gaps) * 1.35 if max(gaps) > 0 else 30)
    ax_gap.set_xlabel("Gap (max − min)", fontsize=9)
    ax_gap.set_title("Role Misalignment", fontsize=11, fontweight="bold",
                      color="#222", pad=10)
    ax_gap.spines[["top","right"]].set_visible(False)
    ax_gap.tick_params(axis="x", labelsize=8)

    for i, g in enumerate(gaps):
        ax_gap.text(g + 0.5, y_pos[i], f"{g:.1f}",
                    va="center", fontsize=8, color="#555")

    # Legend for gap colors
    legend_handles = [
        mpatches.Patch(color="#1D9E75", label="Low gap  (<10)"),
        mpatches.Patch(color="#BA7517", label="Moderate (10–20)"),
        mpatches.Patch(color="#D85A30", label="High gap (>20)"),
    ]
    ax_gap.legend(handles=legend_handles, fontsize=7.5, frameon=False,
                  loc="lower right", bbox_to_anchor=(1.0, -0.18))

    plt.tight_layout(w_pad=2)
    return fig
    print("  Saved: fig3_gaps.png")


# ── 5. MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Practitioner Wellbeing Index — Pipeline")
    print("=" * 60)

    # Load
    df = load_and_clean(CSV_FILE)

    # Clean
    df, n_flags = clean_responses(df)

    # Reverse code (fixes DataFrame mutation bug)
    df = apply_reverse_coding(df)

    # Score
    print("\n[4/5] Computing weighted theme scores")
    role_scores = compute_all_scores(df)
    composite   = compute_composite(role_scores)

    print("\n  Role-level theme scores (0–100):")
    print(role_scores.to_string())
    print(f"\n  Composite scores:\n{composite.to_string()}")

    # Save CSVs
    role_scores.to_csv("scores_by_role.csv")
    composite.to_frame().to_csv("composite_scores.csv")
    print("\n  Saved: scores_by_role.csv, composite_scores.csv")

    # Figures
    print("\n[5/5] Generating figures")
    fig1_global_radar(composite, n_flags)
    fig2_role_radar(role_scores, n_flags)
    fig3_gaps(role_scores)

    print("\n" + "=" * 60)
    print("  Done. Check your working directory for output files.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
