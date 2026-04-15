import os
import tempfile
import textwrap

import altair as alt
import pandas as pd
import streamlit as st

import wellbeing_pipeline_v2 as wp


st.set_page_config(page_title="Wellbeing Index Dashboard", layout="wide")

st.title("Practitioner Wellbeing Index")
st.caption("Welcome to the Practitioner Wellbeing Index, developed by DCC K-Hub, in partnership with CIWA. We hope you find it useful.")

st.markdown(
    """
    <style>
    .summary-card {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
        border: 1px solid #d7e3f2;
        border-radius: 16px;
        padding: 1rem 1rem 0.9rem 1rem;
        min-height: 132px;
        box-shadow: 0 6px 18px rgba(28, 68, 110, 0.06);
    }
    .summary-card__label {
        color: #5c6f82;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .summary-card__value {
        color: #18344e;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    .summary-card__value--score {
        font-size: 2rem;
    }
    .mini-card {
        background: #f7f9fc;
        border: 1px solid #e0e7ef;
        border-radius: 14px;
        padding: 0.85rem;
        min-height: 120px;
    }
    .mini-card__label {
        color: #617487;
        font-size: 0.76rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .mini-card__value {
        color: #19354f;
        font-size: 1.05rem;
        font-weight: 650;
        line-height: 1.3;
        margin-bottom: 0.4rem;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    .mini-card__score {
        color: #5b6d7e;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_score(value):
    return (value - 1) / 4 * 100


def status_label(score):
    if score >= 75:
        return "Strong"
    if score >= 60:
        return "Stable"
    if score >= 45:
        return "Watch"
    return "Needs Attention"


def wrap_text(text, width=22):
    return textwrap.fill(str(text), width=width)


def format_series(series):
    return series.round(1)


def render_summary_card(label, value, score_card=False):
    value_class = "summary-card__value summary-card__value--score" if score_card else "summary-card__value"
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-card__label">{label}</div>
            <div class="{value_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_indicator_card(label, score):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-card__label">At-Risk Indicator</div>
            <div class="mini-card__value">{label}</div>
            <div class="mini-card__score">{score:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_indicator_scores(df, selected_roles, theme=None):
    filtered_df = df[df["Role"].isin(selected_roles)]
    valid_indicators = set(wp.INDICATOR_WEIGHTS[theme]) if theme else None
    indicator_values = {}

    for question, indicator in wp.QUESTION_TO_INDICATOR.items():
        if valid_indicators and indicator not in valid_indicators:
            continue
        if question not in filtered_df.columns:
            continue

        values = filtered_df[question].dropna()
        if values.empty:
            continue

        indicator_values.setdefault(indicator, [])
        indicator_values[indicator].extend(normalize_score(v) for v in values.tolist())

    scores = {
        indicator: round(sum(values) / len(values), 1)
        for indicator, values in indicator_values.items()
        if values
    }
    return pd.Series(scores, dtype="float64").sort_values()


def build_alignment_table(role_scores):
    records = []
    for theme in wp.THEMES:
        theme_scores = role_scores[theme].dropna()
        if theme_scores.empty:
            continue

        records.append(
            {
                "Theme": theme,
                "Current View": round(theme_scores.mean(), 1),
                "Lowest Role": theme_scores.idxmin(),
                "Lowest Score": round(theme_scores.min(), 1),
                "Highest Role": theme_scores.idxmax(),
                "Highest Score": round(theme_scores.max(), 1),
                "Gap": round(theme_scores.max() - theme_scores.min(), 1),
            }
        )

    return pd.DataFrame(records).sort_values(
        ["Gap", "Current View"], ascending=[False, True]
    )


def horizontal_bar_chart(series, value_label, height=320, wrap_width=22):
    chart_df = series.rename(value_label).reset_index()
    label_column = chart_df.columns[0]
    chart_df = chart_df.rename(columns={label_column: "Label"})
    chart_df["Wrapped Label"] = chart_df["Label"].apply(lambda x: wrap_text(x, wrap_width))

    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=6, color="#2A6F97")
        .encode(
            x=alt.X(f"{value_label}:Q", title=value_label, scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Wrapped Label:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Label:N", title="Label"),
                alt.Tooltip(f"{value_label}:Q", title=value_label, format=".1f"),
            ],
        )
        .properties(height=height)
    )


uploaded_file = st.file_uploader("Upload Qualtrics CSV", type=["csv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    df = wp.load_and_clean(tmp_path)
    df, n_flags = wp.clean_responses(df)
    df = wp.apply_reverse_coding(df)

    role_scores = wp.compute_all_scores(df).round(1)
    composite = wp.compute_composite(role_scores).round(1)

    os.remove(tmp_path)

    st.sidebar.header("Filters")
    all_roles = list(role_scores.index)
    selected_roles = st.sidebar.multiselect(
        "Select roles",
        options=all_roles,
        default=all_roles,
    )

    if not selected_roles:
        st.warning("Select at least one role to view the dashboard.")
        st.stop()

    filtered_role_scores = role_scores.loc[selected_roles].round(1)
    filtered_df = df[df["Role"].isin(selected_roles)].copy()
    view_composite = format_series(filtered_role_scores.mean())
    overall_score = round(view_composite.mean(), 1)
    lowest_theme = wrap_text(view_composite.idxmin(), 20)
    role_risk_scores = format_series(filtered_role_scores.mean(axis=1).sort_values())
    role_most_at_risk = role_risk_scores.index[0]
    top_indicator_risks = compute_indicator_scores(df, selected_roles).head(5)
    alignment_df = build_alignment_table(filtered_role_scores).round(1)
    respondents_by_role = (
        filtered_df["Role"].value_counts().reindex(selected_roles, fill_value=0)
    )

    st.header("Current View")

    score_col, status_col, theme_col, role_col = st.columns(4)
    with score_col:
        render_summary_card("Overall Wellbeing Score", f"{overall_score:.1f}", score_card=True)
    with status_col:
        render_summary_card("Status", status_label(overall_score))
    with theme_col:
        render_summary_card("Lowest Theme", lowest_theme.replace("\n", "<br>"))
    with role_col:
        render_summary_card("Role Most At Risk", role_most_at_risk)

    st.caption("Higher scores indicate stronger wellbeing conditions.")

    st.subheader("Top At-Risk Indicators")
    if top_indicator_risks.empty:
        st.info("No indicator data is available for the selected roles.")
    else:
        indicator_cols = st.columns(min(5, len(top_indicator_risks)))
        for col, (indicator, score) in zip(indicator_cols, top_indicator_risks.items()):
            with col:
                render_indicator_card(wrap_text(indicator, 20).replace("\n", "<br>"), score)

    st.caption(
        f"Viewing {len(filtered_df)} respondents across {len(selected_roles)} role(s). "
        "Status reflects the current filtered view."
    )

    st.header("Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Composite Radar")
        fig1 = wp.fig1_global_radar(view_composite, n_flags)
        st.pyplot(fig1)

    with col2:
        st.subheader("By Role Radar")
        fig2 = wp.fig2_role_radar(filtered_role_scores, n_flags)
        st.pyplot(fig2)

    st.divider()

    st.header("Alignment and Risk")
    align_col1, align_col2 = st.columns([1.05, 1.45])

    with align_col1:
        st.subheader("Themes Most Misaligned")
        gap_series = alignment_df.set_index("Theme")["Gap"].round(1)
        st.altair_chart(
            horizontal_bar_chart(gap_series, "Gap", height=300, wrap_width=20),
            use_container_width=True,
        )

    with align_col2:
        st.subheader("Role Alignment Snapshot")
        display_alignment_df = alignment_df.copy()
        display_alignment_df["Theme"] = display_alignment_df["Theme"].apply(lambda x: wrap_text(x, 28))
        st.dataframe(display_alignment_df, hide_index=True, use_container_width=True)

    st.divider()

    st.header("Drilldowns")
    theme = st.selectbox("Select Theme", wp.THEMES)
    theme_values = filtered_role_scores[theme].round(1)

    detail_col1, detail_col2 = st.columns([1, 1.2])

    with detail_col1:
        st.subheader("Theme Scores by Role")
        st.dataframe(theme_values.rename("Score").round(1))
        st.altair_chart(
            horizontal_bar_chart(theme_values, "Score", height=180, wrap_width=18),
            use_container_width=True,
        )

    with detail_col2:
        st.subheader("Indicator Breakdown")
        indicator_df = compute_indicator_scores(df, selected_roles, theme=theme).round(1)
        if indicator_df.empty:
            st.info("No indicator data is available for this theme and selection.")
        else:
            st.dataframe(indicator_df.rename("Score").round(1))
            st.altair_chart(
                horizontal_bar_chart(indicator_df, "Score", height=340, wrap_width=28),
                use_container_width=True,
            )
            st.caption(
                "Lowest indicators in this theme for the current selection: "
                + ", ".join(indicator_df.head(3).index)
            )

    st.divider()

    st.header("Benchmark Comparison")
    benchmark_file = st.file_uploader(
        "Upload Benchmark CSV", type=["csv"], key="benchmark"
    )

    if benchmark_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(benchmark_file.read())
            tmp_path = tmp.name

        df_b = wp.load_and_clean(tmp_path)
        df_b, _ = wp.clean_responses(df_b)
        df_b = wp.apply_reverse_coding(df_b)

        role_scores_b = wp.compute_all_scores(df_b).round(1)
        composite_b = wp.compute_composite(role_scores_b).round(1)

        os.remove(tmp_path)

        compare_df = pd.DataFrame(
            {
                "Current": composite.round(1),
                "Benchmark": composite_b.round(1),
            }
        )

        st.dataframe(compare_df.round(1))
        st.bar_chart(compare_df.round(1))

    st.divider()

    st.header("Data Context")
    context_col1, context_col2 = st.columns(2)

    with context_col1:
        st.subheader("Respondents by Role")
        st.dataframe(respondents_by_role.rename("Respondents"))

    with context_col2:
        st.subheader("Data Notes")
        st.write("Current role coverage:")
        for role, count in respondents_by_role.items():
            st.write(f"- {role}: {count} respondents")
        st.write(
            "Low-N safeguards and measurement refinements are still being improved."
        )

    st.divider()

    st.header("Download Results")
    st.download_button(
        "Download Role Scores",
        role_scores.round(1).to_csv().encode(),
        "scores_by_role_v2.csv",
    )

    st.download_button(
        "Download Composite",
        composite.round(1).to_frame().to_csv().encode(),
        "composite_v2.csv",
    )
