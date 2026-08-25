import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ASSESSMENT_FILE = "children_disease_risk_scores_catboost.csv"
UNION_FILE = "union_disease_risk_summary_catboost.csv"
UPAZILA_FILE = "upazila_disease_risk_summary_catboost.csv"
DISTRICT_FILE = "district_disease_risk_summary_catboost.csv"
FEATURE_FILE = "catboost_disease_feature_importance.csv"

VISUAL_DIR = "risk_visuals"

DISEASES = [
    "Pneumonia",
    "Diarrhoea",
    "Malnutrition",
    "Anaemia",
    "Measles"
]

os.makedirs(VISUAL_DIR, exist_ok=True)


# ============================================================
# HELPER: SAFE LOCATION CLEANING
# ============================================================

def clean_location_data(
    df,
    name_col,
    score_col,
    top_n=15
):

    if df is None:
        return pd.DataFrame()

    if name_col not in df.columns:
        return pd.DataFrame()

    if score_col not in df.columns:
        return pd.DataFrame()

    temp = df[
        [
            name_col,
            score_col
        ]
    ].copy()

    # Make score numeric
    temp[score_col] = pd.to_numeric(
        temp[score_col],
        errors="coerce"
    )

    # Remove invalid scores
    temp = temp.dropna(
        subset=[score_col]
    )

    # Fix location names
    temp[name_col] = (
        temp[name_col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # Fix literal nan / None strings
    temp[name_col] = temp[name_col].replace(
        {
            "nan": "Unknown",
            "NaN": "Unknown",
            "None": "Unknown",
            "": "Unknown"
        }
    )

    # Sort
    temp = (
        temp
        .sort_values(
            score_col,
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    return temp


# ============================================================
# HELPER: SAFE SAVE
# ============================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VISUAL_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# LOAD ASSESSMENT DATA
# ============================================================

if not os.path.exists(
    ASSESSMENT_FILE
):

    raise FileNotFoundError(
        f"Could not find {ASSESSMENT_FILE}"
    )


risk_df = pd.read_csv(
    ASSESSMENT_FILE
)


print("=" * 70)
print("VISUALIZATION")
print("=" * 70)

print(
    "Assessment rows:",
    len(risk_df)
)


# ============================================================
# LOAD OPTIONAL FILES
# ============================================================

union_df = None
upazila_df = None
district_df = None
feature_df = None


if os.path.exists(
    UNION_FILE
):

    union_df = pd.read_csv(
        UNION_FILE
    )

    print(
        "Union file loaded:",
        len(union_df)
    )

else:

    print(
        "Union file not found."
    )


if os.path.exists(
    UPAZILA_FILE
):

    upazila_df = pd.read_csv(
        UPAZILA_FILE
    )

    print(
        "Upazila file loaded:",
        len(upazila_df)
    )

else:

    print(
        "Upazila file not found."
    )


if os.path.exists(
    DISTRICT_FILE
):

    district_df = pd.read_csv(
        DISTRICT_FILE
    )

    print(
        "District file loaded:",
        len(district_df)
    )

else:

    print(
        "District file not found."
    )


if os.path.exists(
    FEATURE_FILE
):

    feature_df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "Feature importance file loaded:",
        len(feature_df)
    )

else:

    print(
        "Feature importance file not found."
    )


# ============================================================
# CLEAN LOCATION COLUMNS GLOBALLY
# ============================================================

if (
    union_df is not None
    and
    "union_name" in union_df.columns
):

    union_df["union_name"] = (
        union_df["union_name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


if (
    upazila_df is not None
    and
    "upazila_name" in upazila_df.columns
):

    upazila_df["upazila_name"] = (
        upazila_df["upazila_name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


if (
    district_df is not None
    and
    "district_name" in district_df.columns
):

    district_df["district_name"] = (
        district_df["district_name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# 1. AVERAGE RISK BY DISEASE
# ============================================================

average_risks = {}


for disease in DISEASES:

    col = (
        f"{disease.lower()}_risk_score"
    )

    if col in risk_df.columns:

        values = pd.to_numeric(
            risk_df[col],
            errors="coerce"
        )

        average_risks[
            disease
        ] = values.mean()


if average_risks:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        list(
            average_risks.keys()
        ),
        list(
            average_risks.values()
        )
    )

    plt.title(
        "Average Disease Risk Score"
    )

    plt.xlabel(
        "Disease"
    )

    plt.ylabel(
        "Average Risk Score"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=30
    )

    save_plot(
        "01_average_disease_risk.png"
    )


# ============================================================
# 2. RISK DISTRIBUTION FOR EACH DISEASE
# ============================================================

for disease in DISEASES:

    col = (
        f"{disease.lower()}_risk_score"
    )

    if col not in risk_df.columns:
        continue

    values = pd.to_numeric(
        risk_df[col],
        errors="coerce"
    ).dropna()

    if values.empty:
        continue

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        values,
        bins=20
    )

    plt.title(
        f"{disease} Risk Score Distribution"
    )

    plt.xlabel(
        "Risk Score"
    )

    plt.ylabel(
        "Number of Assessments"
    )

    plt.xlim(
        0,
        100
    )

    save_plot(
        f"02_{disease.lower()}_distribution.png"
    )


# ============================================================
# 3. HIGHEST-RISK DISEASE COUNTS
# ============================================================

if (
    "highest_risk_disease"
    in risk_df.columns
):

    disease_values = (
        risk_df[
            "highest_risk_disease"
        ]
        .fillna("Unknown")
        .astype(str)
    )

    counts = (
        disease_values
        .value_counts()
    )

    if not counts.empty:

        plt.figure(
            figsize=(10, 6)
        )

        plt.bar(
            counts.index.astype(str),
            counts.values
        )

        plt.title(
            "Highest-Risk Disease by Assessment"
        )

        plt.xlabel(
            "Disease"
        )

        plt.ylabel(
            "Number of Assessments"
        )

        plt.xticks(
            rotation=30
        )

        save_plot(
            "03_highest_risk_disease_counts.png"
        )


# ============================================================
# 4. RISK LEVEL DISTRIBUTION
# ============================================================

if (
    "risk_level"
    in risk_df.columns
):

    order = [
        "Very Low",
        "Low",
        "Moderate",
        "High",
        "Very High"
    ]

    risk_levels = (
        risk_df[
            "risk_level"
        ]
        .fillna("Unknown")
        .astype(str)
    )

    counts = (
        risk_levels
        .value_counts()
        .reindex(
            order,
            fill_value=0
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        counts.index,
        counts.values
    )

    plt.title(
        "Overall Risk Level Distribution"
    )

    plt.xlabel(
        "Risk Level"
    )

    plt.ylabel(
        "Number of Assessments"
    )

    save_plot(
        "04_risk_level_distribution.png"
    )


# ============================================================
# 5. TOP 15 HIGH-RISK UNIONS
# ============================================================

if union_df is not None:

    temp = clean_location_data(
        union_df,
        "union_name",
        "highest_risk_score",
        15
    )

    if not temp.empty:

        plt.figure(
            figsize=(12, 8)
        )

        plt.barh(
            temp[
                "union_name"
            ].astype(str),
            temp[
                "highest_risk_score"
            ]
        )

        plt.title(
            "Top 15 Highest-Risk Unions"
        )

        plt.xlabel(
            "Highest Risk Score"
        )

        plt.ylabel(
            "Union"
        )

        plt.xlim(
            0,
            100
        )

        plt.gca().invert_yaxis()

        save_plot(
            "05_top_risk_unions.png"
        )


# ============================================================
# 6. TOP UNIONS FOR EACH DISEASE
# ============================================================

if union_df is not None:

    for disease in DISEASES:

        score_col = (
            f"{disease.lower()}_risk_score"
        )

        temp = clean_location_data(
            union_df,
            "union_name",
            score_col,
            15
        )

        if temp.empty:
            continue

        plt.figure(
            figsize=(12, 8)
        )

        plt.barh(
            temp[
                "union_name"
            ].astype(str),
            temp[
                score_col
            ]
        )

        plt.title(
            f"Top 15 Unions by {disease} Risk"
        )

        plt.xlabel(
            f"{disease} Risk Score"
        )

        plt.ylabel(
            "Union"
        )

        plt.xlim(
            0,
            100
        )

        plt.gca().invert_yaxis()

        save_plot(
            f"06_union_{disease.lower()}_risk.png"
        )


# ============================================================
# 7. TOP 15 HIGH-RISK UPAZILAS
# ============================================================

if upazila_df is not None:

    temp = clean_location_data(
        upazila_df,
        "upazila_name",
        "highest_risk_score",
        15
    )

    if not temp.empty:

        plt.figure(
            figsize=(12, 8)
        )

        plt.barh(
            temp[
                "upazila_name"
            ].astype(str),
            temp[
                "highest_risk_score"
            ]
        )

        plt.title(
            "Top 15 Highest-Risk Upazilas"
        )

        plt.xlabel(
            "Highest Risk Score"
        )

        plt.ylabel(
            "Upazila"
        )

        plt.xlim(
            0,
            100
        )

        plt.gca().invert_yaxis()

        save_plot(
            "07_top_risk_upazilas.png"
        )


# ============================================================
# 8. UPAZILA RISK BY EACH DISEASE
# ============================================================

if upazila_df is not None:

    for disease in DISEASES:

        score_col = (
            f"{disease.lower()}_risk_score"
        )

        temp = clean_location_data(
            upazila_df,
            "upazila_name",
            score_col,
            15
        )

        if temp.empty:
            continue

        plt.figure(
            figsize=(12, 8)
        )

        plt.barh(
            temp[
                "upazila_name"
            ].astype(str),
            temp[
                score_col
            ]
        )

        plt.title(
            f"Top 15 Upazilas by {disease} Risk"
        )

        plt.xlabel(
            f"{disease} Risk Score"
        )

        plt.ylabel(
            "Upazila"
        )

        plt.xlim(
            0,
            100
        )

        plt.gca().invert_yaxis()

        save_plot(
            f"08_upazila_{disease.lower()}_risk.png"
        )


# ============================================================
# 9. DISTRICT OVERALL RISK
# ============================================================

if district_df is not None:

    temp = clean_location_data(
        district_df,
        "district_name",
        "highest_risk_score",
        1000
    )

    if not temp.empty:

        plt.figure(
            figsize=(11, 7)
        )

        plt.barh(
            temp[
                "district_name"
            ].astype(str),
            temp[
                "highest_risk_score"
            ]
        )

        plt.title(
            "Disease Risk by District"
        )

        plt.xlabel(
            "Highest Risk Score"
        )

        plt.ylabel(
            "District"
        )

        plt.xlim(
            0,
            100
        )

        plt.gca().invert_yaxis()

        save_plot(
            "09_district_overall_risk.png"
        )


# ============================================================
# 10. DISTRICT RISK BY DISEASE
# ============================================================

if district_df is not None:

    for disease in DISEASES:

        score_col = (
            f"{disease.lower()}_risk_score"
        )

        temp = clean_location_data(
            district_df,
            "district_name",
            score_col,
            1000
        )

        if temp.empty:
            continue

        plt.figure(
            figsize=(11, 7)
        )

        plt.barh(
            temp[
                "district_name"
            ].astype(str),
            temp[
                score_col
            ]
        )

        plt.title(
            f"District {disease} Risk"
        )

        plt.xlabel(
            f"{disease} Risk Score"
        )

        plt.ylabel(
            "District"
        )

        plt.xlim(
            0,
            100
        )

        plt.gca().invert_yaxis()

        save_plot(
            f"10_district_{disease.lower()}_risk.png"
        )


# ============================================================
# 11. CATBOOST FEATURE IMPORTANCE
# ============================================================

if feature_df is not None:

    if (
        "disease" in feature_df.columns
        and
        "feature" in feature_df.columns
        and
        "importance" in feature_df.columns
    ):

        feature_df[
            "disease"
        ] = (
            feature_df[
                "disease"
            ]
            .fillna("Unknown")
            .astype(str)
        )

        feature_df[
            "feature"
        ] = (
            feature_df[
                "feature"
            ]
            .fillna("Unknown")
            .astype(str)
        )

        feature_df[
            "importance"
        ] = pd.to_numeric(
            feature_df[
                "importance"
            ],
            errors="coerce"
        )

        for disease in DISEASES:

            temp = (
                feature_df[
                    feature_df[
                        "disease"
                    ] == disease
                ]
                .dropna(
                    subset=[
                        "importance"
                    ]
                )
                .sort_values(
                    "importance",
                    ascending=False
                )
                .head(15)
            )

            if temp.empty:
                continue

            plt.figure(
                figsize=(10, 8)
            )

            plt.barh(
                temp[
                    "feature"
                ].astype(str),
                temp[
                    "importance"
                ]
            )

            plt.title(
                f"Top CatBoost Features - {disease}"
            )

            plt.xlabel(
                "Feature Importance"
            )

            plt.ylabel(
                "Clinical Feature"
            )

            plt.gca().invert_yaxis()

            save_plot(
                f"11_feature_importance_{disease.lower()}.png"
            )


# ============================================================
# 12. UNION MULTI-DISEASE COMPARISON
# ============================================================

if union_df is not None:

    union_risk_columns = [

        f"{d.lower()}_risk_score"

        for d in DISEASES

        if (
            f"{d.lower()}_risk_score"
            in union_df.columns
        )
    ]

    if (
        union_risk_columns
        and
        "union_name" in union_df.columns
        and
        "highest_risk_score"
        in union_df.columns
    ):

        temp = union_df.copy()

        temp[
            "union_name"
        ] = (
            temp[
                "union_name"
            ]
            .fillna("Unknown")
            .astype(str)
        )

        temp[
            "highest_risk_score"
        ] = pd.to_numeric(
            temp[
                "highest_risk_score"
            ],
            errors="coerce"
        )

        temp = (
            temp
            .dropna(
                subset=[
                    "highest_risk_score"
                ]
            )
            .sort_values(
                "highest_risk_score",
                ascending=False
            )
            .head(10)
            .copy()
        )

        for col in union_risk_columns:

            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce"
            )

        temp = temp.set_index(
            "union_name"
        )

        plot_data = (
            temp[
                union_risk_columns
            ]
            .copy()
        )

        plot_data.columns = [

            col
            .replace(
                "_risk_score",
                ""
            )
            .title()

            for col
            in plot_data.columns
        ]

        if not plot_data.empty:

            plot_data.plot(
                kind="bar",
                figsize=(14, 8)
            )

            plt.title(
                "Disease Risk Comparison Across Top 10 Unions"
            )

            plt.xlabel(
                "Union"
            )

            plt.ylabel(
                "Average Risk Score"
            )

            plt.ylim(
                0,
                100
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.legend(
                title="Disease"
            )

            save_plot(
                "12_union_disease_comparison.png"
            )


# ============================================================
# 13. UPAZILA MULTI-DISEASE COMPARISON
# ============================================================

if upazila_df is not None:

    upazila_risk_columns = [

        f"{d.lower()}_risk_score"

        for d in DISEASES

        if (
            f"{d.lower()}_risk_score"
            in upazila_df.columns
        )
    ]

    if (
        upazila_risk_columns
        and
        "upazila_name" in upazila_df.columns
        and
        "highest_risk_score" in upazila_df.columns
    ):

        temp = upazila_df.copy()

        temp[
            "upazila_name"
        ] = (
            temp[
                "upazila_name"
            ]
            .fillna("Unknown")
            .astype(str)
        )

        temp[
            "highest_risk_score"
        ] = pd.to_numeric(
            temp[
                "highest_risk_score"
            ],
            errors="coerce"
        )

        temp = (
            temp
            .dropna(
                subset=[
                    "highest_risk_score"
                ]
            )
            .sort_values(
                "highest_risk_score",
                ascending=False
            )
            .head(10)
            .copy()
        )

        for col in upazila_risk_columns:

            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce"
            )

        temp = temp.set_index(
            "upazila_name"
        )

        plot_data = (
            temp[
                upazila_risk_columns
            ]
            .copy()
        )

        plot_data.columns = [

            col
            .replace(
                "_risk_score",
                ""
            )
            .title()

            for col
            in plot_data.columns
        ]

        if not plot_data.empty:

            plot_data.plot(
                kind="bar",
                figsize=(14, 8)
            )

            plt.title(
                "Disease Risk Comparison Across Top 10 Upazilas"
            )

            plt.xlabel(
                "Upazila"
            )

            plt.ylabel(
                "Average Risk Score"
            )

            plt.ylim(
                0,
                100
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.legend(
                title="Disease"
            )

            save_plot(
                "13_upazila_disease_comparison.png"
            )


# ============================================================
# 14. DISEASE RISK CORRELATION MATRIX
# ============================================================

risk_columns = [

    f"{d.lower()}_risk_score"

    for d in DISEASES

    if (
        f"{d.lower()}_risk_score"
        in risk_df.columns
    )
]


if len(
    risk_columns
) >= 2:

    correlation_data = (
        risk_df[
            risk_columns
        ]
        .copy()
    )

    for col in risk_columns:

        correlation_data[col] = (
            pd.to_numeric(
                correlation_data[col],
                errors="coerce"
            )
        )

    correlation = (
        correlation_data
        .corr()
    )

    labels = [

        col
        .replace(
            "_risk_score",
            ""
        )
        .title()

        for col in correlation.columns
    ]

    plt.figure(
        figsize=(9, 8)
    )

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(
            len(labels)
        ),
        labels,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(
            len(labels)
        ),
        labels
    )

    for i in range(
        len(labels)
    ):

        for j in range(
            len(labels)
        ):

            value = (
                correlation.iloc[
                    i,
                    j
                ]
            )

            if pd.notna(value):

                text = (
                    f"{value:.2f}"
                )

            else:

                text = "NA"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center"
            )

    plt.title(
        "Disease Risk Score Correlation"
    )

    save_plot(
        "14_disease_risk_correlation.png"
    )


# ============================================================
# 15. ACTUAL DISEASE COUNTS
# ============================================================

actual_counts = {}


for disease in DISEASES:

    col = (
        f"actual_{disease.lower()}"
    )

    if col in risk_df.columns:

        values = pd.to_numeric(
            risk_df[col],
            errors="coerce"
        )

        actual_counts[
            disease
        ] = values.sum()


if actual_counts:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        list(
            actual_counts.keys()
        ),
        list(
            actual_counts.values()
        )
    )

    plt.title(
        "Actual Disease Cases in Dataset"
    )

    plt.xlabel(
        "Disease"
    )

    plt.ylabel(
        "Number of Positive Assessments"
    )

    plt.xticks(
        rotation=30
    )

    save_plot(
        "15_actual_disease_counts.png"
    )


# ============================================================
# 16. ACTUAL VS AVERAGE PREDICTED RISK
# ============================================================

actual_rates = {}
predicted_risks = {}


for disease in DISEASES:

    actual_col = (
        f"actual_{disease.lower()}"
    )

    risk_col = (
        f"{disease.lower()}_risk_score"
    )

    if (
        actual_col in risk_df.columns
        and
        risk_col in risk_df.columns
    ):

        actual = pd.to_numeric(
            risk_df[
                actual_col
            ],
            errors="coerce"
        )

        predicted = pd.to_numeric(
            risk_df[
                risk_col
            ],
            errors="coerce"
        )

        actual_rates[
            disease
        ] = (
            actual.mean()
            * 100
        )

        predicted_risks[
            disease
        ] = (
            predicted.mean()
        )


if actual_rates:

    comparison_df = pd.DataFrame(
        {
            "Actual Positive Rate (%)":
                actual_rates,

            "Average Predicted Risk":
                predicted_risks
        }
    )

    comparison_df.plot(
        kind="bar",
        figsize=(11, 7)
    )

    plt.title(
        "Actual Disease Rate vs Average CatBoost Risk"
    )

    plt.xlabel(
        "Disease"
    )

    plt.ylabel(
        "Score / Percentage"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.legend()

    save_plot(
        "16_actual_vs_predicted_risk.png"
    )


# ============================================================
# 17. HIGHEST-RISK CHILDREN / ASSESSMENTS
# ============================================================

if (
    "highest_risk_score"
    in risk_df.columns
):

    temp = risk_df.copy()

    temp[
        "highest_risk_score"
    ] = pd.to_numeric(
        temp[
            "highest_risk_score"
        ],
        errors="coerce"
    )

    temp = (
        temp
        .dropna(
            subset=[
                "highest_risk_score"
            ]
        )
        .sort_values(
            "highest_risk_score",
            ascending=False
        )
        .head(20)
        .copy()
    )

    if not temp.empty:

        if (
            "child_id"
            in temp.columns
        ):

            temp[
                "plot_label"
            ] = (
                temp[
                    "child_id"
                ]
                .fillna("Unknown")
                .astype(str)
            )

        else:

            temp[
                "plot_label"
            ] = (
                temp.index
                .astype(str)
            )

        plt.figure(
            figsize=(12, 9)
        )

        plt.barh(
            temp[
                "plot_label"
            ],
            temp[
                "highest_risk_score"
            ]
        )

        plt.title(
            "Top 20 Highest-Risk Assessments"
        )

        plt.xlabel(
            "Highest Disease Risk Score"
        )

        plt.ylabel(
            "Child ID"
        )

        plt.xlim(
            0,
            100
        )

        plt.gca().invert_yaxis()

        save_plot(
            "17_top_high_risk_children.png"
        )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(
    f"Graphs saved inside: {VISUAL_DIR}"
)

print(
    "\nYou can now open the risk_visuals folder."
)