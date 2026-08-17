"""
Shared feature-set definitions for modality-specific
and multimodal modeling.

These feature lists preserve the predictor definitions
used during the Week 4 unimodal baseline experiments so
that subsequent multimodal datasets use the same validated
feature groups.
"""

# =============================================================================
# Demographic features
# =============================================================================

DEMOGRAPHIC_FEATURES = [
    "age",
    "age_at_diagnosis",
    "height_cm",
    "weight_kg",
    "gender",
    "handedness",
    "family_history_any",
    "family_history_first_degree",
    "alcohol_effect_on_tremor",
]


# =============================================================================
# Questionnaire features
# =============================================================================

QUESTIONNAIRE_ITEMS = [
    f"Q{i:02d}"
    for i in range(1, 31)
]

DERIVED_QUESTIONNAIRE_FEATURES = [
    "total_symptom_count",
    "gastrointestinal_count",
    "urinal_count",
    "pain_count",
    "miscellaneous_count",
    "apathy_attention_memory_count",
    "distortion_perception_count",
    "depression_anxiety_count",
    "sexual_function_count",
    "cardiovascular_count",
    "sleep_fatigue_count",
]

QUESTIONNAIRE_FEATURES = (
    QUESTIONNAIRE_ITEMS
    + DERIVED_QUESTIONNAIRE_FEATURES
)

# =============================================================================
# Wearable features
# =============================================================================

def get_wearable_features(
    dataframe,
):
    """
    Return wearable predictor columns from the
    participant-level wearable feature dataset.
    """

    return [
        column
        for column in dataframe.columns
        if column != "patient_id"
    ]


# =============================================================================
# Metadata / non-predictor columns
# =============================================================================

METADATA_COLUMNS = [
    "patient_id",
    "condition_group",
    "label",
]