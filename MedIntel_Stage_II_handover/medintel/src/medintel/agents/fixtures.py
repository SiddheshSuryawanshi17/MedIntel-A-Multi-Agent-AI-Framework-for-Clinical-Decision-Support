"""Pre-written context drafts from the Stage I concept prototype, each sentence tagged with its sources.
P-1047 keeps one claim the record does not support, so tests can show it being removed (TC-12)."""

from medintel.models import DraftSentence

BP = ["record:systolic_bp", "record:diastolic_bp"]


def _s(text: str, *sources: str) -> DraftSentence:
    return DraftSentence(text=text, sources=list(sources))


PROTOTYPE_DRAFTS: dict[str, list[DraftSentence]] = {
    "P-1041": [
        _s(
            "Blood pressure, cholesterol and fasting glucose are within the reference ranges, "
            "and the patient does not smoke.",
            *BP,
            "record:total_cholesterol",
            "record:fasting_glucose",
            "record:smoker",
        ),
        _s(
            "The ML model estimates a low risk, and the retrieved evidence describes this profile as low risk.",
            "ml",
            "evidence:KB-GEN-001",
        ),
        _s("No rule was triggered and the components agree.", "rules", "conflicts"),
    ],
    "P-1042": [
        _s(
            "Systolic blood pressure was 186 mmHg at triage, which triggers rule R-001.",
            "record:systolic_bp",
            "rule:R-001",
            "events",
        ),
        _s(
            "The ML model estimates a low long-term risk because the patient is young and the other risk "
            "factors are normal.",
            "ml",
            "record:age",
        ),
        _s(
            "The rule result and the retrieved evidence on severely raised blood pressure disagree with the "
            "model, so the case is escalated for clinician review.",
            "rule:R-001",
            "evidence:KB-HTN-001",
            "conflicts",
        ),
    ],
    "P-1043": [
        _s(
            "Blood pressure, total cholesterol and fasting glucose are raised, and the patient smokes.",
            *BP,
            "record:total_cholesterol",
            "record:fasting_glucose",
            "record:smoker",
        ),
        _s(
            "The ML model estimates a high risk, consistent with the retrieved evidence on each of these factors.",
            "ml",
            "evidence",
        ),
        _s("No rule was triggered and the components agree.", "rules", "conflicts"),
    ],
    "P-1044": [
        _s(
            "Blood pressure, cholesterol and fasting glucose are mildly raised but below the usual treatment "
            "thresholds.",
            *BP,
            "record:total_cholesterol",
            "record:fasting_glucose",
        ),
        _s(
            "The ML model leans toward high risk, but no sufficiently relevant evidence was found in the "
            "knowledge base.",
            "ml",
            "evidence",
        ),
        _s(
            "The notes mention palpitations and a recent fainting episode, which this framework does not assess.",
            "notes",
        ),
    ],
    "P-1045": [
        _s(
            "Blood pressure and total cholesterol are raised and the patient smokes.",
            *BP,
            "record:total_cholesterol",
            "record:smoker",
        ),
        _s(
            "Serum creatinine is missing, so rule R-002 flags that kidney-related checks could not be run.",
            "record:creatinine",
            "rule:R-002",
        ),
        _s("The ML model estimates a high risk, consistent with the retrieved evidence.", "ml", "evidence"),
    ],
    "P-1046": [
        _s(
            "The patient is 16 years old, so rule R-003 withholds the risk recommendation because the model "
            "is intended for adults.",
            "record:age",
            "rule:R-003",
        ),
        _s(
            "The recorded values raise no immediate concern, but assessment should follow pediatric reference ranges.",
            "record",
            "evidence:KB-PED-002",
        ),
    ],
    "P-1047": [
        _s(
            "Blood pressure and total cholesterol are raised, and serum creatinine is 1.6 mg/dL.",
            *BP,
            "record:total_cholesterol",
            "record:creatinine",
        ),
        _s("She has a known history of type 2 diabetes.", "record:history"),
        _s(
            "The ML model estimates a high risk, consistent with the retrieved evidence on blood pressure, "
            "cholesterol and kidney function.",
            "ml",
            "evidence",
        ),
    ],
    "P-1048": [
        _s(
            "All recorded values are within the reference ranges and the patient does not smoke.",
            "record",
            "record:smoker",
        ),
        _s("The ML model and the retrieved evidence both indicate low risk.", "ml", "evidence"),
    ],
}
