from dataclasses import dataclass
from typing import Literal

from medintel.numeric import number_text

FieldKind = Literal["id", "int", "num", "enum", "bool", "text"]
Value = bool | int | float | str | None


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: FieldKind
    required: bool = True
    unit: str = ""
    min: float | None = None
    max: float | None = None
    ref: tuple[float, float] | None = None
    values: tuple[str, ...] = ()
    fahrenheit_check: bool = False


PATIENT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("patient_id", "Patient ID", "id"),
    FieldSpec("age", "Age", "int", unit="years", min=0, max=110),
    FieldSpec("sex", "Sex", "enum", values=("F", "M")),
    FieldSpec("systolic_bp", "Systolic BP", "num", unit="mmHg", min=60, max=260, ref=(90, 129)),
    FieldSpec("diastolic_bp", "Diastolic BP", "num", unit="mmHg", min=30, max=160, ref=(60, 79)),
    FieldSpec("heart_rate", "Heart rate", "num", unit="bpm", min=25, max=220, ref=(60, 100)),
    FieldSpec(
        "temperature_c",
        "Temperature",
        "num",
        unit="°C",
        min=34,
        max=42,
        ref=(36.1, 37.2),
        fahrenheit_check=True,
    ),
    FieldSpec("total_cholesterol", "Total cholesterol", "num", unit="mg/dL", min=80, max=400, ref=(0, 199)),
    FieldSpec("fasting_glucose", "Fasting glucose", "num", unit="mg/dL", min=40, max=500, ref=(70, 99)),
    FieldSpec("creatinine", "Creatinine", "num", required=False, unit="mg/dL", min=0.2, max=12, ref=(0.6, 1.2)),
    FieldSpec("bmi", "BMI", "num", unit="kg/m²", min=12, max=70, ref=(18.5, 24.9)),
    FieldSpec("smoker", "Current smoker", "bool"),
    # Cardiology work-up fields of the UCI Heart Disease data, read by the trained risk model. Optional: when a
    # record lacks them the model fills them in and the report says so.
    FieldSpec(
        "chest_pain_type",
        "Chest pain type",
        "enum",
        required=False,
        values=("typical_angina", "atypical_angina", "non_anginal", "asymptomatic"),
    ),
    FieldSpec(
        "resting_ecg", "Resting ECG", "enum", required=False, values=("normal", "st_t_abnormality", "lv_hypertrophy")
    ),
    FieldSpec("max_heart_rate", "Peak exercise heart rate", "num", required=False, unit="bpm", min=60, max=220),
    FieldSpec("exercise_angina", "Exercise-induced angina", "bool", required=False),
    FieldSpec("st_depression", "Exercise ST depression", "num", required=False, unit="mm", min=-3, max=10),
    FieldSpec(
        "st_slope", "Peak exercise ST slope", "enum", required=False, values=("upsloping", "flat", "downsloping")
    ),
    FieldSpec("major_vessels", "Major vessels on fluoroscopy", "int", required=False, min=0, max=3),
    FieldSpec(
        "thal", "Thallium stress test", "enum", required=False, values=("normal", "fixed_defect", "reversible_defect")
    ),
    FieldSpec("notes", "Clinical notes", "text", required=False),
)

FIELDS: dict[str, FieldSpec] = {f.key: f for f in PATIENT_FIELDS}
STRUCTURED_KEYS: tuple[str, ...] = tuple(f.key for f in PATIENT_FIELDS if f.key not in ("patient_id", "notes"))
REQUIRED_COLUMNS: tuple[str, ...] = tuple(f.key for f in PATIENT_FIELDS if f.required)
CARDIOLOGY_KEYS: tuple[str, ...] = (
    "chest_pain_type",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "st_depression",
    "st_slope",
    "major_vessels",
    "thal",
)
VALUE_LABELS: dict[str, str] = {
    "typical_angina": "typical angina",
    "atypical_angina": "atypical angina",
    "non_anginal": "non-anginal pain",
    "asymptomatic": "no chest pain (asymptomatic)",
    "st_t_abnormality": "ST-T wave abnormality",
    "lv_hypertrophy": "left ventricular hypertrophy",
    "upsloping": "upsloping",
    "flat": "flat",
    "downsloping": "downsloping",
    "fixed_defect": "fixed defect",
    "reversible_defect": "reversible defect",
}


def canonical_choice(spec: FieldSpec, text: str) -> str | None:
    wanted = text.strip().lower().replace("-", "_").replace(" ", "_")
    return next((value for value in spec.values if value.lower() == wanted), None)


def format_value(key: str, value: Value) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return VALUE_LABELS.get(value, value) if key in CARDIOLOGY_KEYS else value
    spec = FIELDS.get(key)
    unit = f" {spec.unit}" if spec and spec.unit else ""
    return f"{number_text(value)}{unit}"


def abnormal(key: str, value: Value) -> Literal["high", "low"] | None:
    spec = FIELDS.get(key)
    if spec is None or spec.ref is None or value is None or isinstance(value, (bool, str)):
        return None
    if value > spec.ref[1]:
        return "high"
    if value < spec.ref[0]:
        return "low"
    return None
