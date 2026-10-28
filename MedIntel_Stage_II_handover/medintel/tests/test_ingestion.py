from medintel.ingestion import DataIngestionService, IngestionResult, case_id_for, parse_csv

HEADER = (
    "patient_id,age,sex,systolic_bp,diastolic_bp,heart_rate,temperature_c,"
    "total_cholesterol,fasting_glucose,creatinine,bmi,smoker,notes"
)
GOOD = "P-1,45,M,130,85,72,36.8,190,95,1.0,26,no,ok"


def load(*rows: str) -> IngestionResult:
    return DataIngestionService().load_text("\n".join([HEADER, *rows]), "test.csv")


def test_missing_patient_id_is_rejected_tc01(morning: IngestionResult) -> None:
    row = next(r for r in morning.rejected if r.row == 10)
    assert row.patient_id == ""
    assert [(p.field, p.message, p.test_case) for p in row.problems] == [("patient_id", "field required", "TC-01")]


def test_fahrenheit_temperature_is_rejected_not_converted_tc02(morning: IngestionResult) -> None:
    row = next(r for r in morning.rejected if r.row == 11)
    problem = row.problems[0]
    assert (problem.field, problem.test_case) == ("temperature_c", "TC-02")
    assert problem.message == "101.3 looks like °F; expected °C (34-42). Not converted."


def test_valid_rows_become_cases_with_events(morning: IngestionResult) -> None:
    assert morning.records_read == 10
    assert [c.case_id for c in morning.cases][:2] == ["C-1041", "C-1042"]
    case = morning.cases[1]
    assert case.source == "morning_batch.csv:3"
    assert case.x_s["systolic_bp"] == 186
    assert case.x_t == "Headache since morning, one high reading at triage."
    assert [e.time for e in case.x_l] == ["07:58", "08:12", "08:20"]


def test_repeated_patient_id_is_set_aside() -> None:
    result = load(GOOD, GOOD)
    assert len(result.cases) == 1
    assert result.rejected[0].problems[0].message == "repeated patient_id"


def test_missing_required_column_rejects_the_whole_file() -> None:
    text = "\n".join([HEADER.replace(",bmi", ""), GOOD.replace(",26,", ",")])
    result = DataIngestionService().load_text(text, "test.csv")
    assert result.missing_columns == ["bmi"]
    assert result.cases == [] and result.rejected == []


def test_type_range_and_enum_checks() -> None:
    result = load(
        "P-2,45.5,M,130,85,72,36.8,190,95,1.0,26,no,",
        "P-3,45,X,130,85,72,36.8,190,95,1.0,26,no,",
        "P-4,45,M,abc,85,72,36.8,190,95,1.0,26,no,",
        "P-5,45,M,300,85,72,36.8,190,95,1.0,26,maybe,",
    )
    messages = {r.patient_id: [p.message for p in r.problems] for r in result.rejected}
    assert messages["P-2"] == ['"45.5" is not a whole number']
    assert messages["P-3"] == ["expected F or M"]
    assert messages["P-4"] == ['"abc" is not a number']
    assert messages["P-5"] == ["300 is outside 60-260 mmHg", "expected yes or no"]


def test_optional_creatinine_may_be_blank() -> None:
    result = load("P-6,45,F,130,85,72,36.8,190,95,,26,yes,")
    assert result.rejected == []
    assert result.cases[0].x_s["creatinine"] is None
    assert result.cases[0].x_s["smoker"] is True


def test_quoted_fields_crlf_and_bom() -> None:
    text = "﻿" + HEADER + "\r\n" + 'P-7,45,F,130,85,72,36.8,190,95,1.0,26,no,"Cough, mild"' + "\r\n\r\n"
    result = DataIngestionService().load_text(text, "test.csv")
    assert result.cases[0].x_t == "Cough, mild"
    assert parse_csv("").missing_columns[0] == "patient_id"


def test_direct_record_input_uses_the_same_checks() -> None:
    record = {
        "patient_id": "P-8",
        "age": 52,
        "sex": "f",
        "systolic_bp": 132.0,
        "diastolic_bp": 84,
        "heart_rate": 124,
        "temperature_c": 37.0,
        "total_cholesterol": 205,
        "fasting_glucose": 99,
        "creatinine": 0.9,
        "bmi": 27.5,
        "smoker": False,
    }
    result = DataIngestionService().load_record(record)
    assert result.rejected == []
    assert result.cases[0].x_s["sex"] == "F"
    assert result.cases[0].source == "direct-input:2"


def test_case_ids_follow_the_patient_digits() -> None:
    assert case_id_for("P-1042") == "C-1042"
    assert case_id_for("MRN 7") == "C-0007"
    assert case_id_for("anon") == "C-0000"
