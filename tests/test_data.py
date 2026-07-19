from hospital_pricing.data import FEATURES, load_data, normalize_column_name


def test_normalize_column_name_handles_spacing_and_dashes() -> None:
    assert normalize_column_name(" BP -HIGH ") == "BP_HIGH"
    assert normalize_column_name("LENGTH OF STAY – ICU") == "LENGTH_OF_STAY_ICU"


def test_load_data_returns_expected_case_study_shape() -> None:
    X, y = load_data("data/mission_hospital.xlsx")
    assert X.shape == (248, len(FEATURES))
    assert y.notna().all()
    assert (y > 0).all()
