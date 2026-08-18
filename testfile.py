import pandas as pd
import numpy as np

def seven_number_summary(numbers: pd.Series) -> pd.Series:
    """
    Given a panda dataframe, returns a series of the seven-number
    summary: the mean, standard deviaiton, minimum, first quartile, median, third quartile,
    and maximum.
    """
    summary = {
        "mean": numbers.mean(),
        "std": numbers.std(),
        "min": numbers.min(),
        "Q1": numbers.quantile(0.25),
        "median": numbers.median(),
        "Q3": numbers.quantile(0.75),
        "max": numbers.max()
    }
    return pd.Series(summary)

def missing_values(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    Replace all occurrences of a given string with NaN,
    then drop rows containing NaN. Returns a cleaned dataframe.
    """
    df = df.replace(value, pd.NA)
    df = df.dropna()
    return df


def test_missing_values() -> None:
    '''
    Tests the missing values method with fake dataset
    '''
    df = pd.DataFrame({
        "A": ["ok", "bad", "ok"],
        "B": [1, 2, "missing"]
    })

    # Test 1: Replace "bad" with NaN and drop row
    cleaned = missing_values(df.copy(), "bad")
    assert "bad" not in cleaned["A"].values
    assert cleaned.shape[0] == 2   # one row dropped

    # Test 2: Replace "missing" with NaN and drop row
    cleaned2 = missing_values(df.copy(), "missing")
    assert cleaned2.shape[0] == 2
    # Column B still has object dtype (pandas doesn't auto-convert),
    # but the remaining values should be the original numbers 1 and 2
    assert list(cleaned2["B"]) == [1, 2]

    # Test 3: No matching values → dataframe unchanged
    cleaned3 = missing_values(df.copy(), "not_here")
    assert cleaned3.equals(df)  # identical

    print("missing_values() tests passed.")


# ---------------------------
# Tests for seven_number_summary()
# ---------------------------

def test_seven_number_summary() -> None:
    '''
    Tests the seven number summary method with fake dataset
    '''
    # Fake numeric series
    s = pd.Series([1, 2, 3, 4, 5])

    summary = seven_number_summary(s)

    # Test 1: Mean
    assert summary["mean"] == 3

    # Test 2: Standard deviation
    assert np.isclose(summary["std"], s.std())

    # Test 3: Quartiles
    assert summary["Q1"] == 2
    assert summary["median"] == 3
    assert summary["Q3"] == 4

    # Test 4: Min/max
    assert summary["min"] == 1
    assert summary["max"] == 5

    print("seven_number_summary() tests passed.")
    
def main():
    test_missing_values()
    test_seven_number_summary()


if __name__ == '__main__':
    main()
