import pandas as pd
from scipy.stats import pearsonr

def missing_values(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    Replace all occurrences of a given string with NaN,
    then drop rows containing NaN. Returns a cleaned dataframe.
    """
    df = df.replace(value, pd.NA)
    df = df.dropna()
    return df


def seven_number_summary(numbers: pd.Series) -> pd.Series:
    """
    Given a pandas Series, returns a series of the seven-number
    summary: the mean, standard deviation, minimum, first quartile,
    median, third quartile, and maximum.
    """
    summary = {
        "mean": numbers.mean(),
        "std": numbers.std(),
        "min": numbers.min(),
        "Q1": numbers.quantile(0.25),
        "median": numbers.median(),
        "Q3": numbers.quantile(0.75),
        "max": numbers.max()}
    return pd.Series(summary)
    

def clean_smoke(smoking: pd.DataFrame) -> pd.DataFrame:
    '''
    Given the smoke CSV, removes all missing values, keeps
    variables of interest in the desired format, and removes
    the Non-Hispanic part of each comparing group cell to clean
    the data to be in a more desired format and removes duplicates.
    '''
    smoking = missing_values(smoking, "Not Applicable")
    smoking_cols = ["Year", "State", "Comparing (Focus group)",
                     "Cigarette Use Prevalence % (Focus group)"]
    smoking_summary = smoking[smoking_cols].copy()
    smoking_summary["Cigarette Use Prevalence % (Focus group)"] = pd.to_numeric(
        smoking_summary["Cigarette Use Prevalence % (Focus group)"])
    smoking_summary["Comparing (Focus group)"] = (
        smoking_summary["Comparing (Focus group)"].str.replace("Non-Hispanic ", "", regex=False))
    smoking_summary = smoking_summary.dropna()
    smoking_summary = smoking_summary.drop_duplicates()
    return smoking_summary


def clean_mental(mental: pd.DataFrame) -> pd.DataFrame:
    '''
    Given the mental health CSV, removes all missing values, keeps
    variables of interest in the desired format, and removes
    the "/" part of each AIAN cell to clean
    the data to be in a more desired format and removes duplicates.
    '''
    mental = missing_values(mental, "Suppressed")
    mental_cols = ["Year", "Area", "Demographics_Value", "Percent"]
    mental_summary = mental[mental_cols].copy()
    mental_summary["Percent"] = pd.to_numeric(
        mental_summary["Percent"])
    mental_summary["Demographics_Value"] = (
        mental_summary["Demographics_Value"].str.replace("/", "", regex=False))
    mental_summary = mental_summary.dropna()
    mental_summary = mental_summary.drop_duplicates()
    return mental_summary


def clean_disability(disability: pd.DataFrame) -> pd.DataFrame:
    '''
    Given the disability CSV, removes all missing values, keeps
    variables of interest in the desired format, and removes duplicates.
    '''
    disability_cols = ["Year", "LocationDesc", "Data_Value"]
    disability_summary = disability[disability_cols].copy()
    disability_summary["Data_Value"] = pd.to_numeric(
        disability_summary["Data_Value"])
    disability_summary = disability_summary.dropna()
    disability_summary = disability_summary.drop_duplicates()
    return disability_summary

# Question 1
def merge_mental_smoking(mental: pd.DataFrame, smoking: pd.DataFrame) -> pd.DataFrame:
    """
    Merge mental health and smoking datasets on Year and demographic group.
    Preserves the real State column from the smoking dataset.
    """
    mental = mental.rename(columns={"Area": "State", "Demographics_Value": "Comparing (Focus group)"})
    merged = pd.merge(
        mental,
        smoking,
        on=["State", "Year", "Comparing (Focus group)"],
        how="inner"
    )
    return merged.drop_duplicates()


def pearson_by_race(merged):
    """
    Computes Pearson correlation between smoking prevalence and mental health prevalence
    for each racial demographic group.
    Returns a dataframe with r and p-values.
    """

    results = []

    for race in merged["Comparing (Focus group)"].unique():
        subset = merged.loc[merged["Comparing (Focus group)"] == race]

        smoking = subset["Cigarette Use Prevalence % (Focus group)"]
        mental = subset["Percent"]

        r, p = pearsonr(smoking, mental)

        results.append({
            "Race": race,
            "Pearson_r": r,
            "p_value": p
        })

    return pd.DataFrame(results)


def overall_pearson(merged):
    """
    Computes overall Pearson correlation across all races and states.
    """

    smoking = merged["Cigarette Use Prevalence % (Focus group)"]
    mental = merged["Percent"]

    r, p = pearsonr(smoking, mental)
    return r, p

# Question 3
def merge_all_data(merged: pd.DataFrame, disability: pd.DataFrame) -> pd.DataFrame:
    '''
    Given the merged smoking and mental health dataframe and 
    cleaned disability dataframe, returns a merged dataset 
    '''
    disability_avg = (
        disability.rename(columns={"LocationDesc": "State"})
        .groupby("State")["Data_Value"]
        .mean()
        .reset_index()
        .rename(columns={"Data_Value": "Disability_Avg"})
    )

    state_avg = (
        merged.groupby("State")[[
            "Percent",
            "Cigarette Use Prevalence % (Focus group)"]]
        .mean()
        .reset_index()
        .rename(columns={
            "Percent": "Mental_Avg",
            "Cigarette Use Prevalence % (Focus group)": "Smoking_Avg"}))

    merged_all = pd.merge(
        disability_avg,
        state_avg,
        on="State",
        how="inner")
    return merged_all


def main():
    smoking = pd.read_csv("Cigarette_Smoking_Among_Adults.csv")
    smoking_summary = clean_smoke(smoking)
    mental = pd.read_csv("Mental_Health_Indicators.csv")
    mental_summary = clean_mental(mental)
    disability = pd.read_csv("DHDS.csv")
    disability_summary = clean_disability(disability)

    # Question 1
    merged_mental_smoking = merge_mental_smoking(mental_summary, smoking_summary)
    print(merged_mental_smoking["State"].unique())
    print(merged_mental_smoking["Comparing (Focus group)"].unique())
    print(seven_number_summary(merged_mental_smoking["Cigarette Use Prevalence % (Focus group)"]))
    print(seven_number_summary(merged_mental_smoking["Percent"]))
    race_corr = pearson_by_race(merged_mental_smoking)
    print("Pearson correlation by race:")
    print(race_corr)
    overall_r, overall_p = overall_pearson(merged_mental_smoking)
    print("\nOverall Pearson correlation:", overall_r)
    print("Overall p-value:", overall_p)

    # Question 2
    print(smoking_summary["State"].unique())
    print(smoking_summary["Comparing (Focus group)"].unique())
    print(seven_number_summary(smoking_summary["Cigarette Use Prevalence % (Focus group)"]))

    # Question 3
    merged_all = merge_all_data(merged_mental_smoking, disability_summary)
    print(merged_all["State"].unique())
    print(seven_number_summary(merged_all["Mental_Avg"]))
    print(seven_number_summary(merged_all["Smoking_Avg"]))
    print(seven_number_summary(merged_all["Disability_Avg"]))

if __name__ == '__main__':
    main()
