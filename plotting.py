import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import (
    clean_smoke,
    clean_mental,
    clean_disability,
    merge_mental_smoking,
    merge_all_data,
)


def linePlot_Merged_Mental_Smoking(merged: pd.DataFrame) -> None:
    '''
    Given the merged data of the mental and smoking datasets, 
    computes the averages for smoking and mental health
    and then plots a line graph for each year
    '''
    yearly_avg = (
        merged.groupby("Year")[[
            "Cigarette Use Prevalence % (Focus group)",
            "Percent"]]
        .mean()
        .reset_index()
        .rename(columns={
            "Cigarette Use Prevalence % (Focus group)": "Smoking_Avg",
            "Percent": "Mental_Avg"}))
    yearly_avg["Year"] = yearly_avg["Year"].astype(int)

    plt.figure(figsize=(12, 7))
    # Smoking line
    sns.lineplot(
        data=yearly_avg,
        x="Year",
        y="Smoking_Avg",
        marker="o",
        label="Smoking Avg (%)")

    # Mental health line
    sns.lineplot(
        data=yearly_avg,
        x="Year",
        y="Mental_Avg",
        marker="o",
        label="Mental Health Avg (%)")

    plt.title("Average Smoking vs Mental Health Prevalence by Year (Merged Data)", fontsize=18)
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Average Prevalence (%)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Group", fontsize=14, title_fontsize=16)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("lineplot_merged_mental_smoking.png")

def barPlot_Demographics_Mental_Smoking(merged: pd.DataFrame) -> None:
    '''
    Given the merged data of the mental and smoking datasets, 
    computes the yearly averages for smoking and mental health
    and then plots bar plot over each demographic group 
    '''
    demo_stats = (
        merged.groupby("Comparing (Focus group)")[[
            "Percent",
            "Cigarette Use Prevalence % (Focus group)"
        ]]
        .mean()
        .reset_index()
        .rename(columns={
            "Percent": "Mental_Avg",
            "Cigarette Use Prevalence % (Focus group)": "Smoking_Avg"}))

    plot_df = demo_stats.loc[:, ["Comparing (Focus group)", "Mental_Avg", "Smoking_Avg"]]
    x = np.arange(len(plot_df))
    width = 0.4

    plt.figure(figsize=(14, 8))
    plt.bar(
        x - width / 2,
        plot_df.loc[:, "Mental_Avg"],
        width=width,
        color="blue",
        label="Mental Health Avg (%)")

    plt.bar(
        x + width / 2,
        plot_df.loc[:, "Smoking_Avg"],
        width=width,
        color="orange",
        label="Smoking Avg (%)")
    plt.xticks(ticks=x, labels=plot_df.loc[:, "Comparing (Focus group)"], rotation=45, ha="right", fontsize=14)

    plt.xlabel("Demographic Group", fontsize=16)
    plt.ylabel("Average Prevalence (%)", fontsize=16)
    plt.title("Smoking vs Mental Health Prevalence by Demographic (Average Across Years)", fontsize=18)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig("bargraph_demographics_mental_smoking.png")


# Question 2
def linePlot_Smoking(smoking: pd.DataFrame) -> None:
    """
    Given the cleaned smoking dataset, 
    computes the averages across states for smoking
    and then plots line plot over each year, each line
    representing a different demographic 
    """
    avg_by_year_demo = (
        smoking.groupby(["Year", "Comparing (Focus group)"])["Cigarette Use Prevalence % (Focus group)"]
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(12, 7))

    for demo in avg_by_year_demo["Comparing (Focus group)"].unique():
        subset = avg_by_year_demo.loc[
            avg_by_year_demo["Comparing (Focus group)"] == demo,
            ["Year", "Cigarette Use Prevalence % (Focus group)"]
        ]

        sns.lineplot(
            data=subset,
            x="Year",
            y="Cigarette Use Prevalence % (Focus group)",
            marker="o",
            label=demo
        )

    plt.title("Average Cigarette Use Over Time by Demographic", fontsize=18)
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Average Cigarette Use Prevalence (%)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Demographic Group", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig("lineplot_smoking.png")

    
def barPlot_Smoking_2011_2023(smoking: pd.DataFrame) -> None:
    '''
    Given the cleaned smoking dataset, 
    computes the averages across demographics for smoking
    and then plots bar plot over states, comparing between 2011 and 2023
    '''
    # Filter only 2011 and 2023
    df_filtered = smoking.loc[smoking["Year"].isin([2011, 2023])]

    # Compute average smoking prevalence per state for each year
    avg_by_state = (
        df_filtered.groupby(["Year", "State"])["Cigarette Use Prevalence % (Focus group)"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=avg_by_state,
        x="State",
        y="Cigarette Use Prevalence % (Focus group)",
        hue="Year",
        palette={2011: "blue", 2023: "orange"}
    )

    plt.title("Average Smoking Prevalence by State: 2011 vs 2023", fontsize=18)
    plt.xlabel("State", fontsize=16)
    plt.ylabel("Average Cigarette Use Prevalence (%)", fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=14)
    plt.legend(title="Year", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig("barplot_smoking.png")


def barPlot_State_Disability_Mental_Smoking(merged_all: pd.DataFrame) -> None:
    '''
    Given the merged dataframe for all datasets, computes an 
    average across all demographics for the year 2021 and creates a bar plot
    showing each prevelance for each state. 
    '''
    plot_df = merged_all.loc[:, ["State", "Disability_Avg", "Mental_Avg", "Smoking_Avg"]]

    plot_long = plot_df.melt(
        id_vars="State",
        value_vars=["Disability_Avg", "Mental_Avg", "Smoking_Avg"],
        var_name="Metric", value_name="Value")

    plt.figure(figsize=(18, 10))
    sns.barplot(
        data=plot_long,
        x="State",
        y="Value",
        hue="Metric")

    plt.title("Disability vs Mental Health vs Smoking Prevalence by State", fontsize=18)
    plt.xlabel("State", fontsize=16)
    plt.ylabel("Average Prevalence (%)", fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, title_fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("barplot_state__disability_mental_smoking.png")

def boxplot_all_three(merged_all: pd.DataFrame) -> None:
    """
    Given the merged dataframe for all datasets, creates a bar plot,
    consisting of 3 bars showing each prevelance of .
    """

    plt.figure(figsize=(10, 7))

    sns.boxplot(
        data=merged_all[["Disability_Avg", "Mental_Avg", "Smoking_Avg"]],
        palette="Set2"
    )

    plt.title("Distribution of Disability, Mental Health, and Smoking Prevalence Across States", fontsize=18)
    plt.ylabel("Prevalence (%)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")
    plt.savefig("box_state_disability_mental_smoking.png")


def main():
    smoking = pd.read_csv("Cigarette_Smoking_Among_Adults.csv")
    smoking_summary = clean_smoke(smoking)
    mental = pd.read_csv("Mental_Health_Indicators.csv")
    mental_summary = clean_mental(mental)
    disability = pd.read_csv("DHDS.csv")
    disability_summary = clean_disability(disability)

    # Question 1
    merged_mental_smoking = merge_mental_smoking(mental_summary, smoking_summary)
    linePlot_Merged_Mental_Smoking(merged_mental_smoking)
    barPlot_Demographics_Mental_Smoking(merged_mental_smoking)

    # Question 2
    linePlot_Smoking(smoking_summary)
    barPlot_Smoking_2011_2023(smoking_summary)

    # Question 3
    merged_all = merge_all_data(merged_mental_smoking, disability_summary)
    barPlot_State_Disability_Mental_Smoking(merged_all)
    boxplot_all_three(merged_all)

if __name__ == '__main__':
    main()
