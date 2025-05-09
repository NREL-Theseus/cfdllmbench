import pandas as pd

exec_basic = pd.read_csv("basic_success_report.csv")
exec_adv = pd.read_csv("advanced_success_report.csv")

sim_basic = pd.read_csv("similarity_report_basic.csv")
sim_adv = pd.read_csv("similarity_report_advanced.csv")

nmse_basic = pd.read_csv("basic_nmse_report.csv")
nmse_adv = pd.read_csv("advanced_nmse_report.csv")

# Helper: NMSE threshold scoring
def score_nmse(val):
    if val < 0.1:
        return 1.0
    elif val < 0.3:
        return 0.5
    else:
        return 0.0

def evaluate(split_name, exec_df, sim_df, nmse_df):
    exec_df["key"] = exec_df["Dataset"].astype(str) + "_" + exec_df["Directory"].astype(str)
    sim_df["key"] = sim_df["Dataset"].astype(str) + "_" + sim_df["Directory"].astype(str)
    nmse_df["key"] = nmse_df["Dataset"].astype(str) + "_" + nmse_df["Directory"].astype(str)
    nmse_df["NMSE_Score"] = nmse_df["NMSE"].apply(score_nmse)

    merged = exec_df.merge(sim_df, on="key").merge(nmse_df, on="key")
    merged["Success_Score"] = ((merged["Execution"] == 1) & (merged["NMSE_Score"] == 1)).astype(float)

    return pd.DataFrame({
        "Split": [split_name],
        "Execution Score": [merged["Execution"].mean()],
        "CodeBLEU Score": [merged["CodeBLEU"].mean()],
        "Tree Score": [merged["TreeScore"].mean()],
        "NMSE Score": [merged["NMSE_Score"].mean()],
        "Success Ratio": [merged["Success_Score"].mean()]
    })

# Evaluate for both splits
basic_scores = evaluate("Basic", exec_basic, sim_basic, nmse_basic)
adv_scores = evaluate("Advanced", exec_adv, sim_adv, nmse_adv)
final_scores = pd.concat([basic_scores, adv_scores], ignore_index=True)
final_scores.to_csv("final_benchmark_scores.csv", index=False)
