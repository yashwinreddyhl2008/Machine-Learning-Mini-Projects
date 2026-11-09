
import pandas as pd
import matplotlib.pyplot as plt
import re

from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Function to extract metrics from TXT files
# --------------------------------------------------


def read_txt_results(file_path):

    with open(file_path, "r") as file:
        content = file.read()

    metrics = {}

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    for metric in metric_names:

        pattern = rf"(?m)^\s*{re.escape(metric)}:\s*([0-9]*\.?[0-9]+)"

        match = re.search(pattern, content)

        if match:
            metrics[metric] = float(match.group(1))
        else:
            raise ValueError(
                f"{metric} not found in {file_path.name}"
            )

    return metrics


# --------------------------------------------------
# 3. Load Random Forest results
# --------------------------------------------------

baseline_metrics = read_txt_results(
    RESULTS_DIR / "baseline_results.txt"
)

improved_metrics = read_txt_results(
    RESULTS_DIR / "improved_results.txt"
)


# --------------------------------------------------
# 4. Load ANN results
# --------------------------------------------------

ann_data = pd.read_csv(
    RESULTS_DIR / "ann_results.csv"
)

ann_metrics = {
    "Accuracy": float(ann_data.loc[0, "Accuracy"]),
    "Precision": float(ann_data.loc[0, "Precision"]),
    "Recall": float(ann_data.loc[0, "Recall"]),
    "F1 Score": float(ann_data.loc[0, "F1 Score"])
}


# --------------------------------------------------
# 5. Create comparison DataFrame
# --------------------------------------------------

results = pd.DataFrame({

    "Model": [
        "Random Forest",
        "Improved Random Forest",
        "Artificial Neural Network"
    ],

    "Accuracy": [
        baseline_metrics["Accuracy"],
        improved_metrics["Accuracy"],
        ann_metrics["Accuracy"]
    ],

    "Precision": [
        baseline_metrics["Precision"],
        improved_metrics["Precision"],
        ann_metrics["Precision"]
    ],

    "Recall": [
        baseline_metrics["Recall"],
        improved_metrics["Recall"],
        ann_metrics["Recall"]
    ],

    "F1 Score": [
        baseline_metrics["F1 Score"],
        improved_metrics["F1 Score"],
        ann_metrics["F1 Score"]
    ]

})


# --------------------------------------------------
# 6. Display comparison
# --------------------------------------------------

print("\nMODEL PERFORMANCE COMPARISON")
print("--------------------------------")

print(results.to_string(index=False))


# --------------------------------------------------
# 7. Save comparison as CSV
# --------------------------------------------------

results.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 8. Generate comparison graph
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0.90, 1.00)
)

plt.title(
    "Machine Learning vs Deep Learning Performance"
)

plt.ylabel("Score")
plt.xlabel("Model")

plt.xticks(rotation=15)

plt.legend(loc="lower right")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "model_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 9. Generate comparison report
# --------------------------------------------------

best_accuracy = results["Accuracy"].max()
best_f1 = results["F1 Score"].max()

best_models = results[
    (results["Accuracy"] == best_accuracy) &
    (results["F1 Score"] == best_f1)
]["Model"].tolist()

with open(
    RESULTS_DIR / "model_comparison.txt",
    "w"
) as file:

    file.write("PHISHING WEBSITE DETECTION\n")
    file.write("MODEL PERFORMANCE COMPARISON\n\n")

    file.write(results.to_string(index=False))

    file.write("\n\nCONCLUSION:\n")

    if len(best_models) > 1:
        file.write(
            "The following models achieved the highest "
            "accuracy and F1 score: "
            + ", ".join(best_models)
            + ".\n"
        )
    else:
        file.write(
            f"{best_models[0]} achieved the highest "
            "accuracy and F1 score.\n"
        )

    file.write(
        "\nThe comparison uses the evaluation metrics "
        "saved by the individual model scripts.\n"
    )


# --------------------------------------------------
# 10. Completion message
# --------------------------------------------------

print("\nComparison table saved successfully.")
print("Comparison graph saved successfully.")
print("Comparison report saved successfully.")
