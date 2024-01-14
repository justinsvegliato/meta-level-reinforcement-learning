import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def create_importance_scores_fig(max_importances_df):
    plt.rcParams["text.latex.preamble"] = r"\usepackage{times}"
    sns.set(
        font_scale=2.5,
        rc={
            "font.family": "serif",
            "text.usetex": True,
            # 'font.size': 20,
            "savefig.facecolor": "white",
        },
    )

    pretty_labels = {
        "parent_id": "Parent ID Features",
        "id": "ID Features",
        "action": "Action Features",
        "state": "State Features",
        "exp_root_return": "Expected Reward from Root",
        "exp_value": "Expected Value at Node State",
        "path_return": "Sum of Rewards to Node",
        "reward": "Reward at Node",
        "meta_cost_of_computation": "Cost of Computation",
    }

    # max_importances_df = max_importances_df[max_importances_df['group'].isin(pretty_labels.keys())]
    max_importances_df["Features"] = max_importances_df["group"].apply(
        pretty_labels.get
    )

    order = (
        max_importances_df.groupby("Features")
        .mean()
        .sort_values("score", ascending=False)
        .index
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=max_importances_df, x="score", y="Features", order=order, capsize=0.2
    )
    ax.set_xlabel("Importance")


def reproduce_figure():
    data = pd.read_csv("figures/feature_importances/data.csv")
    create_importance_scores_fig(data)
    plt.savefig(
        "figures/feature_importances/feature_importances.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    reproduce_figure()
