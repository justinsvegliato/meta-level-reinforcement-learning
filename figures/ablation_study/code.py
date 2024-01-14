import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def create_ablation_fig(abalation_df):
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

    _, ax = plt.subplots(figsize=(5, 5))

    # colours = [
    #     sns.color_palette()[-3],
    #     sns.color_palette()[0],
    #     sns.color_palette()[0],
    #     sns.color_palette()[0],
    # ]
    sns.barplot(
        data=abalation_df,
        y="Meta-level Policy",
        x="Return",
        # color=colours,
        ax=ax,
        orient="h",
        capsize=0.15,
    )
    plt.xlabel("Object-Level Return")
    # plt.xlabel('')
    # ax.set_xticklabels([])
    plt.ylabel("")


def reproduce_figure():
    data = pd.read_csv("figures/ablation_study/data.csv")
    create_ablation_fig(data)
    plt.savefig("figures/ablation_study/ablation_study.pdf", bbox_inches="tight")


if __name__ == "__main__":
    reproduce_figure()
