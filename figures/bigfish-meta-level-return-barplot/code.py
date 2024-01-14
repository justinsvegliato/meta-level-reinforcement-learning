import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def set_plotting_style(font_scale=1.0):
    plt.rcParams["text.latex.preamble"] = r"\usepackage{times}"
    sns.set(
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "text.usetex": True,
            # 'font.size': 20,
            "savefig.facecolor": "white",
        },
    )


plot_key = "EvalRewrittenAverageReturn"

plot_name = "Mean Meta-level Return"


def create_mean_meta_level_return_fig(means_df):
    policy_labels = {
        "Instant Terminate": "$\hat{Q}^*$",
        "Random (No Terminate)": "Random",
        "AStar": "$A^*$",
        "Learned Meta-Policy": "RLTS",
    }
    means_df["Meta-level Policy"] = means_df["Meta-level Policy"].apply(
        policy_labels.get
    )

    means_df.sort_values(by="Meta-level Policy", inplace=True)

    # sns.lineplot(data=means_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', alpha=0.25)
    # ax = sns.scatterplot(data=means_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', legend=False)

    set_plotting_style(font_scale=2.25)
    plt.figure(figsize=(10, 6))
    pallete = sns.color_palette()
    policy_pallete = [pallete[3], pallete[0], pallete[2]][::-1]

    policy_order = {policy: i for i, policy in enumerate(policy_labels.values())}

    means_df.sort_values(
        by="Meta-level Policy", inplace=True, key=lambda x: -x.map(policy_order)
    )

    ax = sns.barplot(
        data=means_df,
        x="pretrained_percentile",
        y=plot_key,
        hue="Meta-level Policy",
        palette=policy_pallete,
    )

    # ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15),
    #           ncol=3, fancybox=False, shadow=False)
    ax.legend(ncols=3, fontsize=20)
    ax.set_ylim([0, 1.1])

    plt.xlabel("Pretrained Quantile")
    plt.ylabel(plot_name)


def reproduce_figure():
    data = pd.read_csv("figures/bigfish-meta-level-return-barplot/data.csv")
    create_mean_meta_level_return_fig(data)
    plt.savefig(
        "figures/bigfish-meta-level-return-barplot/bigfish-meta-level-return-barplot.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    reproduce_figure()
