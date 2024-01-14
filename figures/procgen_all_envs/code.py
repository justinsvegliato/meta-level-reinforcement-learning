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


def get_policy_labels():
    return {
        "Instant Terminate": "$\hat{Q}^*$",
        "Random (No Terminate)": "Random",
        "AStar": "$A^*$",
        "Learned Meta-Policy": "RLTS",
        "RLTS-ablate-struc": "No Structural\nFeatures",
        "RLTS-ablate-state": "No State\nFeatures",
        "RLTS-ablate-rewards": "Meta-Reward\nAblation",
        "RLTS-no-ablation": "No Ablations",
    }


def create_all_envs_fig(episodes_df):
    set_plotting_style(font_scale=2.25)
    plt.figure(figsize=(10, 6))

    pallete = sns.color_palette()
    policy_pallete = [pallete[1], pallete[3], pallete[0], pallete[2]]

    policy_labels = {
        k: v
        for k, v in get_policy_labels().items()
        if v in episodes_df["Meta-level Policy"].unique()
    }

    ax = sns.barplot(
        data=episodes_df,
        x="Environment",
        y="Normalized Return",
        hue="Meta-level Policy",
        hue_order=policy_labels.values(),
        palette=policy_pallete,
        capsize=0.05,
        errwidth=1.5,
    )

    ax.set_ylim([-0.05, 1.1])

    plt.ylabel("Normalized Object-level Return")

    ax.legend(ncol=episodes_df.Environment.nunique(), loc="upper left", fontsize=20)

    # plt.savefig('figures/procgen_all_envs.pdf', bbox_inches='tight')
    # plt.show()

    # ax = sns.barplot(data=episodes_df, x='Environment', y='Number of Steps', hue='Meta-level Policy',
    #                  hue_order=policy_labels.values(), palette=policy_pallete, errwidth=1.5)

    # plt.figure(figsize=(12, 5))
    # ax = sns.boxplot(data=episodes_df,
    #                 x='Environment', y='Normalized Return',
    #                 hue='Meta-level Policy',
    #                 hue_order=policy_labels.values(),
    #                 palette=policy_pallete)

    # ax.set_ylim([-0.05, 1.3])
    # ax.legend(ncol=episodes_df.Environment.nunique(), loc='upper left', fontsize=15)


def reproduce_figure():
    data = pd.read_csv("figures/procgen_all_envs/data.csv")
    create_all_envs_fig(data)
    plt.savefig("figures/procgen_all_envs/procgen_all_envs.pdf", bbox_inches="tight")


if __name__ == "__main__":
    reproduce_figure()
