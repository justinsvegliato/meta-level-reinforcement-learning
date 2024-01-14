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


def create_cost_of_computation_fig(tdu_df):
    pallete = sns.color_palette()
    policy_pallete = [pallete[3], pallete[0], pallete[4], pallete[2]]

    # policy_labels = get_policy_labels()

    # hue_order = [
    #     policy for policy in policy_labels.values() if policy in tdu_df['Meta-level Policy'].unique()
    # ] + ['RLTS (No Cost)']
    # hue_order = [hue_order[0], hue_order[1], hue_order[3], hue_order[2]]

    hue_order = ["Random", "$A^*$", "RLTS ($c_{max}=0.002$)", "RLTS ($c_{max}=0.05$)"]
    tdu_df["Meta-level Policy"] = tdu_df["Meta-level Policy"].apply(
        lambda x: "RLTS ($c_{max}=0.05$)" if x == "RLTS" else x
    )

    set_plotting_style(font_scale=2.5)

    plt.figure(figsize=(15 / 2, 5))
    ax = sns.barplot(
        data=tdu_df,
        x="cost_of_computation",
        y="Time-Dependent Utility",
        hue="Meta-level Policy",
        hue_order=hue_order,
        palette=policy_pallete,
        errwidth=1.5,
        capsize=0.05,
    )
    # tdu_df.to_csv('figures/data/bigfish-tdu.csv', index=False)
    ax.legend(fontsize=15, loc="lower left")
    ax.set_xlabel("Cost of Computation")


def reproduce_figure():
    data = pd.read_csv("figures/bigfish-tdu-vs-cost/data.csv")
    create_cost_of_computation_fig(data)
    plt.savefig(
        "figures/bigfish-tdu-vs-cost/bigfish-tdu-vs-cost.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    reproduce_figure()
