from utils.pipe_tools.visualizations import plt


def dataset_visualizer(plot="missing-values", url=None, **kwargs):

    figure = {
        "missing-values": missing_values_visualizer,
    }

    plt.close("all")
    # fig, ax = plt.subplots(figsize=(6,6))
    store = figure[plot](url=url, **kwargs)
    if url and plot != "missing-values":
        fig.savefig("figures/" + url + "_" + plot + "-plot.png", dpi=300)
    return store


def missing_values_visualizer(data, exclude=None, url=None, **kwargs):
    import missingno as msno

    ax = msno.matrix(data)
    fig = ax.get_figure()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2 = msno.bar(data)
    fig2 = ax2.get_figure()

    if exclude:
        ax3 = msno.heatmap(data.loc[:, data.columns != exclude], figsize=(10, 6))
    else:
        ax3 = msno.heatmap(data, figsize=(10, 6))
    fig3 = ax3.get_figure()

    if url:
        fig.savefig("figures/" + url + "_missing_data-pattern.png", dpi=300)
        fig2.savefig("figures/" + url + "_missing_bar-plot.png", dpi=300)
        fig3.savefig("figures/" + url + "_missing_heatmap.png", dpi=300)
    return {"msno": fig, "bar": fig2, "heatmap": fig3}
