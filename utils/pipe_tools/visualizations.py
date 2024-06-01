import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
import seaborn as sns

default_data = sns.load_dataset("penguins")


class Visualization:

    def __init__(self, block=default_data, target=None):

        import os

        from utils.pipe_tools.workbench import get_data_frame

        if not os.path.exists("./figures"):
            os.mkdir("./figures")
        self.df = get_data_frame(block)
        d, s = None, None
        if hasattr(block, "dictionary"):
            d = block.dictionary
        if hasattr(block, "scaling"):
            s = block.scaling

        # print(df.isna().sum(), df.shape)

        from utils.pipe_tools.dataset import dataset_visualizer
        from utils.pipe_tools.distributions import distribution_visualizer
        from utils.pipe_tools.relations import relation_visualizer

        self.dataset = self.visualizer(data=self.df, func=dataset_visualizer)
        self.distribution = self.visualizer(
            data=self.df, func=distribution_visualizer, dictionary=d, scaling=s
        )
        self.relation = self.visualizer(
            data=self.df, func=relation_visualizer, dictionary=d, scaling=s
        )

        self.target = target
        self.setup_style()

    def __call__(self, **kwargs):
        default(
            data=self.df,
            feature=self.df.columns[0:3],
            sample_target=self.target,
            **kwargs,
        )

    def visualizer(self, data, func, **kwargs):
        def visualize(replace_data=None, **kwargs2):
            if isinstance(replace_data, pd.DataFrame):
                data = replace_data
            return func(data=data, sample_target=self.target, **kwargs, **kwargs2)

        return visualize

    def setup_style(self, style="ggplot", palette="magma"):
        plt.style.use(style)
        sns.set_palette(palette)
        plt.rc("figure", autolayout=True)
        plt.rc(
            "axes",
            labelweight="bold",
            labelsize="large",
            titleweight="bold",
            titlesize=14,
            titlepad=10,
        )


def default(
    data,
    plotter=sns.pairplot,
    plot: str = "frequency",
    kind: str = "scatter",
    feature=None,
    target=None,
    sample_target: str = None,
    dictionary: dict = None,
    scaling: dict = None,
    legend=None,
    sample_size: int = 500,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    url: str = None,
    **kwargs,
):

    if (
        plot in ("frequency", "categorical")
        and len(feature) < 1
        and feature[0] not in data.columns
    ):
        raise Exception(
            "Need to provide at least two correct features for visualizing a relation!"
        )
        return
    if plot in ("rel", "reg", "joint"):
        features = feature + target
        if len(features) < 2 and (
            features[0] not in data.columns or features[1] not in data.columns
        ):
            raise Exception(
                "Need to provide at least two correct features for visualizing a relation!"
            )
            return

    if sample_size > data.shape[0]:
        sample_size = data.shape[0]
    sample = data.sample(sample_size)
    params = {}
    # print(dictionary, scaling)

    if feature:
        f, t = [], []
        params = {}

        if isinstance(feature, str):
            f = [feature or data.columns[0] or None]
        else:
            f = list(feature)

        if isinstance(target, str):
            t = [target or sample_target or None]
        elif hasattr(target, "__len__"):
            t = list(target)
        else:
            t = [None]

        params = {
            "kind": kind,
            "data": sample,
            "x": f[0],
            "y": f[1] if (len(f) > 1 and kind not in ("ecdf",)) else t[0],
            "size": f[2] if len(f) > 2 else None,
            "style": f[3] if len(f) > 3 else None,
            "hue": t[0] if len(t) else None,
            "col": t[1] if len(t) > 1 else None,
            # "legend": list(dictionary[t[0]].values()),
            **kwargs,
        }
        params = {k: v for (k, v) in params.items() if v is not None}
    elif plot == "bootstrap":
        params = {"series": data[target], **kwargs}
    elif plot == "joint":
        params = {
            "data": sample,
            "x": target[0],
            "y": target[1],
        }
    # print({k:v for (k, v) in params.items() if k != "data"})

    # plt.close("all")
    plt.clf()
    # plt.figure(figsize=(6,8))
    fig = plotter(**params)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if hasattr(params, "legend") or legend:
        plt.legend(labels=params["legend"] or legend)
    plt.show()

    if isinstance(url, str):
        fig.savefig(
            f"figures/{url or title}[{ '_'.join(f) }---{ '_'.join(t) }]_{kind}-plot.png",
            dpi=300,
        )

    return fig


def get_feature_list(a, b):
    return [a] if a and isinstance(a, str) else list(a or b)
