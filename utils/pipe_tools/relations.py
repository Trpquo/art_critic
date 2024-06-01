def relation_visualizer(
    feature=None, target=None, features=None, targets=None,
    plot:str="rel", kind:str="scatter",
    **kwargs):

    from visualizations import default, get_feature_list
    feature = get_feature_list(feature, features)
    target = get_feature_list(target, targets)

    if plot == "relation": plot = "rel"
    if plot == "regression": plot = "reg"
    if plot == "reg" and (len(feature) == 0 or len(feature) > 4): plot = "pair"
    if kind in ("joint", "radial"): plot = kind


    # seaborn: relplot (scatterplot i lineplot), 
    #   regplot, lmplot,  jointplot, pairplot, heatmap
    from seaborn import relplot, lmplot, heatmap

    figure = {
        "rel": relplot,
        "reg": lmplot,
        "heatmap": heatmap,
        "joint": joint_plot,
        "pair": pair_grid_visualizer,
        "radial": radial_separability_visualizer,
    }
    if plot not in figure.keys(): 
        raise Exception(f"{plot.capitalize()} is not a valid type of relation plot! Please use: { ' or '.join(figure.keys()) }." )
        return

    if plot in ["joint", "pair"]:
        params = {
            "target": feature + target,
        }
    elif plot == "radial":
        params = {
            "features": features or feature,
            "target": target[0]        }
    else:
        params = {
            "feature": feature,
            "target": target,
        }

    params = { k: [x for x in v if x is not None] for (k,v) in params.items() }

    return default(plotter=figure[plot], plot=plot, kind=kind, **kwargs, **params)




def joint_plot(data, x, y):
    from yellowbrick.features import JointPlotVisualizer
    jpv = JointPlotVisualizer(feature=x, target=y)
    jpv.fit(data[x], data[y])
    jpv.poof()



def pair_grid_visualizer(data, target):
    from seaborn import pairplot
    return pairplot(data, vars=target or data.columns)


def radial_separability_visualizer(data, features, target, labels):
    from yellowbrick.features import RadViz
    y = data.pop(target)
    rv = RadViz(classes=labels, features=features or data.columns)
    rv.fit(data, y)
    _ = rv.transform(data)
    rv.poof()

