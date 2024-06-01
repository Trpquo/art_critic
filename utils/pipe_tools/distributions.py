def distribution_visualizer( 
    feature=None, target=None, features=None, targets=None,
    plot:str="frequency", kind:str="hist",
    **kwargs):

    from visualizations import default, get_feature_list
    feature = get_feature_list(feature, features)
    target = get_feature_list(target, targets)

    # API: https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot
    
        # seaborn:
        #   displot (kdeplot, histplot i ecdfplot), 
        #   catplot (pointplot, barplot, countplot, stripplot, swarmplot; violinplot, boxplot, boxenplot), 
        #   scatterplot
        # pandas.plotting.bootstrap_plot(series, fig=None, size=50, samples=500, **kwds)

    if kind=="scatter":
        # ovo spada u relplot i ovdje brka stvar, ali je vrsta prikaza distribucije.
        from relations import relation_visualizer
        return relation_visualizer(feature, target, plot, kind, **kwargs)
    
    
    from seaborn import displot # kinds: hist, kde ili ecdf
    from seaborn import catplot # kinds: strip, swarm; box, boxen, violin; point, bar, count
    from pandas.plotting import bootstrap_plot


    figure = {
        "frequency": displot,
        "categorical": catplot,
        "bootstrap": bootstrap_plot
    }
    if plot not in figure.keys(): 
        raise Exception(f"{plot.capitalize()} is not a valid type of distribution plot! Please use: { ' or '.join(figure.keys()) }." )
        return


    if plot=="categorical" and kind=="hist": kind="bar"
    if kind=="bootstrap": plot=kind

    if plot == "bootstrap":
        params =  {
            "target": feature or target,
            "size": 50,
        }
    elif plot == "categorical":
        f = (list(feature), [feature, None])[isinstance(feature, str)] 
        t = (list(target),  [target, None])[isinstance(target, str)]
        params = {
            "feature": [t[0], f[0]],
            "target": f[1:] + t[1:],
        }
    else:
        params = {
            "feature": feature,
            "target": target,
        }

    if plot != "bootstrap":
         params = { k: [x for x in v if x is not None] for (k,v) in params.items() }
    
    # print(params)

    return default(plotter=figure[plot], plot=plot, kind=kind, **kwargs, **params)

