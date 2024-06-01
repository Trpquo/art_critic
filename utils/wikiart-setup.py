# %%
from secrets import cloud

from cleaner import add_categories, append_wikiart_columns
from pipe_tools.workbench import Stats

wikiart = Stats(
    data_url=cloud / "wikiart/wikidata/wikiart.data",
    title="Wikiart-artworks",
    print_log=False,
)
artists = Stats(data_url=cloud / "wikiart/wikidata/labels.data", title="Wikiart-labels")
# paintings = Stats(data_url="../"wikiart/wikidata/wikiart.data", title="Wikiart-catalogue")

# %%
famous_artists = artists.raw_data.dropna(subset=["wikipediaUrl"]).contentId
selection = wikiart.raw_data[
    wikiart.raw_data.artistContentId.isin(famous_artists)
].dropna(subset=["style", "genre"])

# %%
style_columns = add_categories(selection, classification="style")
genre_columns = add_categories(selection, classification="genre")

# selection.shape, style_columns.shape, genre_columns.shape

# %%
final = selection.join([style_columns, genre_columns])
final = final[final["not painting"] == 0.0]
final = final[final["Vague or too specific"] == 0.0]
final.drop(["not painting", "Vague or too specific"], axis=1, inplace=True)

# %%
final = append_wikiart_columns(
    df=final, artists_data=famous_artists, data_directory=cloud / "wikiart/wikidata"
)


# %%
final = final[
    [
        "contentId",
        "artistName",
        "title",
        "year",
        "style",
        "genre",
        "Icon",
        "Realism",
        "Romanticism",
        "Impressionism",
        "Expressionism",
        "Surrealism",
        "Naive Art",
        "Fauvism",
        "Brutalism",
        "Dada",
        "Cubism",
        "Abstract Order",
        "Abstract Chaos",
        "Applied Art",
        "Oriental Illustration",
        "Photography",
        "abstract",
        "painting",
        "drawing",
        "composit",
        "applied art",
        "photography",
        "url",
        "artistUrl",
        "artistContentId",
        "localUrl",
        "webUrl",
    ]
]

final.to_parquet("../data/wikiart_paintings.parquet")

# %%
