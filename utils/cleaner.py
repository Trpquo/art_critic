from secrets import cloud


def column_splitter(
    df,
    column: str = None,
    new_columns: list = None,
    splitter: str = "_",
    spacer: str = None,
):
    df[new_columns] = df[column].str.split(splitter, expand=True, n=1)
    if spacer:
        for c in new_columns:
            df[c] = space_out(df[c], spacer)
    return df


def space_out(series, spacer):
    return series.str.replace(spacer, " ").str.strip()


def wikiart_emotional_columns(df):
    wiki_main = list(df.columns)[:9]
    wiki_emo = [a for a in list(df.columns) if "ImageOnly" in a]
    df = df[wiki_main + wiki_emo]
    emotions = df[["ID"] + wiki_emo]
    emotion = undummify(emotions, prefix_sep=": ")
    emotion.rename(columns={"ImageOnly": "emotion"}, inplace=True)
    df["emotion"] = emotion["emotion"]
    df.drop(wiki_emo, axis=1, inplace=True)
    return df


def wikiart_add_urls(df_em, df_info):
    IDs = list(df_em.ID)
    wikiart_info_match = df_info[df_info.ID.isin(IDs)]
    df_em["Image URL"] = wikiart_info_match["Image URL"]
    return df_em


def add_categories(df, classification: str):
    import pandas as pd
    from styles import genre_categories, style_categories

    match classification:
        case "style":
            categories = style_categories
        case "genre":
            categories = genre_categories

    columns = pd.DataFrame()
    for category in categories.keys():
        categories[category].add(category)

    for idx, row in df.iterrows():
        row_categories = row[classification].split(r",+\s*")
        for category in categories.keys():
            columns.loc[idx, category] = len(
                set(filter(lambda s: s in categories[category], row_categories))
            )

    return columns


def get_painting_urls(df, data_directory: str = cloud / "wikiart/wikidata"):
    import numpy as np
    import pandas as pd

    url_columns = pd.DataFrame()

    for idx, painting in df.iterrows():
        artist_opus = pd.read_json(f"{data_directory}/meta/{painting.artistUrl}.json")
        painting_info = artist_opus[artist_opus.contentId == painting.contentId]
        year = painting_info.completitionYear.values[0]
        if pd.isna(year) or not year:
            year = np.nan
        else:
            year = int(year)
        url_columns.loc[idx, "year"] = year

        year = year if not pd.isna(year) else "unknown-year"
        url_columns.loc[idx, "localUrl"] = (
            cloud
            / f"wikiart/wikidata/images/{painting.artistUrl}/{year}/{painting.contentId}.jpg"
        )
        url_columns.loc[idx, "webUrl"] = painting_info.image.values[0]

    return url_columns


def get_painting_titles(df, data_directory: str = cloud / "wikiart/wikidata"):
    import pandas as pd

    name_column = pd.Series(dtype="object", name="title")

    for idx, painting in df.iterrows():
        artist_opus = pd.read_json(f"{data_directory}/meta/{painting.artistUrl}.json")
        painting_info = artist_opus[artist_opus.contentId == painting.contentId]
        name_column.loc[idx] = painting_info.title.values[0]

    return name_column


def append_wikiart_columns(
    df,
    artists_data,
    columns: list,
    data_directory: str = cloud / "wikiart/wikidata",
):
    import numpy as np
    import pandas as pd

    df = df.merge(
        artists_data.rename(columns={"contentId": "artistContentId"})[
            ["artistContentId", "artistName"]
        ],
        on="artistContentId",
        how="left",
    )

    columns = pd.DataFrame(dtype="object")

    for idx, painting in df.iterrows():
        artist_opus = pd.read_json(f"{data_directory}/meta/{painting.artistUrl}.json")
        painting_info = artist_opus[artist_opus.contentId == painting.contentId]
        columns.loc[idx, "title"] = painting_info.title.values[0]

        year = painting_info.completitionYear.values[0]
        if pd.isna(year) or not year:
            year = np.nan
        else:
            year = int(year)
        columns.loc[idx, "year"] = year

        year_string = year if not pd.isna(year) else "unknown-year"
        columns.loc[idx, "localUrl"] = (
            cloud
            / f"wikiart/wikidata/images/{painting.artistUrl}/{year_string}/{painting.contentId}.jpg"
        )
        columns.loc[idx, "webUrl"] = painting_info.image.values[0]

    return df.merge(columns, how="left")


def collect_artworks_from_artists(sample_artists):
    from os.path import exists

    import pandas as pd

    sample_artworks = pd.DataFrame()
    missing_artists = pd.DataFrame()

    for _, artist in sample_artists.iterrows():
        artist_url = cloud / f"wikiart/wikidata/meta/{artist.url}.json"
        if exists(artist_url):
            artworks = pd.read_json(artist_url)
            sample_artworks = pd.concat([sample_artworks, artworks])
        else:
            missing_artists = pd.concat([missing_artists, artist])
            print("Missing: " + artist.artistName)

    return sample_artworks


def undummify(df, prefix_sep="_"):
    import pandas as pd

    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df
