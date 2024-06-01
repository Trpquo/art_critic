# %%
import pandas as pd

from utils.pipe_tools.workbench import Stats
from utils.secrets import cloud

emotions = Stats(
    data_url=cloud / "data/WikiArt-Emotions/Wikiart-Emotions-All.tsv",
    title="Wikiart-emotions",
)
info = Stats(
    data_url=cloud / "data/WikiArt-Emotions/Wikiart-info.tsv", title="Wikiart-info"
)


# %%
emo = emotions.raw_data.merge(info.raw_data[["ID", "Image URL"]], on="ID", how="inner")
# emo.columns

# %%
emo.drop(
    [
        "Art (image+title): agreeableness",
        "Art (image+title): anger",
        "Art (image+title): anticipation",
        "Art (image+title): arrogance",
        "Art (image+title): disagreeableness",
        "Art (image+title): disgust",
        "Art (image+title): fear",
        "Art (image+title): gratitude",
        "Art (image+title): happiness",
        "Art (image+title): humility",
        "Art (image+title): love",
        "Art (image+title): optimism",
        "Art (image+title): pessimism",
        "Art (image+title): regret",
        "Art (image+title): sadness",
        "Art (image+title): shame",
        "Art (image+title): shyness",
        "Art (image+title): surprise",
        "Art (image+title): trust",
        "Art (image+title): neutral",
        "TitleOnly: agreeableness",
        "TitleOnly: anger",
        "TitleOnly: anticipation",
        "TitleOnly: arrogance",
        "TitleOnly: disagreeableness",
        "TitleOnly: disgust",
        "TitleOnly: fear",
        "TitleOnly: gratitude",
        "TitleOnly: happiness",
        "TitleOnly: humility",
        "TitleOnly: love",
        "TitleOnly: optimism",
        "TitleOnly: pessimism",
        "TitleOnly: regret",
        "TitleOnly: sadness",
        "TitleOnly: shame",
        "TitleOnly: shyness",
        "TitleOnly: surprise",
        "TitleOnly: trust",
        "TitleOnly: neutral",
    ],
    axis=1,
    inplace=True,
)
emo.rename(columns={"Ave. art rating": "Rating", "Image URL": "webUrl"}, inplace=True)
emo.webUrl = emo.webUrl.apply(lambda s: s + "!Large.jpg")

# %%
emo.sample(5)

# %%
emo["artistUrl"] = emo.Artist.apply(lambda s: s.strip().lower().replace(" ", "-"))


def format_year(val):
    if isinstance(val, int) or isinstance(val, str):
        return "-" + val
    else:
        return "-" + str(int(val))


emo["url"] = emo.Title.apply(
    lambda s: s.strip().lower().replace(" ", "-")
) + emo.Year.apply(format_year)


# %%
def create_emotion_string(row):
    emotions = {
        "regret",
        "sadness",
        "shame",
        "shyness",
        "surprise",
        "trust",
        "neutral",
        "agreeableness",
        "anger",
        "anticipation",
        "arrogance",
        "disagreeableness",
        "disgust",
        "fear",
        "gratitude",
        "happiness",
        "humility",
        "love",
        "optimism",
        "pessimism",
    }
    string = ""
    for emotion in emotions:
        string += f"{emotion}, " if row[f"ImageOnly: {emotion}"] > threshold else ""
    return string[:-2]


emotion_column = pd.Series(dtype="object")
threshold = 0.3
for idx, row in emo.iterrows():
    emotion_column.loc[idx] = create_emotion_string(row)

emoem = pd.concat([emo, emotion_column], axis=1)

# %%
emoem.rename(columns={0: "emotion"}, inplace=True)
emoem[
    [
        "emotion",
        "ImageOnly: agreeableness",
        "ImageOnly: anger",
        "ImageOnly: anticipation",
        "ImageOnly: arrogance",
        "ImageOnly: disagreeableness",
        "ImageOnly: disgust",
        "ImageOnly: fear",
        "ImageOnly: gratitude",
        "ImageOnly: happiness",
        "ImageOnly: humility",
        "ImageOnly: love",
        "ImageOnly: optimism",
        "ImageOnly: pessimism",
        "ImageOnly: regret",
        "ImageOnly: sadness",
        "ImageOnly: shame",
        "ImageOnly: shyness",
        "ImageOnly: surprise",
        "ImageOnly: trust",
        "ImageOnly: neutral",
    ]
].sample(10)

# %%
emoemo = emoem.drop(
    [
        "ImageOnly: agreeableness",
        "ImageOnly: anger",
        "ImageOnly: anticipation",
        "ImageOnly: arrogance",
        "ImageOnly: disagreeableness",
        "ImageOnly: disgust",
        "ImageOnly: fear",
        "ImageOnly: gratitude",
        "ImageOnly: happiness",
        "ImageOnly: humility",
        "ImageOnly: love",
        "ImageOnly: optimism",
        "ImageOnly: pessimism",
        "ImageOnly: regret",
        "ImageOnly: sadness",
        "ImageOnly: shame",
        "ImageOnly: shyness",
        "ImageOnly: surprise",
        "ImageOnly: trust",
        "ImageOnly: neutral",
    ],
    axis=1,
)
emoemo.to_parquet("../data/Wikiart-Emotions-All-t.parquet")
