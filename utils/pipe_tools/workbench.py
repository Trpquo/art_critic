# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.pipe_tools.sampling import Sample


class Stats:
    def __init__(self, title, data_url="", df=None, print_log=False):
        if df:
            self.raw_data = df
        elif data_url:
            if "/" in data_url:
                # sep = ","
                url_arr = data_url.split("/")
                _, file_extension = url_arr[-1].split(".")
                match file_extension:
                    case "csv":
                        self.raw_data = pd.read_csv(data_url)
                    case "data":
                        self.raw_data = pd.read_csv(data_url)
                    case "tsv":
                        self.raw_data = pd.read_csv(data_url, sep="\t")
                    case "json":
                        self.raw_data = pd.read_json(data_url)
                    case "parquet":
                        self.raw_data = pd.read_parquet(data_url)
                    case _:
                        raise Exception("Unknown file format.")
            else:
                self.raw_data = sns.load_dataset(data_url)
        else:
            raise Exception("What data???")

        self.title = title or self.raw_data.index.name
        self.dictionary = {}
        self.sample = {}
        self.report = self.create_report(self.raw_data)
        self.update_visuals()
        import os

        if not os.path.exists("./data"):
            os.mkdir("./data")

        self.log = print_log
        if self.log:
            print(self.raw_data.dtypes)
            print(self.raw_data.describe())
            return self.report

    def preprocess(
        self,
        target=None,
        selected_columns: list = [],
        columns_to_impute: list = [],
        categorical_columns: list = [],
        print_log=False,
    ):

        self.data = self.raw_data.copy()
        self.columns_to_impute = columns_to_impute
        self.target = target

        if len(selected_columns) > 0:
            self.data = self.data.drop(
                [col for col in self.raw_data.columns if col not in selected_columns],
                axis=1,
            )

        if len(columns_to_impute) == 0:
            self.data.dropna(inplace=True)

        if len(categorical_columns) > 0:
            for cat in categorical_columns:
                # self.data[cat] = self.data[cat].astype("category") # ne radi sa većinom vrsta sklearn modela
                uniques = self.data[cat].unique()
                mapping = {k: v for (k, v) in zip(uniques, range(len(uniques)))}
                self.data[cat] = self.data[cat].map(mapping)
                self.dictionary[cat] = {v: k for (k, v) in mapping.items()}

        # obj_cols = self.data.select_dtypes(include=["object"]).columns
        # dummies = pd.get_dummies(self.data[obj_cols], drop_first=True)
        # self.data.drop(obj_cols, axis=1, inplace=True)
        # self.data = pd.concat([self.data, dummies], axis=1)
        self.data = pd.get_dummies(self.data, drop_first=True)
        self.report = self.create_report(self.data)

        self.update_visuals()

        # ovo ispisuje izvještaj o preliminarnoj analizi podataka ako se traži
        if self.log or print_log:
            return self.report

    def create_report(self, data=None, report=False):
        # if isinstance(data, type(None)):
        #     data = get_data_frame(self)
        # else:
        #     data = get_data_frame(data)
        if report:
            # import pandas_profiling
            raise Exception(
                "No report here! Issue: pandas-profiling is not working with pydantic v2.*. To make it work in 2023, every package should be downgraded to it's 2020 versions."
            )
            return  # data.profile_report()
        else:
            return lambda _: print(
                "No reporting because pandas-profiling got outdated."
            )

    def update_visuals(self):
        from utils.pipe_tools.visualizations import Visualization

        self.visualize = Visualization(block=self)

    def create_sample(self, standardize=True, target=None, val_precent=0.4):

        if not isinstance(self.data, pd.DataFrame):
            raise Exception("Data needs to be preprocessed first!")
            return

        obj_columns = self.data.select_dtypes(include="object").columns
        bool_columns = self.data.select_dtypes(include="bool").columns

        if len(obj_columns) > 0 or len(bool_columns) > 0:
            raise Exception(
                "There are still nonnumerical columns in dataset."
                + "\nString columns are: "
                + ", ".join(obj_columns)
                + "\nBool columns are: "
                + ", ".join(bool_columns)
            )
            return

        if target:
            self.target = target
        elif not self.target:
            self.target = self.data.columns[-1]

        dictionary = self.dictionary if hasattr(self, "dictionary") else None

        sample = Sample(
            data=self.data,
            target=self.target,
            columns_to_impute=self.columns_to_impute,
            standardize=standardize,
            dictionary=dictionary,
        )
        sample.report = self.create_report(data=get_data_frame(sample))

        return sample


def get_data_frame(block):
    if isinstance(block, pd.DataFrame):
        return block
    elif isinstance(block, Stats):
        if hasattr(block, "data"):
            return block.data
        else:
            return block.raw_data
    elif isinstance(block, Sample):
        return pd.concat([block.X["train"], block.y["train"]], axis=1)
    elif (
        isinstance(block, dict)
        or isinstance(block, list)
        or isinstance(block, np.ndarray)
    ):
        return pd.Dataframe(
            block
        )  # ovo se čini ekstremno glupo, ali ne znam što bolje za sad
    else:
        raise Exception("No DataFrame provided!")
    return
