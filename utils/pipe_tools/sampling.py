from utils.pipe_tools.visualizations import Visualization


class Sample:

    def __init__(
        self,
        data,
        target,
        columns_to_impute=[],
        split=0.4,
        standardize=True,
        dictionary=None,
        random_state=42,
    ):

        self.target = target
        self.scaling = None
        if dictionary:
            self.dictionary = dictionary

        from sklearn import model_selection

        X = data.copy()
        y = X.pop(target)

        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X, y, test_size=split, random_state=random_state
        )

        # impute missing values
        if len(columns_to_impute) > 0:
            # from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer()
            temp = imputer.fit_transform(X_train[columns_to_impute])
            X_train[columns_to_impute] = temp
            temp = imputer.transform(X_val[columns_to_impute])
            X_val[columns_to_impute] = temp

            meds = X_train.median()
            X_train = X_train.fillna(meds)
            X_val = X_val.fillna(meds)

            self.imputed_columns = columns_to_impute

        if standardize:  # (important for anything not DecisionTree)
            import pandas as pd
            from sklearn.preprocessing import StandardScaler

            cols = X_train.columns
            train_idx = X_train.index
            val_idx = X_val.index
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train), columns=cols, index=train_idx
            )
            X_val = pd.DataFrame(scaler.transform(X_val), columns=cols, index=val_idx)
            self.scaling = dict(
                zip(
                    scaler.feature_names_in_,
                    [
                        {"mean": scaler.mean_[i], "var": scaler.var_[i]}
                        for i in range(scaler.n_features_in_)
                    ],
                )
            )

        self.X = {"train": X_train, "val": X_val}
        self.y = {"train": y_train, "val": y_val}

        self.update_visuals()

    def update_visuals(self):
        self.visualize = Visualization(block=self, target=self.target)
