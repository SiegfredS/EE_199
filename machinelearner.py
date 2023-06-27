from sklearn.linear_model import (LinearRegression,
                                  Lasso,
                                  Ridge,
                                  BayesianRidge)
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.tree import (DecisionTreeRegressor,
                          plot_tree)
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from scipy.stats import t
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class MLModel():

    def __init__(self,
                 X_train,
                 X_cross,
                 X_test,
                 y_train,
                 y_cross,
                 y_test,
                 upper=0.9,
                 lower=0.1):
        """
        For hyperparameter fine-tuning, look at the jupyter notebook.
        Why random_state = 42? Inside joke of programmers, pero any will do
        """
        self.model_choices = {"linear": LinearRegression(),
                              "lasso": Lasso(alpha=0.1, random_state=42),
                              "ridge": Ridge(alpha=0.1, random_state=42),
                              "bayesian_ridge": BayesianRidge(),
                              "polynomial": Pipeline([("poly",
                                                       PolynomialFeatures(degree=2,
                                                                          include_bias=False)),
                                                      ("linear",
                                                       LinearRegression(fit_intercept=True))]),
                              "decision_tree": DecisionTreeRegressor(max_depth=20,
                                                                     criterion="friedman_mse",
                                                                     random_state=42),
                              "random_forest": RandomForestRegressor(n_estimators=13,
                                                                     random_state=42),
                              "gbr_quantile": GradientBoostingRegressor(loss="quantile",
                                                                        learning_rate=0.1,
                                                                        n_estimators=10),
                              "gbr_mse": HistGradientBoostingRegressor(loss="squared_error",
                                                                       learning_rate=0.1,
                                                                       max_depth=11),
                              "xgboost": XGBRegressor()
                              }
        self.model = None
        self.X_train = X_train
        self.X_cross = X_cross
        self.X_test = X_test
        self.y_train = y_train
        self.y_cross = y_cross
        self.y_test = y_test
        self.lower = lower
        self.med = 0.5
        self.upper = upper
        self.quantiles = [self.lower, self.med, self.upper]
        self.results = {}
        self.elapsed_times = {}
        self.feat_importances = {}
        self.dataset_list = ["Train", "Cross", "Test"]
        self.gbr_quant_models = {}
        self.gbr_quant_pred = {}
        self.bayesian_std = {}
        self.original = {
            "Train": self.y_train,
            "Cross": self.y_cross,
            "Test": self.y_test,
        }
        self.fit_model()
        self.get_errors()
        self.get_kde_plot()
        self.get_feat_importance_heatmap()

    def fit_model(self):
        for model_name, model in self.model_choices.items():
            # Predictions
            if model_name == "gbr_quantile":
                # Quantile GBR is a special case
                cumulative_time = 0
                for alpha in self.quantiles:
                    if hasattr(model, "alpha"):
                        model.alpha = alpha
                        # Fitting time
                        time_start = datetime.now()
                        self.gbr_quant_models[f"q{alpha:.2f}"] = model.fit(self.X_train, self.y_train)
                        elapsed_time = (datetime.now() - time_start).total_seconds()
                        cumulative_time += elapsed_time

                        # Prediction quantile GBR
                        y_pred_train = model.predict(self.X_train)
                        y_pred_cross = model.predict(self.X_cross)
                        y_pred_test = model.predict(self.X_test)

                        # Store
                        self.gbr_quant_pred[alpha] = {"Train": y_pred_train,
                                                      "Cross": y_pred_cross,
                                                      "Test": y_pred_test}
                    else:
                        pass
                self.results[model_name] = self.gbr_quant_pred[0.5]
                self.elapsed_times[model_name] = cumulative_time
                # Get Feature Importance
                self.feat_importances[model_name] = model.feature_importances_
                continue
                # to next model
            else:
                pass

            # Fitting Time of General Model
            time_start = datetime.now()
            model.fit(self.X_train, self.y_train)
            elapsed_time = (datetime.now() - time_start).total_seconds()
            self.elapsed_times[model_name] = elapsed_time

            # Predictions
            if model_name == "bayesian_ridge":
                y_pred_train, y_std_train = model.predict(self.X_train, return_std=True)
                y_pred_test, y_std_test = model.predict(self.X_test, return_std=True)
                y_pred_cross, y_std_cross = model.predict(self.X_cross, return_std=True)
                self.bayesian_std["Train"] = y_std_train
                self.bayesian_std["Test"] = y_std_test
                self.bayesian_std["Cross"] = y_std_cross
            else:
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                y_pred_cross = model.predict(self.X_cross)

            # Get Feat Importances
            try:
                self.feat_importances[model_name] = model.coef_
            except Exception as e:
                print(e)
                try:
                    self.feat_importances[model_name] = model.feature_importances_
                except Exception as e:
                    print(e)
            # Store the results
            self.results[model_name] = {
                "Train": y_pred_train,
                "Cross": y_pred_cross,
                "Test": y_pred_test,
            }

    def get_errors(self):
        error_dict_to_visualize = {}
        for dataset in self.dataset_list:
            dataset_error_dict = {}
            for model_name in self.model_choices.keys():
                result = self.results[model_name][dataset]
                original_data = self.original[dataset]
                mse = mean_squared_error(result,
                                         original_data,
                                         squared=True)
                rmse = mean_squared_error(result,
                                          original_data,
                                          squared=False)
                mae = mean_absolute_error(result,
                                          original_data, )
                mape = mean_absolute_percentage_error(result,
                                                      original_data, )
                r_squared = r2_score(result,
                                     original_data, )
                dataset_error_dict[model_name] = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "r-squared": r_squared
                }
            error_dict_to_visualize[dataset] = dataset_error_dict
            # Now we visualize
        for dataset in self.dataset_list:
            dataset_error_dict = error_dict_to_visualize[dataset]
            df_to_visualize = pd.DataFrame.from_dict(dataset_error_dict).transpose()
            plt.figure(figsize=(6, 4))
            ax = sns.heatmap(df_to_visualize,
                             annot=True,
                             cmap="coolwarm",
                             vmin=-1,
                             vmax=1)
            ax.set_title(f"Error Metrics for {dataset}")
            # Adjust font size of x and y-axis
            plt.gca().tick_params(axis='x', labelsize=6)
            plt.gca().tick_params(axis='y', labelsize=6)
            plt.tight_layout()
            # Save plot
            plt.savefig(f"static/images/heatmap_{dataset}.png")

            # clear plot
            plt.clf()

            # Not clipped
            dataset_error_dict = error_dict_to_visualize[dataset]
            df_to_visualize = pd.DataFrame.from_dict(dataset_error_dict).transpose()
            plt.figure(figsize=(6, 4))
            ax = sns.heatmap(df_to_visualize, annot=True, cmap="coolwarm")
            ax.set_title(f"Error Metrics for {dataset}")
            plt.gca().tick_params(axis='x', labelsize=6)
            plt.gca().tick_params(axis='y', labelsize=6)
            plt.tight_layout()
            plt.savefig(f"static/images/heatmap_{dataset}_unclipped.png")
            plt.clf()

    def get_kde_plot(self):
        # Para lang iba-iba yung color nila
        model_list = ["linear",
                      "lasso",
                      "ridge",
                      "polynomial",
                      "decision_tree",
                      "random_forest",
                      "gbr_mse",
                      "xgboost",
                      "bayesian_ridge"]
        color_list = ["g",
                      "b",
                      "y",
                      "r",
                      "c",
                      "m",
                      "r",
                      "purple",
                      "olive"]
        dataset = "Test"
        plt.figure(figsize=(6,4))
        sns.kdeplot(self.y_test,
                    shade=False,
                    color="k",
                    label="True LMP Test Set",
                    linestyle="solid",)
        for i in range(len(model_list)):
            model_name = model_list[i]
            color = color_list[i]
            sns.kdeplot(self.results[model_name]["Test"],
                        shade=False,
                        color=f"{color}",
                        label=f"{model_name} prediction",
                        alpha=0.6,
                        linestyle="dashed", )

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"static/images/kde_plot")

    def get_feat_importance_heatmap(self):
        feat_importances_df = pd.DataFrame.from_dict(self.feat_importances)
        feat_importances_df.set_index(self.X_train.columns, inplace=True)
        plt.figure(figsize=(8,6))
        annot_kws = {"fontsize": 6}
        ax = sns.heatmap(feat_importances_df,
                         cmap="coolwarm",
                         annot=True,
                         fmt=".2f",
                         annot_kws=annot_kws,
                         vmin=-0.5,
                         vmax=0.5)
        ax.set_title("Feature Importance Heatmap")
        plt.gca().tick_params(axis='x', labelsize=5)
        plt.gca().tick_params(axis='y', labelsize=5)
        plt.tight_layout()
        plt.savefig("static/images/feat_importance_heatmap")
        plt.clf()

        # Not Clipped
        plt.figure(figsize=(8, 6))
        annot_kws = {"fontsize": 6}
        ax = sns.heatmap(feat_importances_df,
                         cmap="coolwarm",
                         annot=True,
                         fmt=".2f",
                         annot_kws=annot_kws)
        ax.set_title("Feature Importance Heatmap")
        plt.gca().tick_params(axis='x', labelsize=5)
        plt.gca().tick_params(axis='y', labelsize=5)
        plt.tight_layout()
        plt.savefig("static/images/feat_importance_heatmap_unclipped")
        plt.clf()

    def get_runtime(self):
        keys = self.elapsed_times.keys()
        values = self.elapsed_times.values()
        # Table:
        time_df = pd.DataFrame(self.elapsed_times, index=["elapsed time"])
        html_time = time_df.to_html
        # Just plot keys and values of each
        plt.figure(figsize=(6, 4))
        plt.bar(keys, values)
        plt.xlabel("Models")
        plt.ylabel("elapsed time (s)")
        plt.xticks(rotation=45, fontsize=6)
        plt.title("Elapsed time for ML models")
        for i, v in enumerate(values):
            plt.annotate(f"{v:.4f}",
                         xy=(i, v),
                         ha="center",
                         va="bottom",
                         fontsize=6)
        plt.tight_layout()
        plt.savefig("static/images/elapsed_time")
        plt.clf()
        return html_time

    def plot_model(self,
                   model,
                   transformer):
        if model == "gbr_quantile":
            self.plot_quantile_gbr(transformer=transformer)
        elif model == "bayesian_ridge":
            self.plot_bayesian_ridge(transformer=transformer)
        else:
            self.plot_general_model(model=model,
                                    transformer=transformer)

    def plot_quantile_gbr(self,
                          transformer):
        fig = plt.figure(figsize=(10, 6))
        y_test_inv = transformer.inverse_transform(self.y_test.to_numpy().reshape(-1,
                                                                                  1)).flatten()
        y_pred_first_inv = transformer.inverse_transform(self.gbr_quant_pred[self.lower]["Test"].reshape(-1,
                                                                                                         1)).flatten()
        y_pred_med_inv = transformer.inverse_transform(self.gbr_quant_pred[self.med]["Test"].reshape(-1,
                                                                                                     1)).flatten()
        y_pred_last_inv = transformer.inverse_transform(self.gbr_quant_pred[self.upper]["Test"].reshape(-1
                                                                                                        ,1)).flatten()
        results_gbr_df_inv = pd.DataFrame({"y_test": y_test_inv,
                                           f"y_pred_{self.lower}": y_pred_first_inv,
                                           f"y_pred_{self.med}": y_pred_med_inv,
                                           f"y_pred_{self.upper}":y_pred_last_inv,},
                                          index=pd.to_datetime(self.y_test.index))
        results_gbr_df_inv.sort_index(ascending=True, inplace=True)
        plt.plot(results_gbr_df_inv.index, results_gbr_df_inv["y_test"], "k", label="test_value", )
        plt.plot(results_gbr_df_inv.index, results_gbr_df_inv[f"y_pred_{self.med}"], "r-", label="predicted_median", )
        plt.fill_between(results_gbr_df_inv.index,
                         results_gbr_df_inv[f"y_pred_{self.lower}"],
                         results_gbr_df_inv[f"y_pred_{self.upper}"],
                         alpha=0.5,
                         label="0.10 to 0.90 interval prediction")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("LMP")
        plt.title("Time Series for GBR Quantile")
        plt.savefig("static/images/time_series_plot")
        plt.clf()

    def plot_bayesian_ridge(self,
                            transformer):
        def get_prediction_intervals(mean,
                                     n,
                                     error,
                                     alpha=0.01):
            t_value = np.abs(t.ppf(alpha/2, n-1))
            lower = mean - t_value*error
            upper = mean + t_value*error
            return lower, upper

        # Inverse Transform Results and y_Test
        results_bayesian_inv = transformer.inverse_transform(self.results["bayesian_ridge"]["Test"].reshape(-1,
                                                                                                            1)).flatten()
        y_test_inv = transformer.inverse_transform(self.y_test.to_numpy().reshape(-1, 1)).flatten()
        length = len(y_test_inv)
        mean = results_bayesian_inv

        # Calculate lower and upper bounds
        lower, upper = get_prediction_intervals(mean=mean,
                                                n=length,
                                                error=self.bayesian_std["Test"])

        # Store results in DF
        result_df_inv = pd.DataFrame({"y_test": y_test_inv}, index=pd.to_datetime(self.y_test.index))
        result_df_inv.sort_index(ascending=True,
                                 inplace=True)
        result_df_inv["upper"] = upper
        result_df_inv["lower"] = lower
        result_df_inv["y_pred"] = results_bayesian_inv
        print("error")
        print(self.bayesian_std["Test"])
        print("upper")
        print(upper)
        print("lower")
        print(lower)
        print("bayesian")
        print(results_bayesian_inv)
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(result_df_inv.index,
                 result_df_inv["y_test"],
                 "k",
                 alpha=0.2,
                 label="y_test")
        plt.plot(result_df_inv.index,
                 result_df_inv["y_pred"],
                 "b",
                 label="y_pred_bayesian")
        plt.fill_between(result_df_inv.index,
                         result_df_inv["lower"],
                         result_df_inv["upper"],
                         color="orange",
                         alpha=0.7,
                         label="upper and lower intervals")
        plt.legend()
        plt.title(f"Time-Series Bayesian Ridge Model")
        plt.xlabel("Time")
        plt.ylabel("LMP")
        plt.savefig("static/images/time_series_plot")
        plt.clf()

    def plot_general_model(self,
                           model,
                           transformer):
        # Transform pred and test/true data
        y_pred = transformer.inverse_transform(self.results[model]["Test"].reshape(-1,
                                                                                   1)).flatten()
        y_test_inv = transformer.inverse_transform(self.y_test.to_numpy().reshape(-1,
                                                                                  1)).flatten()
        # Store in DF
        results_df_inv = pd.DataFrame({"y_test":y_test_inv,
                                       "y_pred":y_pred},
                                      index=pd.to_datetime(self.y_test.index))
        results_df_inv.sort_index(ascending=True,
                                  inplace=True)
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(results_df_inv.index,
                 results_df_inv["y_test"],
                 "b-",
                 label="test_value")
        plt.plot(results_df_inv.index,
                 results_df_inv["y_pred"],
                 "r-",
                 label="predicted_value")
        plt.title(f"Prediction of {model}")
        plt.xlabel("Time")
        plt.ylabel("LMP")
        plt.legend()
        plt.tight_layout()
        plt.savefig("static/images/time_series_plot")
        plt.clf()

    def get_feat_importance_specific(self,
                                     model):
        annotation_text = None
        if model == "polynomial":
            plt.text(0.5,
                     0.5,
                     "More complicated feature importance for polynomial of 2nd degree. No visualization.",
                     fontsize=12,
                     ha="center",
                     va="center",
                     wrap=True)
            plt.axis("off")
            plt.savefig("static/images/feat_imp_specific")
            plt.clf()
        elif model == "gbr_mse":
            plt.text(0.5,
                     0.5,
                     """This model used HistGradientBoostingRegression which does not support feature importances. The more complex, slower GradientBoostingRegression package runs slower and hence, why HistGBR is used.""",
                     fontsize=12,
                     ha="center",
                     va="center",
                     wrap=True)
            plt.axis("off")
            plt.savefig("static/images/feat_imp_specific")
            plt.clf()
        else:
            plt.figure(figsize=(8, 6))
            model_feat_importances = self.feat_importances[model]
            plt.bar(range(len(model_feat_importances)), model_feat_importances)
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.xticks(range(len(model_feat_importances)),
                       self.X_test.columns,
                       rotation=90,
                       fontsize=6)
            plt.title(f"Feature Importance for {model}")
            for i, v in enumerate(model_feat_importances):
                plt.annotate(f"{v:.2f}",
                             xy=(i, v),
                             ha="center",
                             va="bottom",
                             fontsize=6)
            # Set y limits
            plt.ylim(-1, 1)
            plt.tight_layout()
            plt.savefig("static/images/feat_imp_specific")
            plt.clf()



