from flask import Flask, render_template, redirect, url_for, flash, abort
from flask_ckeditor import CKEditor
from flask_bootstrap import Bootstrap
from datetime import datetime
from weatherlocator import WeatherLocator
from forms import InfoForm
from dbmanager import DatabaseManager
from usdphp import Forex
from preprocessor import PreProcessor
from machinelearner import MLModel
from cleaner import Cleaner



year_now = datetime.now().year
start = datetime(2022, 9, 1)
end = datetime(2022, 11, 29, 23, 59)

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"
ckeditor = CKEditor(app=app)
Bootstrap(app=app)
weatherlocator = WeatherLocator()
forex = Forex()
node_plot = None
geo_fig_html = None
weather_graph_html = None
forex_data = forex.download_forex()
iemop_df = None
weather_data = None
selected_model = None
html_time_table = None
cleaner = Cleaner()
cleaner.clean()


@app.route("/", methods=["GET", "POST"])
def home():
    databasemanager = DatabaseManager()
    info_form = InfoForm()
    global forex_data
    global iemop_df
    global weather_data
    global node_plot
    global geo_fig_html
    global weather_graph_html
    global selected_model
    global html_time_table

    if info_form.validate_on_submit() and info_form.model_field and info_form.node and info_form.address:
        geo_fig_html = weatherlocator.plot_points(location=info_form.address.data)
        weather_graph_html, weather_data = weatherlocator.get_weather()
        selected_node = info_form.node.data
        databasemanager.make_query_and_process(node=selected_node)
        node_plot, iemop_df = databasemanager.plot()
        selected_model = info_form.model_field.data
        preprocessor = PreProcessor(iemop_data=iemop_df,
                                    weather_data=weather_data,
                                    forex_data=forex_data)
        X_train, X_cross, X_test, y_train, y_cross, y_test = preprocessor.retrieve_data()
        machinelearningmodel = MLModel(X_train=X_train,
                                       X_cross=X_cross,
                                       X_test=X_test,
                                       y_train=y_train,
                                       y_cross=y_cross,
                                       y_test=y_test)
        html_time_table = machinelearningmodel.get_runtime()
        machinelearningmodel.get_feat_importance_specific(model=selected_model)
        machinelearningmodel.plot_model(model=selected_model,
                                        transformer=preprocessor.pt)

    return render_template(template_name_or_list="index.html",
                           fig_html=geo_fig_html,
                           weather_graph_html=weather_graph_html,
                           node_graph_html=node_plot,
                           info_form=info_form,
                           selected_model=selected_model,
                           html_time_table=html_time_table,)

#https://psgc.gitlab.io/api/


if __name__ == "__main__":
    app.run(debug=True)