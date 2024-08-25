import pandas as pd

from datetime import datetime

from mdutils.mdutils import MdUtils
from mdutils import Html

from predictions.predictions import logistic_regression, data_splitting

data_frame = pd.read_csv("soil_measures.csv")
feature_selection = ["N","P","K","ph"]
target_column = "crop"

X_train, X_test, y_train, y_test = data_splitting(data_frame=data_frame,
                                                  feature_selection=feature_selection,
                                                  target_column=target_column)

# Create a dictionary to store the model performance for each feature
feature_performance = {}

for feature in feature_selection:
    f1 = logistic_regression(X_train, X_test, y_train, y_test, feature=feature)
    
    # Add feature-f1 score pairs to the dictionary
    feature_performance[feature] = f1

feature_performance_df = pd.DataFrame(list(feature_performance.items()),
                                      columns=["feature", "f1_value"]
                                      )

print (feature_performance_df)
higest_f1_value_df = feature_performance_df[feature_performance_df["f1_value"] == feature_performance_df["f1_value"].max() ]
best_predicitve_feature = higest_f1_value_df["feature"].values[0]
print (f"best_predicitve_feature: {best_predicitve_feature}")

# creating the deliverable
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
final_file_name = "result_"+current_time

# writing the report
md_file = MdUtils(file_name=f"./models/{final_file_name}", title =final_file_name)
md_file.new_paragraph(f"feature_performance dataframe \n {feature_performance_df}")
md_file.new_paragraph(f"best_predicitve_feature: {best_predicitve_feature}")
md_file.create_md_file()