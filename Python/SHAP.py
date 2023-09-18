
import pandas as pd

data = pd.read_csv("T.csv")
data.columns = data.columns.map(lambda row: "_".join(row.lower().split(" ")))
data


from patsy import dmatrices

y, X = dmatrices(
    "TLS ~ sex + age + CEA + CA199 + lauren + location + G + Siz + T + N + RS-1"
    ,
    data=data,
)

X_frame = pd.DataFrame(data=X, columns=X.design_info.column_names)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


import xgboost

model = xgboost.XGBClassifier().fit(X_train, y_train)

predict = model.predict(X_test)


from sklearn.metrics import f1_score

f1 = f1_score(y_test, predict)

f1

import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_frame)


shap.plots.waterfall(shap_values[0])

shap.plots.waterfall(shap_values[1])


shap.summary_plot(shap_values[:10000,:], X[:10000,:], plot_type="violin")



shap.summary_plot(shap_values, plot_type="bar",color='cornflowerblue')

shap.plots.bar(shap_values,max_display=20)
