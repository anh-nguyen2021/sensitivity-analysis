import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
model_ip_data=pd.read_csv("dat.csv")
model_ip_data.ts= pd.to_datetime(model_ip_data.ts)
ts_validate_start = "2021-09-06"
ts_validate_end = "2021-11-01"
ts_validate_duration = "8 weeks"
ts_test_start = "2021-11-01"
ts_test_end = "2021-12-27"
ts_predict_end = "2022-07-04"
forecast_dates = pd.date_range(
    start=ts_test_end,
    end=ts_predict_end,
    freq="W-Mon",
)
val_dates = pd.date_range(
    start=ts_validate_start,
    end=ts_validate_end,
    freq="W-Mon",
)
test_dates = pd.date_range(
    start=ts_test_start,
    end=ts_test_end,
    freq="W-Mon",
)

def mean_encode(data, gb_cols, not_nan_mask=None):
    original_y = data["y"].copy()
    if not_nan_mask is not None:
        data.loc[~not_nan_mask, "y"] = np.nan

    for col in gb_cols:
        if not isinstance(col, str):
            col_name = "meanenc_" + "_".join(col) + "_"
        else:
            col_name = "meanenc_" + "".join(col) + "_"
        for s in ["mean", "std", "max"]:
            data[col_name + s] = data.groupby(col)["y"].transform(s)

    data["y"] = original_y
    return data


def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

from sklearn.inspection import permutation_importance

def single_lgbm_per_customer( val_dates, test_dates, forecast_dates, drop_features, drop, model_name="LGBM",group_var="customer_group", ):
    
	datas = {}
	f = model_ip_data
	train = f.loc[f.ts < np.min(val_dates)].reset_index(drop=True)
	#train = train.loc[train.ts > "2020-01-01"].reset_index(drop=True)
	val = f.loc[f.ts.isin(val_dates)].reset_index(drop=True)
	test = f.loc[f.ts.isin(test_dates)].reset_index(drop=True)
	test1 = test.copy()
	predict = f.loc[f.ts.isin(forecast_dates)].reset_index(drop=True)

	train["y"] = train["y"].astype(float)
	val["y"] = val["y"].astype(float)
	test["y"] = test["y"].astype(float)
	predict["y"] = np.nan

	a = mean_encode(train.copy(), ["id"])
	mean_enc_features = a[["id"] + [x for x in a.columns if "meanenc" in x]].drop_duplicates()

	train = pd.merge(train, mean_enc_features, on="id", how="left")
	val = pd.merge(val, mean_enc_features, on="id", how="left")
	test = pd.merge(test, mean_enc_features, on="id", how="left")
	
	predict = pd.merge(predict, mean_enc_features, on="id", how="left")

	x_train, y_train = train.drop(columns=["y", "ts"]), train["y"]
	x_val, y_val = val.drop(columns=["y", "ts"]), val["y"]
	x_test, y_test = test.drop(columns=["y", "ts"]), test["y"]
	x_predict, y_predict = predict.drop(columns=["y", "ts"]), predict["y"]

	enc_cols = x_train.select_dtypes(exclude="number").columns.to_list()
	encoders = {}
	for col in enc_cols:
	    enc = LabelEncoder()
	    x_train[col] = enc.fit_transform(x_train[col])
	    x_val[col] = enc.transform(x_val[col])
	    x_test[col] = enc.transform(x_test[col])
	    x_predict[col] = enc.transform(x_predict[col])
	    encoders.update({col: enc})
	print(train.ts.min(), train.ts.max(), train.shape)
	print(val.ts.min(), val.ts.max(), val.shape)
	print(test.ts.min(), test.ts.max(), test.shape)
	print(predict.ts.min(), predict.ts.max(), predict.shape)

	datas  = {
	        "train": train,
	        "val": val,
	        "test": test,
	        "predict": predict,
	        "x_train": x_train,
	        "y_train": y_train,
	        "x_val": x_val,
	        "y_val": y_val,
	        "x_test": x_test,
	        "y_test": y_test,
	        "x_predict": x_predict,
	        "y_predict": y_predict,
	    }


	x_train = datas["x_train"]
	y_train = datas["y_train"]

	_model = {}
	for sn in x_train[group_var].unique():
	    _x_train = x_train.loc[x_train[group_var] == sn].drop(columns=drop_features)
	    _y_train = y_train.loc[x_train[group_var] == sn].drop(columns=drop_features)
	    model = LGBMRegressor(
	        n_estimators=1000,
	        boosting="gbdt",
	        learning_rate=0.01,
	        objective="tweedie",
	        tweedie_variance_power=1.5,
	        colsample_bytree=0.9,
	        subsample=0.9,
	        force_col_wise=True,
	        reg_lambda=100,
	        reg_alpha=1000,
	        n_jobs=-1,  
	        verbose=1,
	        random_state=42
	    )
	    model.fit(_x_train, _y_train)
	    print("fitting")
	    _model.update({sn: model})

	x_train = datas["x_train"].drop(columns=drop_features)
	cus=["Robinsons", "Nebraska"]
	fi=[]
	ttt= _model.copy()
	print(ttt)
	feas=x_train.columns.to_list()
	print("features list: ", feas)
    


	predict_df = []

	x_train = datas["x_train"]
	x_val = datas["x_val"]
	x_test = datas["x_test"]
	x_predict = datas["x_predict"]

	# scores = {}
	for sn in _model.keys():
	    model = _model[sn]
	    model.set_params(**{"verbose": 0})
	    yhat_train = model.predict(x_train.loc[x_train[group_var]==sn].drop(columns=drop_features))
	    yhat_val = model.predict(x_val.loc[x_val[group_var]==sn].drop(columns=drop_features))
	    yhat_test = model.predict(x_test.loc[x_test[group_var]==sn].drop(columns=drop_features))
	    yhat_predict = model.predict(x_predict.loc[x_predict[group_var]==sn].drop(columns=drop_features))

	    sn2 = encoders[group_var].inverse_transform([sn])[0]
	    a = pd.concat([
	        datas["train"].loc[datas["train"][group_var]==sn2], 
	        datas["val"].loc[datas["val"][group_var]==sn2], 
	        datas["test"].loc[datas["test"][group_var]==sn2],
	        datas["predict"].loc[datas["predict"][group_var]==sn2]
	    ]).reset_index(drop=True)
	    # score = model.score(x_test.loc[x_test[group_var]==sn].drop(columns=drop_features), datas["test"].loc[datas["test"][group_var]==sn2]['y'].fillna(0))
	    # scores[sn2] = score
	    a["yhat"] = np.concatenate((yhat_train, yhat_val, yhat_test, yhat_predict))
	    a["yhat"] = np.clip(np.round(a["yhat"]), 0, None)
	    a["model"] = model_name
	    predict_df.append(a)
	predict_df = pd.concat(predict_df).reset_index(drop=True)  


	overall_ts = predict_df.fillna(0).groupby(["ts"]).sum().reset_index()
	train = overall_ts.loc[overall_ts.ts < val_dates[0]]
	val = overall_ts.loc[overall_ts.ts.isin(val_dates)]
	test = overall_ts.loc[overall_ts.ts.isin(test_dates)]
	forecasts = overall_ts.loc[overall_ts.ts.isin(forecast_dates)]
	# accuracy = 
	_max = overall_ts[["y", "yhat"]].max().max()
	test["ape"] = abs(test["y"] - test["yhat"]) / test["y"]
	test["ape"] = test["ape"]*100
	test["ape"] = test["ape"].round(2)
	test = test[~test['ape'].isin([np.nan, np.inf, -np.inf])]
	test['accuracy'] = 100-  test['ape']
	
	# test['mad'] = np.abs(test["y"] - test["yhat"])
	# test = test.groupby(['ts']).agg({
	# 'y':'sum',
	# 'yhat':'sum',
	# 'mad':'sum'
	# }).reset_index()
	# test["ape"] = test['mad'] / test["yhat"]
	# test["ape"] = test["ape"]
	# test["ape"] = test["ape"].round(4)
	# test['accuracy'] = (1 - test['ape'])*100
	# import pdb; pdb.set_trace()
	col1, col2 = st.columns((4, 1))
	st.write("  ")
	col1.write(" accuracy {:.2f} % ".format(test.accuracy.mean()))
	csv = convert_df(predict_df)

	col2.download_button(
     label="Save result",
     data=csv,
     file_name='resultt.csv',
     mime='text/csv'
 )
	fig = plt.figure(figsize=(15,8))
	sns.set(font_scale = 2)
	plt.plot(overall_ts.ts, overall_ts.y, label="actual")
	plt.plot(train.ts, train.yhat, label="train")
	plt.plot(val.ts, val.yhat, label="val")
	plt.plot(test.ts, test.yhat, label="test")
	plt.plot(forecasts.ts, forecasts.yhat, label="forecast")
	plt.xlim([overall_ts.ts.min(), overall_ts.ts.max()])
	plt.ylabel("Overall sales")
	plt.legend()
	st.pyplot(fig)
		

	r = permutation_importance(ttt[1], x_test.loc[x_test[group_var]==1].drop(columns=drop_features), test1[test1[group_var]==cus[0]]["y"].fillna(0), n_repeats=10, random_state=0)
	fea_imp = pd.DataFrame({"fea": feas, "imp": r.importances_mean})
	fea_imp.sort_values(by="imp", inplace=True, ascending=False)
	fig1=plt.figure(figsize=(15, 10))
	sns.set(font_scale = 2)
	sns.barplot(data=fea_imp, x="imp", y="fea")
	plt.title(" Customer: {}".format( cus[0]))
	plt.show()
	st.pyplot(fig1)   

	r = permutation_importance(ttt[0], x_test.loc[x_test[group_var]==0].drop(columns=drop_features), test1[test1[group_var]==cus[1]]["y"].fillna(0), n_repeats=10, random_state=0)
	fea_imp = pd.DataFrame({"fea": feas, "imp": r.importances_mean})
	fea_imp.sort_values(by="imp", inplace=True, ascending=False)
	fig1=plt.figure(figsize=(15, 10))
	sns.set(font_scale = 2)
	sns.barplot(data=fea_imp, x="imp", y="fea")
	plt.title(" Customer: {}".format( cus[1]))
	plt.show()
	st.pyplot(fig1)    

	return predict_df, test
    



drop_features1 = ["woy","dpd","dpm", "price","max_price","material","customer_group" ,"Customer", "Prod_Category","Prod_Brand", "Basic_Material",  "Material_Description",
                 "Channel", "Company","dist_comp_categ","max_list_price","list_price","new_price", "new_price_median","dummy_spike", 
                 "dummy_event_spike", "dummy_spike_before_event", "dummy_expect_a_spike", "weeks_since_last_event","weeks_to_next_event", "lag_demand_4_weeks"]


st.write("# Feature Selection")
options = st.multiselect(
   'Select feature to drop',
   ["lag_demand", 'new_price_min','days_since_launch','meanenc_id_std','dpw', 'year', 'month', 'wom', 'days_since_launch', 'events', 'type', 'festival', 'promotion', 'promotion_specific_date', 'last_event', 'next_event'
, 'days_since_last_event', 'days_to_next_event', 'dummy_expect_a_spike_price', 'meanenc_id_mean', 'meanenc_id_max', 'id'])


tmp = drop_features1.copy()
for o in options:
	tmp.append(o)
# st.write("### Stats of "+option)
# a= model_ip_data.agg({option:["min", "max", "median", "mean"]})
# st.write(a)
predict_df, test = single_lgbm_per_customer( val_dates=val_dates, test_dates=test_dates, forecast_dates=forecast_dates,drop_features=tmp, drop=options)
# st.write("# Forecast vs Actual")





