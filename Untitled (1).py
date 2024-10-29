#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
weather = pd.read_csv("3825862.csv", index_col="DATE")


# In[9]:


weather


# In[14]:


weather.apply(pd.isnull).sum()


# In[18]:


core_weather = weather[["PRCP","SNOW","SNWD","TMAX","TMIN"]].copy()


# In[20]:


core_weather.columns = ["precip","snow","snow_depth","temp_max","temp_min"]


# In[22]:


core_weather


# In[24]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[26]:


core_weather["snow"].value_counts()


# In[28]:


core_weather["snow_depth"].value_counts()


# In[30]:


core_weather[pd.isnull(core_weather["precip"])]


# In[32]:


core_weather.loc["2003-08-10":"2003-09-5"]


# In[34]:


core_weather["precip"].value_counts()


# In[38]:


core_weather["precip"] = core_weather["precip"].fillna(0)


# In[40]:


core_weather[pd.isnull(core_weather["snow"])]


# In[42]:


core_weather[pd.isnull(core_weather["snow_depth"])]


# In[44]:


core_weather.loc["1996-03-15":"1996-04-15"]


# In[46]:


core_weather.loc["2017-03-15":"2017-04-20"]


# In[48]:


core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)


# In[54]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[56]:


core_weather.dtypes


# In[58]:


core_weather.index


# In[60]:


core_weather.index = pd.to_datetime(core_weather.index)


# In[62]:


core_weather.index


# In[68]:


core_weather.apply(lambda x: (x==9999).sum())


# In[70]:


core_weather[["temp_max","temp_min"]].plot()


# In[74]:


core_weather.index.year.value_counts()


# In[76]:


core_weather["precip"].plot()


# In[78]:


core_weather["snow"].plot()


# In[80]:


core_weather["snow_depth"].plot()


# In[82]:


core_weather.groupby(core_weather.index.year).sum()


# In[84]:


core_weather["target"] = core_weather.shift(-1)["temp_max"]


# In[86]:


core_weather


# In[90]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[92]:


core_weather


# In[96]:


from sklearn.linear_model import Ridge
reg = Ridge(alpha=.1)


# In[98]:


predictors = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


# In[104]:


train = core_weather.loc[:"2015-12-31"]


# In[106]:


test = core_weather.loc["2016-01-01":]


# In[108]:


reg.fit(train[predictors], train["target"])


# In[110]:


predictions = reg.predict(test[predictors])


# In[112]:


from sklearn.metrics import mean_absolute_error


# In[114]:


mean_absolute_error(test["target"], predictions)


# In[116]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)


# In[118]:


combined


# In[120]:


combined.columns = ["actual", "predictions"]


# In[122]:


combined


# In[124]:


combined.plot()


# In[126]:


reg.coef_


# In[128]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2015-12-31"]
    test = core_weather.loc["2016-01-01":]
    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return  error, combined


# In[132]:


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()


# In[134]:


core_weather


# In[136]:


core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]


# In[138]:


core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]


# In[236]:


predictors = ["precip", "snow", "snow_depth", "temp_max", "temp_min", "month_max", "month_day_max", "max_min"]


# In[238]:


core_weather = core_weather.iloc[30:,:].copy()


# In[240]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[242]:


import numpy as np
core_weather[predictors].applymap(np.isinf).sum()


# In[244]:


core_weather.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[246]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[248]:


core_weather[core_weather['max_min'].isna()]


# In[250]:


core_weather['max_min'] = core_weather['max_min'].fillna(core_weather['max_min'].mean())


# In[252]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[254]:


error


# In[256]:


combined.plot()


# In[258]:


core_weather["monthly_avg"] = core_weather.groupby(core_weather.index.month)["temp_max"].transform(lambda x: x.expanding().mean())


# In[260]:


core_weather


# In[262]:


core_weather["day_of_year_avg"] = core_weather.groupby(core_weather.index.day_of_year)["temp_max"].transform(lambda x: x.expanding().mean())


# In[264]:


core_weather


# In[266]:


predictors = ["precip","snow", "snow_depth", "temp_max", "temp_min", "month_max", "month_day_max", "max_min","day_of_year_avg","monthly_avg"]


# In[268]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[270]:


error


# In[272]:


reg.coef_


# In[274]:


core_weather.corr()["target"]


# In[278]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[280]:


combined.sort_values("diff", ascending = False).head()


# In[302]:


###weekly prediction model:


# In[304]:


core_weather["weekly_avg_temp"] = core_weather["temp_max"].rolling(7).mean().shift(-7)
core_weather = core_weather.dropna(subset=["weekly_avg_temp"])


# In[306]:


core_weather.columns


# In[322]:


predictors = ["precip", "snow", "snow_depth", "temp_max", "temp_min", "month_max", "month_day_max", "max_min", "day_of_year_avg", "monthly_avg"]


# In[324]:


train = core_weather.loc[:"2015-12-31"]
test = core_weather.loc["2016-01-01":]


# In[326]:


reg = Ridge(alpha=100)


# In[328]:


def create_weekly_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2015-12-31"]
    test = core_weather.loc["2016-01-01":]
    reg.fit(train[predictors], train["weekly_avg_temp"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["weekly_avg_temp"], predictions)
    combined = pd.concat([test["weekly_avg_temp"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[330]:


error, combined = create_weekly_predictions(predictors, core_weather, reg)


# In[332]:


error


# In[336]:


combined.plot()


# In[338]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[340]:


combined.sort_values("diff", ascending = False).head()


# In[342]:


error


# In[344]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[346]:


combined.plot()


# In[348]:


error


# In[350]:


core_weather.corr()["target"]


# In[352]:


core_weather.corr()["weekly_avg_temp"]


# In[356]:


core_weather.loc[:, "week_of_year"] = core_weather.index.isocalendar().week


# In[364]:


core_weather = core_weather.copy()


# In[370]:


core_weather.index = core_weather.index.to_series().infer_objects()
core_weather.loc[:, "prev_week_avg_temp"] = core_weather["temp_max"].shift(7).rolling(7).mean()
core_weather.loc[:, "prev_week_avg_precip"] = core_weather["precip"].shift(7).rolling(7).mean()




# In[372]:


core_weather


# In[374]:


core_weather.dropna(inplace=True)
core_weather


# In[376]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(train[predictors], train["weekly_avg_temp"])
predictions = gbr.predict(test[predictors])
error = mean_absolute_error(test["weekly_avg_temp"], predictions)
print("Gradient Boosting Error for weekly_avg_temp:", error)


# In[378]:


from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(train[predictors], train["weekly_avg_temp"])
best_ridge = grid_search.best_estimator_
best_ridge.fit(train[predictors], train["weekly_avg_temp"])
predictions = best_ridge.predict(test[predictors])
error = mean_absolute_error(test["weekly_avg_temp"], predictions)
print("Best Ridge Error for weekly_avg_temp:", error)


# In[380]:


ridge_preds = best_ridge.predict(test[predictors])
gbr_preds = gbr.predict(test[predictors])
final_preds = (ridge_preds + gbr_preds) / 2
error = mean_absolute_error(test["weekly_avg_temp"], final_preds)
print("Ensemble Error for weekly_avg_temp:", error)


# In[408]:


results = pd.DataFrame({
    "Actual": test["weekly_avg_temp"],
    "Ridge_Predictions": ridge_preds,
    "GBR_Predictions": gbr_preds,
    "Ensemble_Predictions": final_preds
}, index=test.index)


# In[410]:


results


# In[416]:


import matplotlib.pyplot as plt
plt.figure(figsize=(14, 7))
plt.plot(results.index, results["Actual"], label="Actual", color="black", linewidth=2)
plt.plot(results.index, results["Ridge_Predictions"], label="Ridge Predictions", linestyle="--")
plt.plot(results.index, results["GBR_Predictions"], label="GBR Predictions", linestyle="--")
plt.plot(results.index, results["Ensemble_Predictions"], label="Ensemble Predictions", color="blue", linewidth=1)

plt.xlabel("Date")
plt.ylabel("Weekly Average Temperature")
plt.title("Actual vs Predicted Weekly Average Temperature")
plt.legend()
plt.grid(True)
plt.show()


# In[418]:


core_weather


# In[420]:


## future training


# In[ ]:




