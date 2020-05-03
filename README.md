# Feature_Selection

<p align="center"><img src="assets/main_thumb.png" width="400"></p>

## Projects

### Feature Importance

* [Feature Importance](https://github.com/cobanov/Feature_Selection/tree/master/Feature%20Importance)
* [YouTube Video](https://youtube.com/MertCobanov)

#### Example
```python
feature_imp = pd.Series(model.feature_importances_, index= X_encoded.columns)
feature_imp.nlargest(10).plot(kind='barh')
```


## Dataset
[Mushroom Dataset - Kaggle](https://www.kaggle.com/uciml/mushroom-classification)
