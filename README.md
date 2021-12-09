# Project 2 Ames Housing

### Problem Statement

Determining the sale price of a house is often too complicated due to the great number of features that influence pricing decision such as number of bedrooms, lot size, floor plan and etc. As a data scientist working for a real estate firm, our employer hopes to use the Ames housing data to help assess whether asking price of a house is higher or lower than the true value of the house in Ames, Iowa. If the home is undervalued, it may be a good investment for the firm. 

Our task is to explore and to find the key features that influences the sale price and develop a regression model to be able to accurately predicts the sale price for a given house in Ames, Iowa. Also, with a recommendation of top 3 features which parts of the house to improve to raise the house sale price and also 3 features that will lead to a decrease in the house sale price. 

A successful housing price prediction model should be able to predict housing prices with a root mean square error (RMSE) that is ideally lower than \$25,000.

### Background

The great housing boom in the United States continues unabated after eight years of strong house price growth. The pandemic created a frenzied real estate market in much of the United States that has yet to let up, with demand for housing still outpacing the number of homes coming on the market, giving sellers a heavy upper hand in most of the country. A limited supply of properties in the market has added to upward house price pressure.

REAL estate investors acquired a record 18 per cent of US homes sold in the third quarter of 2021, wagering USD64 billion that home prices and rents will continue to surge. Investors bought more than 90,000 homes in the three months through September, up 10 per cent from the prior quarter and 80 per cent from a year earlier, according to a report by Redfin Corp ([*source*](https://www.businesstimes.com.sg/real-estate/property-investors-bet-us64b-on-us-homes-in-record-buying-spree)).

The S&P/Case-Shiller seasonally-adjusted national home price index rose by an amazing 19.7% during the year to July 2021 (13.61% inflation-adjusted), a sharp acceleration from the previous year’s 4.85% growth and the biggest y-o-y increase ever recorded. The median sales price of new homes sold soared 20.1% y-o-y in August 2021 to USD 390,900 according to the U.S. Census Bureau. For existing homes, the median price was up by 14.9% to USD 356,700 in August 2021 from a year earlier, according to the National Association of Realtors (NAR) ([*source*](https://www.globalpropertyguide.com/North-America/United-States/Price-History)).

### Contents
- Part 1 Exploratory Data Analysis(EDA) & Cleaning
- Part 2 Preprocessing and Modeling

### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|ms_zoning|object|train_clean.csv|The general zoning classification of the house sale|
|street|object|train_clean.csv|Type of road access to property| 
|alley|object|train_clean.csv|Type of alley access to property|
|lot_shape|object|train_clean.csv|General shape of property| 
|land_contour|object|train_clean.csv|Flatness of the property|
|lot_config|object|train_clean.csv|Lot configuration|
|land_slope|object|train_clean.csv|Slope of property| 
|neighborhood|object|train_clean.csv|Physical locations within Ames city limits| 
|condition_1|object|train_clean.csv|Proximity to various conditions| 
|condition_2|object|train_clean.csv|Proximity to various conditions (if more than one is present)| 
|bldg_type|object|train_clean.csv|Type of dwelling| 
|house_style|object|train_clean.csv|Style of dwelling|
|overall_qual|int64|train_clean.csv|Rates the overall material and finish of the house|  
|overall_cond|int64|train_clean.csv|Rates the overall condition of the house|  
|year_built|int64|train_clean.csv|Original construction date|  
|year_remod_add|int64|train_clean.csv|Remodel date (same as construction date if no remodeling or additions)| 
|roof_style|object|train_clean.csv|Type of roof|
|roof_matl|object|train_clean.csv|Roof material|
|exterior_1st|object|train_clean.csv|Exterior covering on house|
|exterior_2nd|object|train_clean.csv|Exterior covering on house (if more than one material)| 
|mas_vnr_type|object|train_clean.csv|Masonry veneer type|
|mas_vnr_area|float64|train_clean.csv|Masonry veneer area in square feet|
|exter_qual|object|train_clean.csv|Evaluates the quality of the material on the exterior| 
|exter_cond|object|train_clean.csv|Evaluates the present condition of the material on the exterior| 
|foundation|object|train_clean.csv|Type of foundation| 
|bsmt_qual|object|train_clean.csv|Evaluates the height of the basement| 
|bsmt_cond| object|train_clean.csv|Evaluates the general condition of the basement| 
|bsmt_exposure|object|train_clean.csv|Refers to walkout or garden level walls| 
|bsmtfin_type_1|object|train_clean.csv|Rating of Type 1 basement finished area| 
|bsmtfin_sf_1|float64|train_clean.csv|Type 1 finished square feet|
|bsmtfin_type_2|object|train_clean.csv|Rating of Type 2 basement finished area (if multiple types)| 
|bsmtfin_sf_2|float64|train_clean.csv|Type 2 finished square feet|
|bsmt_unf_sf|float64|train_clean.csv|Unfinished square feet of basement area|
|total_bsmt_sf|float64|train_clean.csv|Total square feet of basement area|
|heating|object|train_clean.csv|Type of heating| 
|heating_qc|object|train_clean.csv|Heating quality and condition| 
|central_air|object|train_clean.csv|Central air conditioning| 
|electrical|object|train_clean.csv|Electrical system| 
|1st_flr_sf|int64|train_clean.csv|First Floor square feet|  
|2nd_flr_sf|int64|train_clean.csv|Second floor square feet|  
|log_gr_liv_area|float64|train_clean.csv|Log of grade (ground) living area square feet|
|bsmt_full_bath|float64|train_clean.csv|Basement full bathrooms|
|bsmt_half_bath|float64|train_clean.csv|Basement half bathrooms|
|full_bath|int64|train_clean.csv|Full bathrooms above grade|  
|half_bath|int64|train_clean.csv|Half baths above grade|  
|bedroom_abvgr|int64|train_clean.csv|Bedrooms above grade (does NOT include basement bedrooms)| 
|kitchen_abvgr|int64|train_clean.csv|Kitchens above grade|  
|kitchen_qual|object|train_clean.csv|Kitchen quality| 
|totrms_abvgrd|int64|train_clean.csv|Total rooms above grade (does not include bathrooms)| 
|functional|object|train_clean.csv|Home functionality (Assume typical unless deductions are warranted)| 
|fireplaces|int64|train_clean.csv|Number of fireplaces|  
|fireplace_qu|object|train_clean.csv|Fireplace quality| 
|garage_type|object|train_clean.csv|Garage location|
|garage_yr_blt|float64|train_clean.csv|Year garage was built|
|garage_finish|object|train_clean.csv|Interior finish of the garage| 
|garage_cars|float64|train_clean.csv|Size of garage in car capacity|
|garage_area|float64|train_clean.csv|Size of garage in square feet|
|garage_qual|object|train_clean.csv|Garage quality| 
|garage_cond|object|train_clean.csv|Garage condition| 
|paved_drive|object|train_clean.csv|Paved driveway| 
|wood_deck_sf|int64|train_clean.csv|Wood deck area in square feet|  
|open_porch_sf|int64|train_clean.csv|Open porch area in square feet| 
|enclosed_porch|int64|train_clean.csv|Enclosed porch area in square feet|  
|screen_porch|int64|train_clean.csv|Screen porch area in square feet|  
|misc_feature|object|train_clean.csv|Miscellaneous feature not covered in other categories| 
|mo_sold|int64|train_clean.csv|Month Sold (MM)|
|yr_sold|int64|train_clean.csv|Year Sold (YYYY)|  
|sale_type|object|train_clean.csv|Type of sale| 

### Conclusions

1. Linear regression is the most basic form, where the model is not penalized for its choice of weights, at all. That means, during the training stage, if the model feels like one particular feature is particularly important, the model may place a large weight to the feature. These models tend to overfitting. Hence, Regularization is needed. Regularization is an important concept that is used to avoid overfitting of the data, especially when the trained and test data are much varying.

2. Ridge Regression is a regularizaton technique that includes an L2 penalty. This has the effect of shrinking the coefficients for those input variables that do not contribute much to the prediction task. Therefore, it prevent multicollinearity and reduces the model complexity by coefficient shrinkage.
    - <p>Limitation of Ridge Regression: It shrinks coefficients towards zero but not absolute, thus it includes almost all the predictors and not capable of performing feature selection. Also, it trades variance for bias.</p>
   
3. Lasso Regression is a regularization technique which is a very useful method to handle collinearity, filter out noise from data, and eventually prevent overfitting. In addition, Lasso Regression selects only some feature while reduces the coefficients of others to zero by imposing a constraint on the model parameters. Variables with a regression coefficient equal to zero after the shrinkage process are excluded from the model. Variables with non-zero regression coefficients variables are most strongly associated with the response variable. Thus, in this project, Lasso Regression picked 120 features and eliminated the other 86 features. This property is known as feature selection and which is absent in case of linear & ridge regression.
    - <p>Limitation of Lasso Regression: Lasso sometimes struggles with some types of data. If the number of predictors is greater than the number of observations, Lasso will pick at most n predictors as non-zero, even if all predictors are relevant. Also, if there are two or more highly collinear variables then Lasso regression select one of them randomly which is not good for the interpretation of data</p>
    
4. In this project, Lasso Regression model had the best predictive performance on housing sale price in Ames, Iowa, and outperformed the other linear model tested (Linear & Ridge Regression) based on two metrics comparison (RMSE & R2). Lasso Model able to explain approximately 92.5% of the variability or fluctuation in the sample test data sale price by the predictor features. Also, Lasso Model able to predict the house sample test data sale price with only \$17,762.94 root mean square error (RMSE).

5. Based on finding above, total house square feet, overall quality & overall condition are the top 3 features which parts of the house to improve to raise the house sale price. e.g:
    - Holding all else constant, for every one-unit increase in the house square feet (1 square feet), house sale price increases by about 15.3%. E.g : An average house sale price in Ames, Iowa is \$177,633.33. Thus, increase house square feet by one square feet the house sale price increase by \$27,178. This number is pretty high, which makes it easy for our model to become overfit, as even a slight increase in total house square feet will lead to a much larger shift as compared to a unit change for any of our other features.
    - Holding all else constant, for every one-unit increase in the house overall quality (Rates of overall material and finish of the house), house sale price increases by about 8%. Thus, increase the rates of overall material and finish of the house by 1 score the house sale price increase by \$14,210.
    - Holding all else constant, for every one-unit increase in the house overall condition (Rates the overall condition of the house), house sale price increases by about 4.3%. Thus, increase the rates of overall condition of the house by 1 score the house sale price increase by \$7,638.</p>    

6. Based on finding above, house age, townhouse inside unit and unfinished square feet of basement area the top 3 features that will lead to a decrease in the house sale price. e.g:
    - Holding all else constant, for every one-unit increase in the house age (1 year), house sale price decreases by about 5.5%. Thus, increase the house age by 1 year the house sale price decrease by \$9,770.
    - Holding all else constant, the effect of it being townhouse inside unit, house sale price decreases by about 1.86% (\$3,304).
    - Holding all else constant, for every one-unit increase in the house unfinished square feet of basement area (1 square feet), house sale price decreases by about 1.63%. Thus, increase the house unfinished square feet of basement area the house sale price decrease by \$2,895.</p>


### Recommendations

Based on our model, as a real estate firm looking to increase the selling price of the house could do the following:

1. Increase the overall material and finish quality of the house through renovation and painting.
2. Improve the house overall condition through cleaning, renovate the garage if it is in bad condition and etc.
3. Increase garage size to allow it to fit more than one car.
4. Increase the number of bathrooms in the house, or renovate existing bedroom to add additional bathroom (if the house has more than four bedrooms).
5. Switch to a brick exterior if using a hardboard or stucco exterior.
6. Avoid holding the house for too long, as prices for all house types decreased with age.
7. Finishing an unfinished  basement as a finished basement will add significant value to your property. Also, it will create additional living space.

### Limitations & Improvement

As the model was developed using data on houses sold between 2006 - 2010 in Ames, USA, it may have limited applicabilities.

1. The model only accounted for 92.5% of the variations in sample test data sale price. The remaining 7.5% could be due to factors related to area desirability (i.e. location). In the current dataset, only neighborhoods and proximity to roads are included under this category. In reality, factors such as the presence of schools, hospitals and malls are some examples of other factors that are also likely to affect house price.

2. In addition, it captures only a small time frame of four years. This is not enough to capture any annual patterns in sale price that could arise as a result of external factors, such as policy changes. This model also doesn't take into account the inflation of housing prices. Since the end of the financial crisis in 2008, housing prices throughout the US have been increasing steadily year over year. Our model would need significant retraining to predict the current house prices in Ames today.

3. The model is specific to houses in Ames and may not be as accurate when applied to data from another city given that each city tends to differ greatly in terms of external factors like geographical features, seasonal weather or the economic climate of that particular city.

Therefore, to improve the applicability and accuracy of the model to predict today price consider adding the following:

1. A wider time frame
2. Different locations & seasonal weather
3. Availability of facilities nearby

### References

1. https://www.globalpropertyguide.com/North-America/United-States/Price-History
2. https://walletinvestor.com/real-estate-forecast/ia/story/ames-housing-market
3. https://data.census.gov/cedsci/table?q=DP04&tid=ACSDP5Y2019.DP04
4. https://www.statista.com/statistics/200445/reported-violent-crime-rate-in-the-us-states/
5. https://www.amestrib.com/story/news/2020/10/16/ames-ranked-top-15-places-live/3678012001/
6. https://data.library.virginia.edu/interpreting-log-transformations-in-a-linear-model/
