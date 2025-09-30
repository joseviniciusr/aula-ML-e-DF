"""
**Python library to easily automate data fusion models (low-, mid-, high-levels)**
 
**Description**:

Additionally to modeling (by PLS, RF, SVM or MLP), the individual modeling function independently extracts the Latent Variable Scores (-based on PLS transform).

The data fusion functions are mainly powered by dictionaries where the keys represent all employed sensors (m). Then, all possible combinations ranging from 2 to m are modeled along with several performance metrics are extracted.

**Author**: Msc. Jose Vinicius Ribeiro

**Applied Nuclear Physics Laboratory**, State University of Londrina, Brazil

**Contact**: ribeirojosevinicius@gmail.com
"""

import pandas as pd
import numpy as np


def optimized_individual_modeling(datacal, ycal, target, model='pls', datapred=None, ypred=None, 
                                maxLV=None, kern=None, random_seed=None, nnlayers=[100, 50, 20], actf='relu', LVscores=None):
    """
    ## Individual modeling (calibration and prediction) via pls, rf or svm
     **Latent variables are extractedas well**

    **Input parameters**:
    - **datacal**: pd.DataFrame with the X calibration data
    - **datapred**: pd.DataFrame with the X prediction data (Optional)
    - **Ycal**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input
    - **Ypred**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input (Optional)
    - **target**: string representing the target variable (according to the column names of Ycal/Ypred)
    - **model**: the model to be used, chosen among 'pls', 'rf', 'svm', 'mlp'
    - **maxLV**: If model='pls', maximum number of LVs to be used
    - **kern**: If model='svm', the kernel must be chosen among 'linear', 'rbf', 'sigmoid' 
    - **nnlayers**: List of integers representing the number of neurons in each layer (if model='mlp')
    - **actf**: String representing the activation function to be used in the MLP model. Options: ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    - **random_seed**: Int. If model='rf', the random seed should be chosen

    **Return**:
    - If LVscores = True:

         (predictor, df_results, calres, lv_scorescal, predres, lv_scorespred)         
    - Else:

         *predictor, df_results, calres, predres)

    **Where**:
    - **predictor**: the trained model (PLS, RF, SVM or MLP)
    - **df_results**: DataFrame with performance metrics
    - **calres**: DataFrame with predictions in calibration (with 'Ref' column of real values).
    - **lv_scorescal**: DataFrame with the LV scores in the calibration.
    - **predres**: DataFrame with the predictions in the prediction (with 'Ref' column of actual values, if available).
    - **lv_scorespred**: DataFrame with the LV scores in the prediction.     
    """

    # List to store metric results
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr

    results = []
    ycal=ycal[target]
    ypred=ypred[target] if (ypred is not None and target in ypred) else None

    # DataFrames to store the predicted calibration and prediction values ​​(if provided by datapred)
    calres = pd.DataFrame(index=range(len(ycal)))
    predres = pd.DataFrame(index=range(len(ypred)))  if (datapred is not None and ypred is not None) else None

    if model == 'pls':
        if maxLV is None:
            print("For the 'pls' model, enter the desired number of latent variables in the LV argument")
            return None, None, None, None, None
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        # Loop for each number of latent variables (from 1 to n_max_comp)
        kern = None
        random_seed = None
        for n_comp in range(1, maxLV + 1):
            # Define the PLS model with the current number of latent variables
            pls = PLSRegression(n_components=n_comp, scale=False)
            
            # Fit the model to the calibration data
            pls.fit(datacal, ycal)
            y_cal = pls.predict(datacal).flatten()  # Predicted values ​​for calibration (flattens to 1D)

            # Adds column of predicted calibration values ​​for current number of LVs
            calres[f'LV_{n_comp}'] = y_cal
            
            # Cross-validation
            y_cv = cross_val_predict(pls, datacal, ycal, cv=10).flatten()
            
            # Calculates calibration metrics
            R2_cal = r2_score(ycal, y_cal)
            r2_cal = (np.corrcoef(ycal, y_cal)[0, 1]) ** 2
            rmse_cal = np.sqrt(mean_squared_error(ycal, y_cal))

            # Calculates cross-validation metrics
            R2_cv = r2_score(ycal, y_cv)
            r2_cv = (np.corrcoef(ycal, y_cv)[0, 1]) ** 2
            rmsecv = np.sqrt(mean_squared_error(ycal, y_cv))
            bias_cv = sum(ycal - y_cv)/ycal.shape[0]
            SDV_cv = (ycal - y_cv) - bias_cv
            SDV_cv = SDV_cv*SDV_cv
            SDV_cv = np.sqrt(sum(SDV_cv)/(ycal.shape[0] - 1))
            tbias_cv = abs(bias_cv)*(np.sqrt(ycal.shape[0])/SDV_cv)
            rpd_cv = ycal.std() / rmsecv
            rpiq_cv = iqr(ycal, rng=(25, 75)) / rmsecv

            #predictor
            predictor = pls

            # Checks if datapred and ypred were provided to calculate the prediction metrics
            if datapred is not None and ypred is not None:
                # Makes prediction for the prediction dataset
                y_pred = pls.predict(datapred).flatten()
                
                # Stores predicted values ​​in predres
                predres[f'LV_{n_comp}'] = y_pred
                
                # Calculates prediction metrics
                R2_pred = r2_score(ypred, y_pred)
                r2_pred = (np.corrcoef(ypred, y_pred)[0, 1]) ** 2
                rmsep = np.sqrt(mean_squared_error(ypred, y_pred))
                bias_pred = sum(ypred - y_pred)/ypred.shape[0]
                SDV_pred = (ypred - y_pred) - bias_pred
                SDV_pred = SDV_pred*SDV_pred
                SDV_pred = np.sqrt(sum(SDV_pred)/(ypred.shape[0] - 1))
                tbias_pred = abs(bias_pred)*(np.sqrt(ypred.shape[0])/SDV_pred)
                rpd_pred = ypred.std() / rmsep
                rpiq_pred = iqr(ypred, rng=(25, 75)) / rmsep
            else:
                # Sets the prediction metric values ​​to None if datapred or ypred are not provided
                R2_pred = r2_pred = rmsep = rpd_pred = rpiq_pred = bias_pred = SDV_pred = tbias_pred = None

            results.append({
                'LVs number': n_comp,
                'R2 Cal': R2_cal,
                'r2 Cal': r2_cal,
                'RMSEC': rmse_cal,
                'R2 CV': R2_cv,
                'r2 CV': r2_cv,
                'RMSECV': rmsecv,
                'Bias CV': bias_cv,
                'tbias CV': tbias_cv,
                'RPD CV': rpd_cv,
                'RPIQ CV': rpiq_cv,
                'R2 Pred': R2_pred,
                'r2 Pred': r2_pred,
                'RMSEP': rmsep,
                'Bias Pred': bias_pred,
                'tbias Pred': tbias_pred,
                'RPD Pred': rpd_pred,
                'RPIQ Pred': rpiq_pred
            })    
            
    elif model == 'rf':
        if random_seed is None:
            print("For the 'rf' model, enter the desired seed number in random_seed")
            return None, None, None, None, None
        kern = None
        # Fit a Random Forest model with default hyperparameters
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cross_decomposition import PLSRegression
        rf = RandomForestRegressor(criterion='squared_error',
                                   random_state=random_seed)
        
        #Calibration
        rf.fit(datacal, ycal)
        ycal_res = rf.predict(datacal)
        calres['RF'] = ycal_res

        # Calculates calibration metrics
        R2_cal = r2_score(ycal, ycal_res)
        r2_cal = (np.corrcoef(ycal, ycal_res)[0, 1]) ** 2
        rmse_cal = np.sqrt(mean_squared_error(ycal, ycal_res))

        #predictor
        predictor = rf

        # Checks if datapred and ypred were provided to calculate the prediction metrics
        if datapred is not None and ypred is not None:
            ypred_res = rf.predict(datapred)
            predres['RF'] = ypred_res

            # Calculates prediction metrics
            R2_pred = r2_score(ypred, ypred_res)
            r2_pred = (np.corrcoef(ypred, ypred_res)[0, 1]) ** 2
            rmse_pred = np.sqrt(mean_squared_error(ypred, ypred_res))
            bias_pred = sum(ypred - ypred_res)/ypred.shape[0]
            SDV_pred = (ypred - ypred_res) - bias_pred
            SDV_pred = SDV_pred*SDV_pred
            SDV_pred = np.sqrt(sum(SDV_pred)/(ypred.shape[0] - 1))
            tbias_pred = abs(bias_pred)*(np.sqrt(ypred.shape[0])/SDV_pred)
            rpd_pred = ypred.std() / rmse_pred
            rpiq_pred = iqr(ypred, rng=(25, 75)) / rmse_pred
        else: 
            R2_pred = r2_pred = rmse_pred = rpd_pred = rpiq_pred = bias_pred = SDV_pred = tbias_pred = None
    
        # Stores the results in a dictionary
        results.append({
            'Model': 'RF',
            'R2 Cal': R2_cal,
            'r2 Cal': r2_cal,
            'RMSEC': rmse_cal,
            'R2 Pred': R2_pred,
            'r2 Pred': r2_pred,
            'RMSEP': rmse_pred,
            'Bias Pred': bias_pred,
            'tbias Pred': tbias_pred,
            'RPD Pred': rpd_pred,
            'RPIQ Pred': rpiq_pred
        })
        
    elif model == 'svm':
        if kern is None:
            print("For the 'svm' model choose a kernel among 'linear', 'poly', 'rbf', 'sigmoid' using the kern argument")
            return None, None, None, None, None
        maxLV = None 
        random_seed = None   
        from sklearn.svm import SVR
        from sklearn.cross_decomposition import PLSRegression
        svm = SVR(kernel=kern)
        svm.fit(datacal, ycal)

        #Calibration
        ycal_res = svm.predict(datacal)
        calres['SVM'] = ycal_res

        # Calculates calibration metrics
        R2_cal = r2_score(ycal, ycal_res)
        r2_cal = (np.corrcoef(ycal, ycal_res)[0, 1]) ** 2
        rmse_cal = np.sqrt(mean_squared_error(ycal, ycal_res))

        #predictor
        predictor = svm

        # Checks if datapred and ypred were provided to calculate the prediction metrics
        if datapred is not None and ypred is not None:
            ypred_res = svm.predict(datapred)
            predres['SVM'] = ypred_res

            # Calculates prediction metrics
            R2_pred = r2_score(ypred, ypred_res)
            r2_pred = (np.corrcoef(ypred, ypred_res)[0, 1]) ** 2
            rmse_pred = np.sqrt(mean_squared_error(ypred, ypred_res))
            bias_pred = sum(ypred - ypred_res)/ypred.shape[0]
            SDV_pred = (ypred - ypred_res) - bias_pred
            SDV_pred = SDV_pred*SDV_pred
            SDV_pred = np.sqrt(sum(SDV_pred)/(ypred.shape[0] - 1))
            tbias_pred = abs(bias_pred)*(np.sqrt(ypred.shape[0])/SDV_pred)
            rpd_pred = ypred.std() / rmse_pred
            rpiq_pred = iqr(ypred, rng=(25, 75)) / rmse_pred
        else: 
            R2_pred = r2_pred = rmse_pred = rpd_pred = rpiq_pred = bias_pred = SDV_pred = tbias_pred = None
                
        # Stores the results in a dictionary
        results.append({
            'Model': 'SVM',
            'R2 Cal': R2_cal,
            'r2 Cal': r2_cal,
            'RMSEC': rmse_cal,
            'R2 Pred': R2_pred,
            'r2 Pred': r2_pred,
            'RMSEP': rmse_pred,
            'Bias Pred': bias_pred,
            'tbias Pred': tbias_pred,
            'RPD Pred': rpd_pred,
            'RPIQ Pred': rpiq_pred
        })

    elif model == 'mlp':
        if nnlayers and actf  and random_seed is None:
            print("For the 'mlp' model choose a random_seed, a neural network layer properly and a activation function among ('relu', 'identify', 'tanh', 'sigmoid') using the nnlayers and actf arguments")
            return None, None, None, None, None
        maxLV = None 
        kern=None   
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=nnlayers, activation=actf, random_state=random_seed, max_iter=1000)
        mlp.fit(datacal, ycal)

        #Calibration
        ycal_res = mlp.predict(datacal)
        calres['MLP'] = ycal_res

        # Calculates calibration metrics
        R2_cal = r2_score(ycal, ycal_res)
        r2_cal = (np.corrcoef(ycal, ycal_res)[0, 1]) ** 2
        rmse_cal = np.sqrt(mean_squared_error(ycal, ycal_res))

        #predictor
        predictor = mlp

        # Checks if datapred and ypred were provided to calculate the prediction metrics
        if datapred is not None and ypred is not None:
            ypred_res = mlp.predict(datapred)
            predres['MLP'] = ypred_res

            # Calculates prediction metrics
            R2_pred = r2_score(ypred, ypred_res)
            r2_pred = (np.corrcoef(ypred, ypred_res)[0, 1]) ** 2
            rmse_pred = np.sqrt(mean_squared_error(ypred, ypred_res))
            bias_pred = sum(ypred - ypred_res)/ypred.shape[0]
            SDV_pred = (ypred - ypred_res) - bias_pred
            SDV_pred = SDV_pred*SDV_pred
            SDV_pred = np.sqrt(sum(SDV_pred)/(ypred.shape[0] - 1))
            tbias_pred = abs(bias_pred)*(np.sqrt(ypred.shape[0])/SDV_pred)
            rpd_pred = ypred.std() / rmse_pred
            rpiq_pred = iqr(ypred, rng=(25, 75)) / rmse_pred
        else: 
            R2_pred = r2_pred = rmse_pred = rpd_pred = rpiq_pred = bias_pred = SDV_pred = tbias_pred = None
                
        # Stores the results in a dictionary
        results.append({
            'Model': 'MLP',
            'R2 Cal': R2_cal,
            'r2 Cal': r2_cal,
            'RMSEC': rmse_cal,
            'R2 Pred': R2_pred,
            'r2 Pred': r2_pred,
            'RMSEP': rmse_pred,
            'Bias Pred': bias_pred,
            'tbias Pred': tbias_pred,
            'RPD Pred': rpd_pred,
            'RPIQ Pred': rpiq_pred
        })

    # Converts the list of results to a DataFrame
    df_results = pd.DataFrame(results)
    calres.insert(0, 'Ref', ycal)
    if predres is not None:
        predres.insert(0, 'Ref', ypred)

    # Independent extraction of LV scores, if requested and if maxLV is provided
    if LVscores is not None and maxLV is not None:
        from sklearn.cross_decomposition import PLSRegression
        
        pls_scores = PLSRegression(n_components=maxLV, scale=False)
        pls_scores.fit(datacal, ycal)
        lv_scorescal = pd.DataFrame(pls_scores.transform(datacal),
                                    columns=[f'LV_{i+1}' for i in range(maxLV)])
        if datapred is not None and ypred is not None:
            lv_scorespred = pd.DataFrame(pls_scores.transform(datapred),
                                         columns=[f'LV_{i+1}' for i in range(maxLV)])
        else:
            lv_scorespred = None
    else:
        lv_scorescal = None
        lv_scorespred = None

    # Conditional return: if LVscores is True, also return the LV scores dataFrames
    # Otherwise, return only the results and the calibration and prediction dataFrames.
    if LVscores:
        return predictor, df_results, calres, lv_scorescal, predres, lv_scorespred
    else:
        return predictor, df_results, calres, predres

def all_modeling(datacal, ycal, target, datapred, ypred, maxLV, kern, random_seed, nnlayers, actf):
    """
    Applies the individual_optimized_model function to each of the four models ('pls', 'rf', 'svm', 'mlp'),
    returns a dictionary with df_results for each model.

    If datapred and ypred are not provided, perform only calibration (no prediction).
    **Parameters**:
    - datacal, ycal, target (np.array or pd.DF): calibration data and target name
    - datapred, ypred (np.array or pd.DF): prediction data (optional)
    - target (str): name of the column of the parameter to be predicted (present in ycal and ypred)
    - maxLV (int): max number of LVs (needed for 'pls')
    - kern (str): kernel for SVM (required for 'svm')
    - random_seed (int): seed for RF and MLP
    - nnlayers, actf (list, str): MLP configuration

    **Retorno**:
    - dict: {
        'pls': df_results_pls,
        'rf': df_results_rf,
        'svm': df_results_svm,
        'mlp': df_results_mlp
      }
    """
    if maxLV is None or kern is None or random_seed is None or nnlayers is None or actf is None:
            print("Please, provide all required hyperparameters: maxLV, kern, random_seed, nnlayers and actf.")
            return None, None, None, None, None
   
    modelos = ['pls', 'rf', 'svm', 'mlp']
    resultados = {}
    for m in modelos:
        # Perform modeling without extracting latent variables
        print(f"Running the model: {m}")

        pred = datapred is not None and ypred is not None
        
        out = optimized_individual_modeling(
            datacal, ycal, target,
            model=m,
            datapred=datapred if pred else None,
            ypred=ypred if pred else None,
            maxLV=maxLV, 
            kern=kern,
            random_seed=random_seed,
            nnlayers=nnlayers, 
            actf=actf,
            LVscores=False
            )
        # Standard output: (df_results, calres, predres)
        df_results = out[1]
        resultados[m] = df_results
    return resultados

# function to generate high-level fusion models with all possible combinations of predictors
def high_level_fusion_automated(datacal, ycal, target, datapred=None, ypred=None):
    """
    ## High-level data fusion. 
    **Generates multiple linear regression models for all possible combinations of inputs.**

    Parameters:
    - **datacal_dict**: dictionary of DataFrames with the calibration data.
    - **datapred_dict**: dictionary of DataFrames with the prediction data.
    - **Ycal**: DataFrame containing the actual values ​​for calibration.
    - **Ypred**: DataFrame containing the actual values ​​for prediction.
    - **target**: string representing the target variable.

    **Retorna**:
    - results: dictionary containing the models, coefficients, predictions and metrics for each possible combination.
    """

    import itertools
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr
    import numpy as np
    import pandas as pd
    results = {}

    # Number of columns desired per combination (from 2 to all)
    for i in range(2, len(datacal) + 1):
        for combinacoes in itertools.combinations(datacal.keys(), i):
            # itertools.combinations(datacal_dict.keys(), r): Generates all possible combinations of size r using the keys of the dictionary
            # Create DataFrame with only the last columns of the chosen combinations
            high_level_cal = pd.DataFrame({key: datacal[key].iloc[:, -1] for key in combinacoes})
            high_level_cal.columns = high_level_cal.columns.astype(str)
            if datapred is not None and ypred is not None:
                high_level_pred = pd.DataFrame({key: datapred[key].iloc[:, -1] for key in combinacoes}) 
                high_level_pred.columns = high_level_pred.columns.astype(str)
            else:
                high_level_pred = None

            # Set actual values ​​for calibration and prediction
            ytrain = ycal[target].values  # Convert to NumPy array to avoid warnings
            ytest = ypred[target].values if (datapred is not None and ypred is not None) else None

            # Create and train the multiple linear regression model
            model = LinearRegression()
            model.fit(high_level_cal, ytrain)

            # Predictions in calibration and testing
            ycal_res = model.predict(high_level_cal)
            ypred_res = model.predict(high_level_pred) if (datapred is not None and ypred is not None) else None

            # Calculation of metrics for calibration
            R2_cal = r2_score(ytrain, ycal_res)
            r2_cal = (np.corrcoef(ytrain, ycal_res)[0, 1]) ** 2
            rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_res))

            if (datapred is not None and ypred is not None):
                # Calculation of metrics for prediction
                R2_pred = r2_score(ytest, ypred_res)
                r2_pred = (np.corrcoef(ytest, ypred_res)[0, 1]) ** 2
                rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_res))
                bias_pred = sum(ytest - ypred_res)/ytest.shape[0]
                SDV_pred = (ytest - ypred_res) - bias_pred
                SDV_pred = SDV_pred*SDV_pred
                SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1)) 
                tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)
                rpd = ytest.std() / rmse_pred
                rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred
            else:
                R2_pred = None
                r2_pred = None
                rmse_pred = None
                bias_pred = None
                tbias_pred = None
                rpd = None
                rpiq = None

            if (datapred is not None and ypred is not None):
                # Create DataFrame with predictions
                predicoes = pd.DataFrame({
                    'Results': np.concatenate((ycal_res, ypred_res)),  # Predicted values
                    'Ref': np.concatenate((ytrain, ytest)),  # Combined real values
                    'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)  # Set indication
                })

                combinacoes_name = "_".join(combinacoes)
                # Joins the names of selected columns by combining with an underscore
                            
                # Store results in the results dictionary
                results[combinacoes_name] = {
                    'model': model,
                    'coefficients': model.coef_,
                    'intercept': model.intercept_,
                    'predictions': predicoes,  
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                        'R2 Pred': R2_pred,
                        'r2_pred': r2_pred,
                        'RMSEP': rmse_pred,
                        'Bias Pred': bias_pred,
                        'tbias Pred': tbias_pred,
                        'RPD Pred': rpd,
                        'RPIQ Pred': rpiq
                    }
                }
            else: 
                 # Create DataFrame with predictions
                predicoes = pd.DataFrame({
                    'Results':ycal_res,  # Predicted values
                    'Ref': ytrain, # Combined real values
                    'Set': ['cal'] * len(ytrain)  # Set indication
                })

                combinacoes_name = "_".join(combinacoes)
                # Joins the names of selected columns by combining with an underscore
                            
                # Store results in the results dictionary
                results[combinacoes_name] = {
                    'model': model,
                    'coefficients': model.coef_,
                    'intercept': model.intercept_,
                    'predictions': predicoes,  
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                    }
                }   
    return results

def mid_level_fusion_automated(datacal, ycal, target, datapred=None, ypred=None):
    """
    ## Mid-level data fusion
    **Generates multiple linear regression models for all possible combinations of inputs.**

    **Parameters**:
    - **datacal**: dictionary of DataFrames with the calibration scores data.
    - **datapred**: dictionary of DataFrames with the prediction scores data.
    - **ycal**: DataFrame containing the actual values ​​for calibration.
    - **ypred**: DataFrame containing the actual values ​​for prediction.
    - **target**: string representing the target variable.

    **Retorna**: dictionary containing the models, coefficients, predictions and metrics for each possible combination.
    """
    import itertools
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr
    import numpy as np
    import pandas as pd

    results = {}

    # Number of columns desired per combination (from 2 to all)
    for i in range(2, len(datacal) + 1):
        for combinacoes in itertools.combinations(datacal.keys(), i):
            mid_level_cal = pd.concat([datacal[key] for key in combinacoes], axis=1)
            mid_level_cal.columns = mid_level_cal.columns.astype(str)

            if (datapred is not None and ypred is not None):
                mid_level_pred = pd.concat([datapred[key] for key in combinacoes], axis=1)
                mid_level_pred.columns = mid_level_pred.columns.astype(str)
            else:
                mid_level_pred = None    

            # Set actual values ​​for calibration and prediction
            ytrain = ycal[target].values  
            ytest = ypred[target].values if (datapred is not None and ypred is not None) else None

            # Create and train the multiple linear regression model
            model = LinearRegression()
            model.fit(mid_level_cal, ytrain)

            # Predictions in calibration and testing
            ycal_res = model.predict(mid_level_cal)
            ypred_res = model.predict(mid_level_pred) if (datapred is not None and ypred is not None) else None

            # Calculation of metrics for calibration
            R2_cal = r2_score(ytrain, ycal_res)
            r2_cal = (np.corrcoef(ytrain, ycal_res)[0, 1]) ** 2
            rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_res))

            # Calculation of metrics for prediction
            if (datapred is not None and ypred is not None):
                R2_pred = r2_score(ytest, ypred_res)
                r2_pred = (np.corrcoef(ytest, ypred_res)[0, 1]) ** 2
                rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_res))
                bias_pred = sum(ytest - ypred_res)/ytest.shape[0]
                SDV_pred = (ytest - ypred_res) - bias_pred
                SDV_pred = SDV_pred*SDV_pred
                SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1))
                tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)
                rpd = ytest.std() / rmse_pred
                rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred
            else:
                R2_pred = r2_pred = rmse_pred = rpd = rpiq = bias_pred = SDV_pred = tbias_pred = None
                
            if (datapred is not None and ypred is not None):
                # Create DataFrame with predictions
                predicoes = pd.DataFrame({
                    'Results': np.concatenate((ycal_res, ypred_res)),  
                    'Ref': np.concatenate((ytrain, ytest)), 
                    'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest) 
                })

                combinacoes_name = "_".join(combinacoes)
                
                results[combinacoes_name] = {
                    'model': model,
                    'coefficients': model.coef_,
                    'intercept': model.intercept_,
                    'predictions': predicoes, 
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                        'R2 Pred': R2_pred,
                        'r2_pred': r2_pred,
                        'RMSEP': rmse_pred,
                        'Bias Pred': bias_pred,
                        'tbias Pred': tbias_pred,
                        'RPD Pred': rpd,
                        'RPIQ Pred': rpiq
                    }
                }
            else:
                # Create DataFrame with predictions
                predicoes = pd.DataFrame({
                    'Results': ycal_res,  
                    'Ref': ytrain, 
                    'Set': ['cal'] * len(ytrain) 
                })

                combinacoes_name = "_".join(combinacoes)
                
                results[combinacoes_name] = {
                    'model': model,
                    'coefficients': model.coef_,
                    'intercept': model.intercept_,
                    'predictions': predicoes, 
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                    }
                }

    return results

def auto_scaling(input_data):
    """
    Auto scaling
    ----------
    Returns the preprocessed data, sd in the original space and mean in the original space
    """
    mean_original = np.mean(input_data, axis=0)
    sd_original = np.std(input_data, axis=0)
    X_sc = input_data / sd_original
    X_sc = X_sc - mean_original
    X_sc[np.isnan(X_sc)] = 0
    return pd.DataFrame(X_sc), sd_original, mean_original

def low_level_fusion_automated(datacal, ycal, target, datapred=None, ypred=None,  
                               scale=True, maxLV=None, model='pls', kern=None, 
                               random_seed=None, nnlayers=[100, 50, 20], actf='relu'):
    """
    ## Low-level data fusion
    **Generates models for all possible combinations of inputs (low level fusion) using PLS, Random Forest or SVM.**
    
    **Parâmetros**:
    - **datacal**: dictionary of DataFrames with the calibration spectra.
    - **datapred**: dictionary of DataFrames with the prediction spectra.
    - **ycal**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input
    - **ypred**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input (Optional)
    - **target**: string representing the target variable (according to the column names of Ycal/Ypred)
    - **scale**: Bol that says whether the data will be autoscaled before modeling or not
    - **model**: the model to be used, chosen among 'pls', 'rf', 'svm' and 'mlp'
    - **maxLV**: If model='pls', maximum number of LVs to be used
    - **kern**: If model='svm', the kernel must be chosen among 'linear', 'rbf', 'sigmoid' 
    - **random_seed**: Int. If model='rf', the random seed should be chosen
    - **nnlayers**: List of integers representing the number of neurons in each layer (if model='mlp')
    - **actf**: String representing the activation function to be used in the MLP model. Options: 'relu', 'identify', 'tanh', 'sigmoid'
    - **Note**: the pls model implements a 10-fold cross-validation
    
    **Return**: results: nested dictionary containing the models, predictions, metrics, all the important information
    """
    import itertools
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr

    model = model.lower()
    if model not in ['pls', 'rf', 'svm', 'mlp']:
        print("The 'model' parameter must be 'pls', 'rf', 'svm' or 'mlp'")

    results = {}

    # Iterates through all possible combinations (from 2 to all available sources)
    for i in range(2, len(datacal) + 1):
        for combinacoes in itertools.combinations(datacal.keys(), i):
            # Concatenate spectra for calibration and prediction
            low_level_cal = pd.concat([datacal[key] for key in combinacoes], axis=1)
            low_level_cal.columns = low_level_cal.columns.astype(str)

            if (datapred is not None and ypred is not None):
                low_level_pred = pd.concat([datapred[key] for key in combinacoes], axis=1)
                low_level_pred.columns = low_level_pred.columns.astype(str)
            else:
                low_level_pred = None    
            
            if scale:
                low_level_cal, sdcal, meancal = auto_scaling(low_level_cal) # autoscaling the data
                if (datapred is not None and ypred is not None):
                    low_level_pred = ( low_level_pred / sdcal ) - meancal
                    low_level_pred[np.isnan(low_level_pred)] = 0
                else:
                    low_level_pred = None    
            else:
                low_level_cal = low_level_cal
                if (datapred is not None and ypred is not None):    
                    low_level_pred = low_level_pred
                else:
                    low_level_pred = None

            combinacoes_name = "_".join(combinacoes)
            results[combinacoes_name] = {}
            ytrain = ycal[target].values
            ytest = ypred[target].values if (datapred is not None and ypred is not None) else None

            if model == 'pls':
                if maxLV == None:
                    print("For the 'pls' model, enter the desired number of latent variables in the maxLV argument.")
                    return None
                
                # For each number of latent variables from 1 to maxLV
                from sklearn.cross_decomposition import PLSRegression
                from sklearn.model_selection import cross_val_predict
                kern=None
                random_seed=None
                for maxLV in range(1, maxLV + 1):
                    pls = PLSRegression(n_components=maxLV, scale=False)
                    pls.fit(low_level_cal, ytrain)

                    # Pred
                    ycal_pred = pls.predict(low_level_cal).ravel()
                    # Cross-validation
                    y_cv = cross_val_predict(pls, low_level_cal, ytrain, cv=10)
                    ypred_pred = pls.predict(low_level_pred).ravel() if (datapred is not None and ypred is not None) else None

                    # Metrics for calibration
                    R2_cal = r2_score(ytrain, ycal_pred)
                    r2_cal = (np.corrcoef(ytrain, ycal_pred)[0, 1]) ** 2
                    rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_pred))
                    
                    # Cross-validation metrics
                    R2_cv = r2_score(ytrain, y_cv)
                    r2_cv = (np.corrcoef(ytrain, y_cv)[0, 1]) ** 2
                    rmsecv = np.sqrt(mean_squared_error(ytrain, y_cv))
                    bias_cv = sum(ytrain - y_cv)/ytrain.shape[0]
                    SDV_cv = (ytrain - y_cv) - bias_cv
                    SDV_cv = SDV_cv*SDV_cv
                    SDV_cv = np.sqrt(sum(SDV_cv)/(ytrain.shape[0] - 1))
                    tbias_cv = abs(bias_cv)*(np.sqrt(ytrain.shape[0])/SDV_cv)
                    rpd_cv = np.std(ytrain) / rmsecv if rmsecv != 0 else np.nan
                    rpiq_cv = iqr(ytrain, rng=(25, 75)) / rmsecv if rmsecv != 0 else np.nan

                    # Metrics for prediction
                    R2_pred = r2_score(ytest, ypred_pred) if (datapred is not None and ypred is not None) else None
                    r2_pred = (np.corrcoef(ytest, ypred_pred)[0, 1]) ** 2 if (datapred is not None and ypred is not None) else None
                    rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_pred)) if (datapred is not None and ypred is not None) else None
                    bias_pred = sum(ytest - ypred_pred)/ytest.shape[0] if (datapred is not None and ypred is not None) else None
                    SDV_pred = (ytest - ypred_pred) - bias_pred if (datapred is not None and ypred is not None) else None
                    SDV_pred = SDV_pred*SDV_pred if (datapred is not None and ypred is not None) else None
                    SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1)) if (datapred is not None and ypred is not None) else None
                    tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred) if (datapred is not None and ypred is not None) else None
                    if rmse_pred is not None:
                        rpd = np.std(ytest) / rmse_pred if rmse_pred != 0 else np.nan
                        rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred if rmse_pred != 0 else np.nan
                    else:
                        rpd = rpiq = None    

                    # Create DataFrame with predictions
                    if (datapred is not None and ypred is not None):
                        predicoes = pd.DataFrame({
                            'Results': np.concatenate((ycal_pred, ypred_pred)),
                            'Ref': np.concatenate((ytrain, ytest)),
                            'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)
                        })
                    else:
                        predicoes = pd.DataFrame({
                            'Results': ycal_pred,
                            'Ref': ytrain,
                            'Set': ['cal'] * len(ytrain)
                        })    

                    results[combinacoes_name][f"LV{maxLV}"] = {
                        'model': pls,
                        'coefficients': pls.coef_,
                        'predictions': predicoes,
                        'metrics': {
                            'R2 Cal': R2_cal,
                            'r2 Cal': r2_cal,
                            'RMSEC': rmse_cal,
                            'R2 CV': R2_cv,
                            'r2 CV': r2_cv,
                            'RMSECV': rmsecv,
                            'Bias CV': bias_cv,
                            'tbias CV': tbias_cv,
                            'RPD CV': rpd_cv,
                            'RPIQ CV': rpiq_cv,
                            'R2 Pred': R2_pred if (datapred is not None and ypred is not None) else None, 
                            'r2 Pred': r2_pred if (datapred is not None and ypred is not None) else None,
                            'RMSEP': rmse_pred if (datapred is not None and ypred is not None) else None,
                            'Bias Pred': bias_pred if (datapred is not None and ypred is not None) else None,
                            'tbias Pred': tbias_pred if (datapred is not None and ypred is not None) else None,
                            'RPD Pred': rpd if (datapred is not None and ypred is not None) else None,
                            'RPIQ Pred': rpiq if (datapred is not None and ypred is not None) else None
                        }
                    }
            elif model == 'rf':
                if random_seed == None:
                    print("For the 'rf' model, enter the desired seed number in random_seed")
                    return None
                maxLV=None
                kern=None
                # Fit a Random Forest model with default hyperparameters
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(random_state=random_seed)
                rf.fit(low_level_cal, ytrain)

                # Pred
                ycal_pred = rf.predict(low_level_cal)
                ypred_pred = rf.predict(low_level_pred) if (datapred is not None and ypred is not None) else None

                # Metrics for calibration
                R2_cal = r2_score(ytrain, ycal_pred)
                r2_cal = (np.corrcoef(ytrain, ycal_pred)[0, 1]) ** 2
                rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_pred))

                # Metrics for prediction
                R2_pred = r2_score(ytest, ypred_pred) if (datapred is not None and ypred is not None) else None
                r2_pred = (np.corrcoef(ytest, ypred_pred)[0, 1]) ** 2 if (datapred is not None and ypred is not None) else None
                bias_pred = sum(ytest - ypred_pred)/ytest.shape[0] if (datapred is not None and ypred is not None) else None
                SDV_pred = (ytest - ypred_pred) - bias_pred if (datapred is not None and ypred is not None) else None
                SDV_pred = SDV_pred*SDV_pred if (datapred is not None and ypred is not None) else None
                SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1)) if (datapred is not None and ypred is not None) else None
                tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred) if (datapred is not None and ypred is not None) else None
                rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_pred)) if (datapred is not None and ypred is not None) else None
                if rmse_pred is not None:
                        rpd = np.std(ytest) / rmse_pred if rmse_pred != 0 else np.nan
                        rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred if rmse_pred != 0 else np.nan
                else:
                        rpd = rpiq = None    
                # Create DataFrame with predictions
                if (datapred is not None and ypred is not None):
                    predicoes = pd.DataFrame({
                        'Results': np.concatenate((ycal_pred, ypred_pred)),
                        'Ref': np.concatenate((ytrain, ytest)),
                        'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)
                    })
                else:
                    predicoes = pd.DataFrame({
                        'Results': ycal_pred,
                        'Ref': ytrain,
                        'Set': ['cal'] * len(ytrain)
                    })

                # For Random Forest there are no explicit coefficients or intercepts
                results[combinacoes_name]["RF"] = {
                    'model': rf,
                    'predictions': predicoes,
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                        'R2 Pred': R2_pred if (datapred is not None and ypred is not None) else None,
                        'r2 Pred': r2_pred if (datapred is not None and ypred is not None) else None,
                        'RMSEP': rmse_pred if (datapred is not None and ypred is not None) else None,
                        'Bias Pred': bias_pred if (datapred is not None and ypred is not None) else None,
                        'tbias Pred': tbias_pred if (datapred is not None and ypred is not None) else None,
                        'RPD Pred': rpd if (datapred is not None and ypred is not None) else None,
                        'RPIQ Pred': rpiq if (datapred is not None and ypred is not None) else None
                    }
                }
            elif model == 'svm':
                if kern == None:
                    print("For the 'svm' model choose a kernel among 'linear', 'poly', 'rbf', 'sigmoid' using the kern argument")
                    return None
                maxLV=None 
                random_seed=None   
                from sklearn.svm import SVR
                svm = SVR(kernel=kern)
                svm.fit(low_level_cal, ytrain)

                # Pred
                ycal_pred = svm.predict(low_level_cal)
                ypred_pred = svm.predict(low_level_pred)

                # Metrics for calibration
                R2_cal = r2_score(ytrain, ycal_pred)
                r2_cal = (np.corrcoef(ytrain, ycal_pred)[0, 1]) ** 2
                rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_pred))

                # Metrics for prediction
                R2_pred = r2_score(ytest, ypred_pred) if (datapred is not None and ypred is not None) else None,
                r2_pred = (np.corrcoef(ytest, ypred_pred)[0, 1]) ** 2 if (datapred is not None and ypred is not None) else None,
                rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_pred)) if (datapred is not None and ypred is not None) else None,
                bias_pred = sum(ytest - ypred_pred)/ytest.shape[0] if (datapred is not None and ypred is not None) else None,
                SDV_pred = (ytest - ypred_pred) - bias_pred if (datapred is not None and ypred is not None) else None,
                SDV_pred = SDV_pred*SDV_pred if (datapred is not None and ypred is not None) else None,
                SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1)) if (datapred is not None and ypred is not None) else None,
                tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred) if (datapred is not None and ypred is not None) else None,
                if rmse_pred is not None:
                        rpd = np.std(ytest) / rmse_pred if rmse_pred != 0 else np.nan
                        rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred if rmse_pred != 0 else np.nan
                else:
                        rpd = rpiq = None    

                # Create DataFrame with predictions
                predicoes = pd.DataFrame({
                    'Results': np.concatenate((ycal_pred, ypred_pred)),
                    'Ref': np.concatenate((ytrain, ytest)),
                    'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)
                })

                # For SVM there are no explicit coefficients or intercepts
                results[combinacoes_name]["SVM"] = {
                    'model': svm,
                    'predictions': predicoes,
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                        'R2 Pred': R2_pred if (datapred is not None and ypred is not None) else None,
                        'r2 Pred': r2_pred if (datapred is not None and ypred is not None) else None,
                        'RMSEP': rmse_pred if (datapred is not None and ypred is not None) else None,
                        'Bias Pred': bias_pred if (datapred is not None and ypred is not None) else None,
                        'tbias Pred': tbias_pred if (datapred is not None and ypred is not None) else None,
                        'RPD Pred': rpd if (datapred is not None and ypred is not None) else None,
                        'RPIQ Pred': rpiq if (datapred is not None and ypred is not None) else None,
                    }
                }

            elif model == 'mlp':
                if nnlayers and actf  is None:
                    print("For the 'mlp' model choose a random_seed, a neural network layer properly and a activation function among ('relu', 'identify', 'tanh', 'sigmoid') using the nnlayers and actf arguments")
                    return None, None, None, None, None
                maxLV=None
                kern=None 
                random_seed=None     
                from sklearn.neural_network import MLPRegressor
                mlp = MLPRegressor(hidden_layer_sizes=nnlayers, activation=actf, random_state=random_seed, max_iter=1000)
                mlp.fit(low_level_cal, ytrain)

                # Pred
                ycal_pred = mlp.predict(low_level_cal)
                ypred_pred = mlp.predict(low_level_pred)

                # Metrics for calibration
                R2_cal = r2_score(ytrain, ycal_pred)
                r2_cal = (np.corrcoef(ytrain, ycal_pred)[0, 1]) ** 2
                rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_pred))

                # Metrics for prediction
                if (datapred is not None and ypred is not None):
                    R2_pred = r2_score(ytest, ypred_pred) 
                    r2_pred = (np.corrcoef(ytest, ypred_pred)[0, 1]) ** 2
                    rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_pred))
                    bias_pred = sum(ytest - ypred_pred)/ytest.shape[0]
                    SDV_pred = (ytest - ypred_pred) - bias_pred
                    SDV_pred = SDV_pred*SDV_pred
                    SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1))
                    tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)
                    rpd = np.std(ytest) / rmse_pred if rmse_pred != 0 else np.nan
                    rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred if rmse_pred != 0 else np.nan
                else:
                    R2_pred = r2_pred = rmse_pred = rpd = rpiq = bias_pred = SDV_pred = tbias_pred = None

                # Create DataFrame with predictions
                if (datapred is not None and ypred is not None):
                    predicoes = pd.DataFrame({
                        'Results': np.concatenate((ycal_pred, ypred_pred)),
                        'Ref': np.concatenate((ytrain, ytest)),
                        'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)
                    })

                    # For SVM there are no explicit coefficients or intercepts
                    results[combinacoes_name]["MLP"] = {
                        'model': mlp,
                        'predictions': predicoes,
                        'metrics': {
                            'R2 Cal': R2_cal,
                            'r2 Cal': r2_cal,
                            'RMSEC': rmse_cal,
                            'R2 Pred': R2_pred,
                            'r2 Pred': r2_pred,
                            'RMSEP': rmse_pred,
                            'Bias Pred': bias_pred,
                            'tbias Pred': tbias_pred,
                            'RPD Pred': rpd,
                            'RPIQ Pred': rpiq
                        }
                    }
                else:
                    predicoes = pd.DataFrame({
                        'Results': ycal_pred,
                        'Ref': ytrain,
                        'Set': ['cal'] * len(ytrain)
                    })

                    # For SVM there are no explicit coefficients or intercepts
                    results[combinacoes_name]["MLP"] = {
                        'model': mlp,
                        'predictions': predicoes,
                        'metrics': {
                            'R2 Cal': R2_cal,
                            'r2 Cal': r2_cal,
                            'RMSEC': rmse_cal,
                        }
                    }    
    return results