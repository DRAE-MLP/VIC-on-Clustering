# Import basic libraries for data processing
import pandas as pd
import numpy as np
import time
# Import libraries for ML
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Import libraries for parallel processing
import multiprocessing as mp

def vic_multi_class_auc(df, classifier, scaler, cv):
    '''
    The function gets as an input the dataframe, the classifier, the scaler and the cross-validation
    method. It manually implements the cross-validation for multiclass and return a tuple with the mean AUC of
    each fold with OneVsRestClassifier with its corresponding column name of the target variable.
    '''
    # Initialize local variables
    auc_scores = []
    results = []
    # define X and y
    y_ = df.iloc[:,-1].values
    X = df.iloc[:,:70].values
    # Get the number of clases of the target attribute
    classes = df.iloc[:,-1].unique()
    # Binarize the output for multi classes purposes
    y = label_binarize(y_, classes = classes)
    # Begin the cross-validation method
    for train_index, test_index in cv.split(X):
        # Split train-test sets for each fold
        X_train_, X_test_ = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Scale data fit only on train and transform on both
        X_train = scaler.fit_transform(X_train_)
        X_test = scaler.transform(X_test_)
        # Compute the score depending the number of classes
        if len(classes) == 1:
            print('Error: The number of classes is equal to 1')
            break
        elif len(classes) == 2:
            # Predict probability only for two classes
            y_pred = classifier.fit(X_train, y_train).predict_proba(X_test)[:,1]
            auc_scores = roc_auc_score(y_test, y_pred)
        else:
            # Predict the class only for 3 or more classes
            y_pred = classifier.fit(X_train,y_train).predict(X_test)
            auc_scores = [roc_auc_score(y_test[:,i], y_pred[:,i]) for i in range(len(classes))]
        # Add the average of the score to the results
        results.append(np.mean(auc_scores))
    # Print the column that has been processed for debugging purposes
    print('Finished '+df.columns[-1]+' ...')

    return (df.columns[-1], results)

def collect_result(result):
    '''
    This function is for the collector method of parellelization of the asynchronus method apply
    '''
    global scores
    scores.append(result)
 
if __name__ == '__main__':
    # Import the dataset
    df = pd.read_csv('df_cluster.csv')
    # Define the model, scaler and fold method. IMPORTANT: the methods should not use n_jobs=-1
    # because the parallelization will not work. Evaluate whose faster.
    model = OneVsRestClassifier(GradientBoostingClassifier())
    scaler = StandardScaler()
    cv = KFold(n_splits = 10, shuffle=True, random_state=10)
    # Initiate global varibales
    scores = []
    rang = range(70,106)
    data_to_export = pd.DataFrame()
    # Start the counter of time
    st = time.time()
    # Initialize the pool class with the number of required CPU's
    pool = mp.Pool(mp.cpu_count())

    # Without Parallelization
    #scores = [vic_multi_class_auc(df.iloc[:,np.r_[0:70,element]], model, scaler, cv) for element in rang]

    # Just uncomment the desired method of parellelization

    # Syncronus Parallelization will use only one node. 
    # Apply method 
    #scores = [pool.apply(vic_multi_class_auc, args=(df.iloc[:,np.r_[0:70,element]], model, scaler, cv)) for element in rang]
    # StarMap method
    #scores = pool.starmap(vic_multi_class_auc, [(df.iloc[:,np.r_[0:70,element]], model, scaler, cv) for element in rang])
    
    # Asyncronus
    # Apply method
    '''
    for element in rang:
        pool.apply_async(vic_multi_class_auc, args=(df.iloc[:,np.r_[0:70,element]], model, scaler, cv), callback=collect_result)
    pool.close()
    pool.join()
    '''
    # StarMap method
    '''
    scores = pool.starmap_async(vic_multi_class_auc, [(df.iloc[:,np.r_[0:70,element]], model, scaler, cv) for element in rang]).get()
    pool.close()
    '''

    # Finish the counter of time
    end = time.time()
    # Print the needed time to compute
    print('Time: '+str(round(end-st,2))+' seconds.')
    # Fill the dataframe to export
    for element in range(len(scores)):
        data_to_export[scores[element][0]] = scores[element][1]
    # Export to csv
    print(data_to_export)
    #data_to_export.to_csv('auc_gradientboostingC_clustering.csv')