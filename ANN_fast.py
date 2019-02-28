import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
import mlrose
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight

path = 'C:/Users/John/Dropbox/Dalton State College/georgia_tech/CSE7641/supervisor_learning/'
#path = 'C:/Users/jasplund/Dropbox/Dalton State College/georgia_tech/CSE7641/supervisor_learning/'
wine = pd.read_csv(path+'winequality-white.csv',sep=';')
wine.loc[:, wine.columns != 'quality'] = preprocessing.scale(wine.loc[:, wine.columns != 'quality'])

wine['quality'].loc[wine['quality'].isin([1,2,3,4,5,6])] = 1
wine['quality'].loc[wine['quality'].isin([7,8,9,10])] = 0

#wine_trnX, wine_tstX, wine_trnY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
#
#
#pipeW = Pipeline([('Scale',StandardScaler()),
#                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])
np.random.seed(55)
msk = np.random.rand(len(wine)) < 0.7
msk 

wine_train = wine.iloc[msk]

wine_test = wine[~msk]

wine_d = wine_train.shape[1]
hiddens_wine = [[h]*l for l in [1,2,3] for h in [wine_d,wine_d//2,wine_d*2]]
hiddens_wine 
alphas = [10**-x for x in np.arange(-2,4.01,1)]
#alphas = alphas[2]
alphas  
params_wine = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_wine}
hiddens_wine


def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)

scorer = make_scorer(balanced_accuracy)    

from sklearn.model_selection import KFold

np.random.seed(55)
kf = KFold(n_splits=5,shuffle=True)
splt = [[train,validate] for train,validate in kf.split(wine_train)]
train,validate = splt[0]
#alpha=0.001
#activation = 'relu'
#algorithm = 'random_hill_climb'


#algs = ['random_hill_climb','simulated_annealing','genetic_alg','mimic']
algs = ['genetic_alg']
activation_functions = ['relu', 'sigmoid']
mean_fold_accuracy_train = []
mean_fold_accuracy_val = []
alg_time = []

def model(w_train,
          layer,
          activation,
          algorithm,
          alpha,
          train_acc,
          val_acc,
          train,
          validation):
    np.random.seed(55)
    train_data = w_train.iloc[train].copy()
    val_data = w_train.iloc[validate].copy()
    train_dataX = train_data.loc[:,train_data.columns!='quality'].copy()
    train_dataY = train_data['quality'].copy()
    val_dataX = val_data.loc[:,val_data.columns!='quality'].copy()
    val_dataY = val_data['quality'].copy()
    nn_model1 = mlrose.NeuralNetwork(layer,activation=activation,algorithm=algorithm,max_iters=1000,
                                 bias=False,is_classifier=True,learning_rate=alpha)
    nn_model1.fit(train_dataX,train_dataY)
    y_train_pred = nn_model1.predict(train_dataX)
    y_train_accuracy = accuracy_score(train_dataY,y_train_pred)
    train_acc.append(y_train_accuracy)
    print('Training accuracy is ',y_train_accuracy, ' with learning rate ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    y_val_pred = nn_model1.predict(val_dataX)
    y_val_accuracy = accuracy_score(val_dataY,y_val_pred)
    print('Testing accuracy: ',y_val_accuracy, ' with learning rate ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    val_acc.append(y_val_accuracy)
    return train_acc,val_acc

#model(wine_train,[6],'relu','random_hill_climb',0.01,[],[],train,validate) 

from joblib import Parallel, delayed

#start = time.time()
#out = Parallel(n_jobs=3, prefer="threads")(delayed(model)(wine_train,[6],'relu','random_hill_climb',0.01,[],[],train,validate)  for train,validate in kf.split(wine_train))
#temp_train_acc = [x[0][0] for x in out]
#temp_val_acc = [x[1][0] for x in out]
##for train, validate in kf.split(wine_train):
##    model(wine_train,[6],'relu','random_hill_climb',0.01,[],[],train,validate)
#end = time.time()
#print(end-start)

for algorithm in algs:
    start = time.time()
    for activation in activation_functions:
        for alpha in alphas:
            for layer in hiddens_wine:
                temp_train_acc = []
                temp_val_acc = []
                out = Parallel(n_jobs=2, prefer="threads")(delayed(model)(wine_train,
                         layer,activation,algorithm,alpha,[],[],train,
                         validate)  for train,validate in kf.split(wine_train))
                temp_train_acc = [x[0][0] for x in out]
                temp_val_acc = [x[1][0] for x in out]
                mean_fold_accuracy_train.append([np.mean(temp_train_acc),alpha,layer,activation,algorithm])    
                mean_fold_accuracy_val.append(np.mean(temp_val_acc))
    end = time.time()
    alg_time.append([algorithm,end-start])

start = time.time()
for activation in activation_functions:
    for alpha in alphas:
        for layer in hiddens_wine:
            temp_train_acc = []
            temp_val_acc = []
            out = Parallel(n_jobs=2, prefer="threads")(delayed(model)(wine_train,
                     layer,activation,'gradient_descent',alpha,[],[],train,
                     validate)  for train,validate in kf.split(wine_train))
            temp_train_acc = [x[0][0] for x in out]
            temp_val_acc = [x[1][0] for x in out]
            mean_fold_accuracy_train.append([np.mean(temp_train_acc),alpha,layer,activation,algorithm])    
            mean_fold_accuracy_val.append(np.mean(temp_val_acc))
end = time.time()
alg_time.append([algorithm,end-start])


df = pd.concat([pd.DataFrame(mean_fold_accuracy_train),
                pd.DataFrame(mean_fold_accuracy_val)], join='inner',axis=1)
df.columns = ['mean_train_score','param_MLP__alpha',
              'param_MLP__hidden_layer_sizes',
              'param_MLP__activation','algorithm','mean_test_score']
pd.DataFrame(alg_time).to_csv('./output/alg_times_fast.csv')
rhc = df[df['algorithm']=='random_hill_climb']
rhc.to_csv('./output/random_hill_climb_wine_main_fast.csv')  
sa = df[df['algorithm']=='simulated_annealing']
sa.to_csv('./output/simulated_annealing_wine_main_fast.csv')  
ga = df[df['algorithm']=='genetic_alg']
ga.to_csv('./output/genetic_alg_wine_main_fast.csv')  
gd = df[df['algorithm']=='gradient_descent']
gd.to_csv('./output/gradient_descent_wine_main_fast.csv')  

cv = ms.GridSearchCV(pipeW,n_jobs=4,param_grid=params_wine,refit=True,verbose=10,cv=5,scoring=scorer)
cv.fit(wine_trnX,wine_trnY)
regTable = pd.DataFrame(cv.cv_results_)
regTable.columns
test_score = cv.score(wine_tstX,wine_tstY)
N = wine_trnY.shape[0]

#set best parameters to the values in this new pipe and then run learning_curve
#new_pipe = Pipeline(memory=None,
#     steps=[('Scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('MLP', MLPClassifier(activation='relu', alpha=1.0, batch_size='auto', beta_1=0.9,
#       beta_2=0.999, early_stopping=True, epsilon=1e-08,
#       hidden_layer_sizes=[5], learning_rate='constant',
#       learning_rate_init=0.001,       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False))])    
curve = ms.learning_curve(cv.best_estimator_,wine_trnX,wine_trnY,cv=5,train_sizes=[50,100]+[int(N*x/10) for x in range(1,8)],verbose=10,scoring=scorer)




#




nn_model1 = mlrose.NeuralNetwork(hiddens_wine[0],activation='relu',algorithm='random_hill_climb',max_iters=1000,
                                 bias=False,is_classifier=True,learning_rate=0.001)
nn_model1.fit(wine_trnX,wine_trnY)
y_train_pred = nn_model1.predict(wine_trnX)
y_train_accuracy = accuracy_score(wine_trnY,y_train_pred)
print(y_train_accuracy)

scores = cross_val_score(nn_model1, wine_trnX, wine_trnY, cv=5)


start = time.time()
wine_clf = basicResults(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,params_wine,'ANN','wine')        



wine_final_params =wine_clf.best_params_
wine_OF_params =wine_final_params.copy()
#wine_OF_params['MLP__alpha'] = 0
pipeW.set_params(**wine_final_params)
pipeW.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(wineX,wineY,pipeW,'ANN','wine')
pipeW.set_params(**wine_final_params)
pipeW.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN','wine')                
pipeW.set_params(**wine_OF_params)
pipeW.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeW,wine_trnX,wine_trnY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900]},'ANN_OF','wine')                
end = time.time()
print(end-start)