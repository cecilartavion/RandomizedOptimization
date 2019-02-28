import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import sklearn.model_selection as ms
from statsmodels import robust
import statistics as stats
import matplotlib.pyplot as plt
from sklearn import preprocessing


#path = 'C:/Users/John/Dropbox/Dalton State College/georgia_tech/CSE7641/supervisor_learning/'
path = 'C:/Users/jasplund/Dropbox/Dalton State College/georgia_tech/CSE7641/randomized_optimization/'
wine = pd.read_csv(path+'winequality-white.csv',sep=';')
wine.loc[:, wine.columns != 'quality'] = preprocessing.scale(wine.loc[:, wine.columns != 'quality'])

wine['quality'].loc[wine['quality'].isin([1,2,3,4,5,6])] = 1
wine['quality'].loc[wine['quality'].isin([7,8,9,10])] = 0


##histogram plot for frequencies of the ratings
#out = pd.cut(wine['quality'], bins=range(2,10), include_lowest=True)
#ax = out.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(7,5))
## print(wine)
#ax.set_title('Wine Quality Frequency')
#ax.set_xlabel('Rating')
#ax.set_ylabel('Frequency')
#ax.set_xticklabels([3,4,5,6,7,8,9])
#rects = ax.patches
#print(rects)
#count,_ = np.histogram(wine['quality'], bins = range(3,11))
#print(count)
## Make some labels.
#labels = list(count)
#
#for rect, label in zip(rects, labels):
#    height = rect.get_height()
#    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
#            ha='center', va='bottom')
#fig = ax.get_figure()
##fig.savefig(path + 'hist_wine.pdf')
#
#mad_score_wine = robust.mad(wine['quality'])
#mad_score_wine 
#wine[abs(wine['quality'].values-stats.median(wine['quality'].values))/mad_score_wine > 2]
#
#mad_score_adult = robust.mad(vals['income'])
#vals[abs(vals['income'].values-stats.median(vals['income'].values))/mad_score_adult > 2]
#
#g = sns.heatmap(wine.corr(),annot=True, fmt='.2f') #Use heat map to show little colinearity.
## g = sns.pairplot(wine, hue="quality", palette="husl") #send to figure.
#
#fig = g.get_figure()
##fig.savefig(path+ 'heat_map_wine.pdf')

#
##summary table
#wine.describe().to_latex()
#wine.describe()



#######################################
#######################################
#######################################
#######################################
#######################################

rhc_wine = pd.read_csv(path + 'output/random_hill_climb_wine_main.csv',sep=',')


#######################################
#######################################
#######################################
#######################################
#######################################

rhc_wine_relu = rhc_wine[rhc_wine['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = rhc_wine_relu['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = rhc_wine_relu['mean_train_score'].values[i::9]
alphas = list(np.unique(rhc_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(rhc_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('rhc_train_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################
rhc_wine['param_MLP__activation']
rhc_wine_log = rhc_wine[rhc_wine['param_MLP__activation']=='sigmoid'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = rhc_wine_log['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = rhc_wine_log['mean_train_score'].values[i::9]
alphas = list(np.unique(rhc_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(rhc_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('rhc_train_mean_score_log_wine.pdf')


#######################################
#######################################
#######################################
#######################################
#######################################

rhc_wine_relu = rhc_wine[rhc_wine['param_MLP__activation']=='relu'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = rhc_wine_relu['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = rhc_wine_relu['mean_test_score'].values[i::9]
alphas = list(np.unique(rhc_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(rhc_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('rhc_test_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################

rhc_wine['param_MLP__activation']
rhc_wine_log = rhc_wine[rhc_wine['param_MLP__activation']=='sigmoid'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = rhc_wine_log['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = rhc_wine_log['mean_test_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(rhc_wine_log['param_MLP__alpha']))
hidden_layer_names = np.unique(rhc_wine_log['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('rhc_test_mean_score_log_wine.pdf')



####################################
####################################
####################################
####################################
####################################

mean_accuracy_train = []
mean_accuracy_test = []

for num_samp in [10,100] + [int(pct*len(wine_train)) for pct in np.arange(0.1,1,0.05)]:
    samp = wine_train.sample(n=num_samp, random_state=42)
    wine_trainX = samp.loc[:,samp.columns!='quality'].copy()
    wine_trainY = samp['quality'].copy()
    wine_testX = wine_test.loc[:,wine_test.columns!='quality'].copy()
    wine_testY = wine_test['quality'].copy()
    algorithm = 'randomized hill climbing'
    layer = best_param_rhc[2]
    activation = best_param_rhc[0]
    alpha = best_param_rhc[1]
    nn_model1 = mlrose.NeuralNetwork(layer,activation=activation,algorithm='random_hill_climb',max_iters=1000,
                                     bias=False,is_classifier=True,learning_rate=alpha)
    nn_model1.fit(wine_trainX,wine_trainY)
    y_train_pred = nn_model1.predict(wine_trainX)
    y_train_accuracy = accuracy_score(wine_trainY,y_train_pred)
    y_train_accuracy
    print('Training accuracy is ',y_train_accuracy, ' with step size ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    y_test_pred = nn_model1.predict(wine_testX)
    y_test_accuracy = accuracy_score(wine_testY,y_test_pred)
    print('Testing accuracy: ',y_test_accuracy, ' with step size ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    mean_accuracy_train.append([num_samp,y_train_accuracy])    
    mean_accuracy_test.append([num_samp,y_test_accuracy])
train_df = pd.DataFrame(mean_accuracy_train)
train_df = train_df.set_index([0])
test_df = pd.DataFrame(mean_accuracy_test)
test_df = test_df.set_index([0])
f = plt.figure()
plt.plot(list(np.round(train_df.index,2)),[list(x)[0] for x in list(train_df.values)], label='Train')
plt.plot(list(np.round(test_df.index,2)),[list(x)[0] for x in list(test_df.values)], label='Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=12)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()
#f.savefig('rhc_learning_curve_wine.pdf')

####################################
####################################
####################################
####################################
####################################

mean_accuracy_train = []
mean_accuracy_test = []

for num_samp in [10,100] + [int(pct*len(wine_train)) for pct in np.arange(0.1,1,0.05)]:
    samp = wine_train.sample(n=num_samp, random_state=42)
    wine_trainX = samp.loc[:,samp.columns!='quality'].copy()
    wine_trainY = samp['quality'].copy()
    wine_testX = wine_test.loc[:,wine_test.columns!='quality'].copy()
    wine_testY = wine_test['quality'].copy()
    layer = best_param_sa[2]
    activation = best_param_sa[0]
    alpha = best_param_sa[1]
    algorithm = 'simulated annealing'
    nn_model1 = mlrose.NeuralNetwork(layer,activation=activation,algorithm='simulated_annealing',max_iters=1000,
                                     bias=False,is_classifier=True,learning_rate=alpha)
    nn_model1.fit(wine_trainX,wine_trainY)
    y_train_pred = nn_model1.predict(wine_trainX)
    y_train_accuracy = accuracy_score(wine_trainY,y_train_pred)
    y_train_accuracy
    print('Training accuracy is ',y_train_accuracy, ' with step size ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    y_test_pred = nn_model1.predict(wine_testX)
    y_test_accuracy = accuracy_score(wine_testY,y_test_pred)
    print('Testing accuracy: ',y_test_accuracy, ' with step size ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    mean_accuracy_train.append([num_samp,y_train_accuracy])    
    mean_accuracy_test.append([num_samp,y_test_accuracy])
train_df = pd.DataFrame(mean_accuracy_train)
train_df = train_df.set_index([0])
test_df = pd.DataFrame(mean_accuracy_test)
test_df = test_df.set_index([0])
f = plt.figure()
plt.plot(list(np.round(train_df.index,2)),[list(x)[0] for x in list(train_df.values)], label='Train')
plt.plot(list(np.round(test_df.index,2)),[list(x)[0] for x in list(test_df.values)], label='Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=12)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()
#f.savefig('sa_learning_curve_wine.pdf')

####################################
####################################
####################################
####################################
####################################

mean_accuracy_train = []
mean_accuracy_test = []

for num_samp in [50,100] + [int(pct*len(wine_train)) for pct in np.arange(0.1,1,0.05)]:
    samp = wine_train.sample(n=num_samp, random_state=42)
    wine_trainX = samp.loc[:,samp.columns!='quality'].copy()
    wine_trainY = samp['quality'].copy()
    wine_testX = wine_test.loc[:,wine_test.columns!='quality'].copy()
    wine_testY = wine_test['quality'].copy()
    layer = best_param_ga[2]
    activation = best_param_ga[0]
    alpha = best_param_ga[1]
    algorithm = 'simulated annealing'
    nn_model1 = mlrose.NeuralNetwork(layer,activation=activation,algorithm='genetic_alg',max_iters=1000,
                                     bias=False,is_classifier=True,learning_rate=alpha)
    nn_model1.fit(wine_trainX,wine_trainY)
    y_train_pred = nn_model1.predict(wine_trainX)
    y_train_accuracy = accuracy_score(wine_trainY,y_train_pred)
    y_train_accuracy
    print('Training accuracy is ',y_train_accuracy, ' with step size ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    y_test_pred = nn_model1.predict(wine_testX)
    y_test_accuracy = accuracy_score(wine_testY,y_test_pred)
    print('Testing accuracy: ',y_test_accuracy, ' with step size ', 
          alpha, ', layers ',layer,', activation function,', 
          activation,', and algorithm ', algorithm)
    mean_accuracy_train.append([num_samp,y_train_accuracy])    
    mean_accuracy_test.append([num_samp,y_test_accuracy])
train_df = pd.DataFrame(mean_accuracy_train)
train_df = train_df.set_index([0])
test_df = pd.DataFrame(mean_accuracy_test)
test_df = test_df.set_index([0])
f = plt.figure()
plt.plot(list(np.round(train_df.index,2)),[list(x)[0] for x in list(train_df.values)], label='Train')
plt.plot(list(np.round(test_df.index,2)),[list(x)[0] for x in list(test_df.values)], label='Test')
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=12)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=15)
plt.show()
#f.savefig('ga_learning_curve_wine.pdf')







####################################
####################################
####################################
####################################
####################################


rhc_wine_relu_train = rhc_wine[rhc_wine['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
rhc_wine_log_train = rhc_wine[rhc_wine['param_MLP__activation']=='sigmoid'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
y_relu_train = np.zeros(9)
y_relu_train = list(y_relu_train)
y_log_train = np.zeros(9)
y_log_train = list(y_log_train)

rhc_wine_relu = rhc_wine[rhc_wine['param_MLP__activation']=='relu'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
rhc_wine_log = rhc_wine[rhc_wine['param_MLP__activation']=='sigmoid'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = rhc_wine_relu['param_MLP__alpha']
y_relu = np.zeros(9)
y_relu = list(y_relu)
y_log = np.zeros(9)
y_log = list(y_log )
for i in range(9):
  y_relu[i] = rhc_wine_relu['mean_test_score'].values[i::9]
  y_log[i] = rhc_wine_log['mean_test_score'].values[i::9]
  y_relu_train[i] = rhc_wine_relu_train['mean_train_score'].values[i::9]
  y_log_train[i] = rhc_wine_log_train['mean_train_score'].values[i::9]
alphas = list(np.unique(rhc_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(rhc_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    if i == 0 and j == 0:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['step size','alphas']
        df_relu = df_relu.set_index('step size')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j],label='relu validation')
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['step size','alphas']
        df_log = df_log.set_index('step size')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid validation')
        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
        df_relu_train.columns = ['step size','alphas']
        df_relu_train = df_relu_train.set_index('step size')        
        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j],label='relu train')
        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
        df_log_train.columns = ['step size','alphas']
        df_log_train = df_log_train.set_index('step size')
        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid train')
        axarr[i,j].legend(loc='lower left')
    else:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['step size','alphas']
        df_relu = df_relu.set_index('step size')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['step size','alphas']
        df_log = df_log.set_index('step size')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
        df_relu_train.columns = ['step size','alphas']
        df_relu_train = df_relu_train.set_index('step size')        
        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
        df_log_train.columns = ['step size','alphas']
        df_log_train = df_log_train.set_index('step size')
        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
#fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
#fig.subplots_adjust(bottom = 0.256)
#fig.legend(loc='lower left')
fig.get_figure()
fig.savefig('rhc_all_mean_score_wine.pdf')


####################################
####################################
####################################
####################################
####################################
####################################
####################################
####### Simulated Annealing ########
####################################
####################################
####################################
####################################
####################################
####################################
####################################

sa_wine = pd.read_csv(path + 'output/simulated_annealing_wine_main.csv',sep=',')


#######################################
#######################################
#######################################
#######################################
#######################################

sa_wine_relu = sa_wine[sa_wine['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = sa_wine_relu['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = sa_wine_relu['mean_train_score'].values[i::9]
alphas = list(np.unique(sa_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(sa_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('sa_train_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################
sa_wine['param_MLP__activation']
sa_wine_log = sa_wine[sa_wine['param_MLP__activation']=='sigmoid'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = sa_wine_log['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = sa_wine_log['mean_train_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(sa_wine_log['param_MLP__alpha']))
hidden_layer_names = np.unique(sa_wine_log['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('sa_train_mean_score_log_wine.pdf')


#######################################
#######################################
#######################################
#######################################
#######################################

sa_wine_relu = sa_wine[sa_wine['param_MLP__activation']=='relu'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = sa_wine_relu['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = sa_wine_relu['mean_test_score'].values[i::9]
alphas = list(np.unique(sa_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(sa_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('sa_test_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################

sa_wine['param_MLP__activation']
sa_wine_log = sa_wine[sa_wine['param_MLP__activation']=='sigmoid'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = sa_wine_log['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = sa_wine_log['mean_test_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(sa_wine_log['param_MLP__alpha']))
hidden_layer_names = np.unique(sa_wine_log['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('sa_test_mean_score_log_wine.pdf')













####################################
####################################
####################################
####################################
####################################


sa_wine_relu_train = sa_wine[sa_wine['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
sa_wine_log_train = sa_wine[sa_wine['param_MLP__activation']=='sigmoid'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
y_relu_train = np.zeros(9)
y_relu_train = list(y_relu_train)
y_log_train = np.zeros(9)
y_log_train = list(y_log_train)

sa_wine_relu = sa_wine[sa_wine['param_MLP__activation']=='relu'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
sa_wine_log = sa_wine[sa_wine['param_MLP__activation']=='sigmoid'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = sa_wine_relu['param_MLP__alpha']
y_relu = np.zeros(9)
y_relu = list(y_relu)
y_log = np.zeros(9)
y_log = list(y_log )
for i in range(9):
  y_relu[i] = sa_wine_relu['mean_test_score'].values[i::9]
  y_log[i] = sa_wine_log['mean_test_score'].values[i::9]
  y_relu_train[i] = sa_wine_relu_train['mean_train_score'].values[i::9]
  y_log_train[i] = sa_wine_log_train['mean_train_score'].values[i::9]
alphas = list(np.unique(sa_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(sa_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    if i == 0 and j == 0:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['step size','alphas']
        df_relu = df_relu.set_index('step size')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j],label='relu validation')
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['step size','alphas']
        df_log = df_log.set_index('step size')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid validation')
        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
        df_relu_train.columns = ['step size','alphas']
        df_relu_train = df_relu_train.set_index('step size')        
        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j],label='relu train')
        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
        df_log_train.columns = ['step size','alphas']
        df_log_train = df_log_train.set_index('step size')
        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid train')
        axarr[i,j].legend(loc='lower left')
    else:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['step size','alphas']
        df_relu = df_relu.set_index('step size')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['step size','alphas']
        df_log = df_log.set_index('step size')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
        df_relu_train.columns = ['step size','alphas']
        df_relu_train = df_relu_train.set_index('step size')        
        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
        df_log_train.columns = ['step size','alphas']
        df_log_train = df_log_train.set_index('step size')
        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
#fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
#fig.subplots_adjust(bottom = 0.256)
#fig.legend(loc='lower left')
fig.get_figure()
fig.savefig('sa_all_mean_score_wine.pdf')


####################################
####################################
####################################
####################################
####################################
####################################
####################################
####### Genetic Algorithm ##########
####################################
####################################
####################################
####################################
####################################
####################################
####################################

ga_wine = pd.read_csv(path + 'output/genetic_alg_wine_main_fast_run_1.csv',sep=',')


#######################################
#######################################
#######################################
#######################################
#######################################

ga_wine_relu = ga_wine[ga_wine['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ga_wine_relu['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ga_wine_relu['mean_train_score'].values[i::9]
alphas = list(np.unique(ga_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(ga_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ga_train_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################
ga_wine['param_MLP__activation']
ga_wine_log = ga_wine[ga_wine['param_MLP__activation']=='sigmoid'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ga_wine_log['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ga_wine_log['mean_train_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ga_wine_log['param_MLP__alpha']))
hidden_layer_names = np.unique(ga_wine_log['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ga_train_mean_score_log_wine.pdf')


#######################################
#######################################
#######################################
#######################################
#######################################

ga_wine_relu = ga_wine[ga_wine['param_MLP__activation']=='relu'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ga_wine_relu['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ga_wine_relu['mean_test_score'].values[i::9]
alphas = list(np.unique(ga_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(ga_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ga_test_mean_score_relu_wine.pdf')
####################################
####################################
####################################
####################################
####################################

ga_wine['param_MLP__activation']
ga_wine_log = ga_wine[ga_wine['param_MLP__activation']=='sigmoid'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ga_wine_log['param_MLP__alpha']
y = np.zeros(9)
y = list(y)
for i in range(9):
  y[i] = ga_wine_log['mean_test_score'].values[i::9]
fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
alphas = list(np.unique(ga_wine_log['param_MLP__alpha']))
hidden_layer_names = np.unique(ga_wine_log['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    df = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y[i+3*j])],axis=1)
    df.columns = ['alpha','alphas']
    df = df.set_index('alpha')
    df['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
fig.suptitle('Logistic Sigmoid Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
fig.subplots_adjust(bottom = 0.256)
fig.get_figure()
fig.savefig('ga_test_mean_score_log_wine.pdf')

















#######################################
#######################################
#######################################
#######################################
#######################################

ga_wine_relu_train = ga_wine[ga_wine['param_MLP__activation']=='relu'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
ga_wine_log_train = ga_wine[ga_wine['param_MLP__activation']=='sigmoid'][['mean_train_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
y_relu_train = np.zeros(9)
y_relu_train = list(y_relu_train)
y_log_train = np.zeros(9)
y_log_train = list(y_log_train)

ga_wine_relu = ga_wine[ga_wine['param_MLP__activation']=='relu'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
ga_wine_log = ga_wine[ga_wine['param_MLP__activation']=='sigmoid'][['mean_test_score','param_MLP__alpha','param_MLP__hidden_layer_sizes']]
x = ga_wine_relu['param_MLP__alpha']
y_relu = np.zeros(9)
y_relu = list(y_relu)
y_log = np.zeros(9)
y_log = list(y_log )
for i in range(9):
  y_relu[i] = ga_wine_relu['mean_test_score'].values[i::9]
  y_log[i] = ga_wine_log['mean_test_score'].values[i::9]
  y_relu_train[i] = ga_wine_relu_train['mean_train_score'].values[i::9]
  y_log_train[i] = ga_wine_log_train['mean_train_score'].values[i::9]
alphas = list(np.unique(ga_wine_relu['param_MLP__alpha']))
hidden_layer_names = np.unique(ga_wine_relu['param_MLP__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    if i == 0 and j == 0:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['step size','alphas']
        df_relu = df_relu.set_index('step size')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j],label='relu validation')
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['step size','alphas']
        df_log = df_log.set_index('step size')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid validation')
        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
        df_relu_train.columns = ['step size','alphas']
        df_relu_train = df_relu_train.set_index('step size')        
        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j],label='relu train')
        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
        df_log_train.columns = ['step size','alphas']
        df_log_train = df_log_train.set_index('step size')
        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid train')
        axarr[i,j].legend(loc='lower left')
    else:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['step size','alphas']
        df_relu = df_relu.set_index('step size')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['step size','alphas']
        df_log = df_log.set_index('step size')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
        df_relu_train.columns = ['step size','alphas']
        df_relu_train = df_relu_train.set_index('step size')        
        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
        df_log_train.columns = ['step size','alphas']
        df_log_train = df_log_train.set_index('step size')
        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
#fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
#fig.subplots_adjust(bottom = 0.256)
#fig.legend(loc='lower left')
fig.get_figure()
fig.savefig('ga_all_mean_score_wine.pdf')

