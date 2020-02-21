#!/usr/bin/env python
# coding: utf-8

# # Seleccionando Características para mejorar el desempeño
# Este Laboratorio es Basado en el Curso de Dimensionality Reduction de DATACAMP®
# 
# 
# En este laboratorio implementaremos técnicas para granatizar el desempeño de nuestro modelo de reconocimmiento de patrones, apicando técnicas de selección de características.
# 

# # Contruyendo un clasificador para detectar diabetes
# 
# En este laboratorio se utilizará el conjunto de datos sobre diabetes de la tribu indigena Pima para predecir si una persona tiene diabetes mediante regresión logística. Hay 8 características y un objetivo (Y) en este conjunto de datos. Los datos se han dividido en un conjunto de entrenamiento y otro de prueba. A continuación se cargarán como  X_train, y_train, X_test, y y_test.

# In[14]:


# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
X_train=pd.read_csv("Pima_X_train.csv")#reading a dataset in a dataframe using pandas
print("X_train",X_train.shape)

X_test=pd.read_csv("Pima_X_test.csv")#reading a dataset in a dataframe using pandas
print("X_test",X_test.shape)

y_train = X_train['result']
print("y_train",y_train.shape)


y_test = X_test['result']
print("y_test",y_train.shape)

X_train = X_train.drop('result', axis=1)
print("X_train droped",X_train.shape)

X_test = X_test.drop('result', axis=1)
print("X_test droped",X_test.shape)

#X=[X_train,X_test]
X = pd.concat([X_train, X_test], axis=0)
print("X",X.shape)

y=pd.concat([y_train, y_test], axis=0)
print("y",y.shape)


# Se carga una instancia  StandardScaler() predefinida como scaler una función de regresión logística LogisticRegression() definida como lr.

# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression(solver='liblinear')


# Ajuste el escalador en las características de entrenamiento y transforme estas características en una única  en un sólo comando.

# In[16]:


# Fit the scaler on the training features and transform these in one go
#X_train_std = scaler.____(____)
X_train_std = scaler.fit_transform(X_train)


# Ajuste el modelo de regresión logística en los datos de entrenamiento escalados.

# In[17]:


# Fit the logistic regression model on the scaled training data
lr.fit(X_train_std, y_train)

Escalar las características del conjunto test.
# In[18]:


# Scale the test features
X_test_std = scaler.transform(X_test)


# Predecir la presencia de diabetes en el conjunto de pruebas que fueron escaladas.

# In[19]:


# Predict diabetes presence on the scaled test set
y_pred = lr.predict(X_test_std)


#    Imprimir los resultados de precision

# In[20]:


# Prints accuracy metrics and feature coefficients
print("{0:.1%} accuracy on test set.".format(accuracy_score(y_test, y_pred))) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))


# ¡Excelente! Hemos obtenido casi un 80% de precisión en el conjunto de prueba. Eche un vistazo a las diferencias en los coeficientes del modelo para las diferentes características.

# # Eliminación de características de manera manual y recursiva
# 
# Ahora que hemos creado un clasificador de diabetes, veamos si podemos reducir la cantidad de características sin dañar demasiado la precisión del modelo.
# 
# En la segunda línea de código de éste bloque, las características han sido seleccionadas  del dataframe original. Realice los ajustes pertinentes.
# 

# Ejecute el código dado, luego elimine la característica de X con el coeficiente más bajo de acuerdo a lo generado por el modelo. 

# In[21]:


from sklearn.model_selection import train_test_split

diabetes_df=X;

# Remove the feature with the lowest model coefficient
X = diabetes_df[['pregnant', 'glucose', 'triceps', 'insulin', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))


# Ejecute el código y elimine 2 características más con los coeficientes más bajos del modelo.

# In[22]:


# Remove the 2 features with the lowest model coefficients
X = diabetes_df[['pregnant', 'glucose', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))


# Ejecute el código y solo mantenga la caracterísitca con el coeficiente más alto.

# In[23]:


# Only keep the feature with the highest coefficient
X = diabetes_df[['glucose']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model to the data
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))


# ¡Buen trabajo! Eliminar todas las funciones menos una solo redujo la precisión en un pequeño porcentaje.

# # Eliminación de características de manera automática y recursiva
# 
# Ahora automaticemos este proceso recursivo. Cree un Wrap Eliminador de características recursivas (RFE) alrededor de nuestro estimador de regresión logística y defina el número deseado de características.

# Importamos nuevamente nuestors datos originales

# In[24]:


X_train=pd.read_csv("Pima_X_train.csv")#reading a dataset in a dataframe using pandas


X_test=pd.read_csv("Pima_X_test.csv")#reading a dataset in a dataframe using pandas


y_train = X_train['result']



y_test = X_test['result']


X_train = X_train.drop('result', axis=1)
print("X_train ",X_train.shape)

X_test = X_test.drop('result', axis=1)
print("X_test ",X_test.shape)

#X=[X_train,X_test]
X = pd.concat([X_train, X_test], axis=0)
print("X",X.shape)

y=pd.concat([y_train, y_test], axis=0)
print("y",y.shape)


# Cree el RFE con un LogisticRegression()estimador y seleccionando sólo 3 características.

# In[29]:


from sklearn.feature_selection import RFE
# Create the RFE with a LogisticRegression estimator and 3 features to select
rfe = RFE(estimator=lr, n_features_to_select=3, verbose=1)

# Fits the eliminator to the data
rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Calculates the test set accuracy
acc = accuracy_score(y_test, rfe.predict(X_test))
print("{0:.1%} accuracy on test set.".format(acc)) 


# ¡Excelente! Al eliminar 5 características,  definimos las 3 más relevantes, obtenemos así una precisión del 80,6% en el conjunto de prueba.

# # Selección de características Utilizando Árboles de decisión
# 
# Ahora utilizaremos Árboles de decisión que nos permitirán identificar las características candidatas a ser removidas.

# # Construyendo un Modelo de Bósques aleatorios
# 
# Trabajaremos igualmente con  en el conjunto de datos Pima para predecir si un individuo tiene o nó diabetes. Esta vez usando un clasificador de bosque aleatorio. Ajustará el modelo en los datos de entrenamiento después de realizar la división de prueba de tren y consultará los valores de importancia de la función.
# 
# Ahora cargaremos las características y los conjuntos de datos de datos como X y y. Así como  los paquetes y las funciones necesarias.

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#rf = RandomForestClassifier(n_estimators=10)
rf = RandomForestClassifier()

X_train=pd.read_csv("Pima_X_train.csv")#reading a dataset in a dataframe using pandas


X_test=pd.read_csv("Pima_X_test.csv")#reading a dataset in a dataframe using pandas


y_train = X_train['result']



y_test = X_test['result']


X_train = X_train.drop('result', axis=1)
print("X_train ",X_train.shape)

X_test = X_test.drop('result', axis=1)
print("X_test ",X_test.shape)

#X=[X_train,X_test]
X = pd.concat([X_train, X_test], axis=0)
print("X",X.shape)

y=pd.concat([y_train, y_test], axis=0)
print("y",y.shape)


# Establezca un conjunto de evaluación del 25%,  estableciendo una relación de conjuntos de entrenamiento-prueba 75% -25%.

# In[36]:


# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))


# ¡Buen trabajo! El modelo de bosques aleatorios obtiene casi un 79% de precisión en el conjunto de prueba y la "glucosa" es la característica más importante (0.19).

# # Bosque Aleatorio para la selección de características
# 
# Ahora usemos el modelo de Bosque aleatorio ajustado para seleccionar las características más importantes de nuestro conjunto de datos de entrada X.
# 
# rf, corresponde al modelo entrenado en la secuencia anterior.

# In[42]:


# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.13

# Prints out the mask
print(mask)


# Seleccione las características más importantes aplicando la máscara a X.

# In[48]:


# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.13

# Apply the mask to the feature dataset X
reduced_X = X.loc[:, mask]

# prints out the selected column names
print(reduced_X.columns)


# ¡Bien hecho! Solo las características 'glucosa' 'insulina' y 'age' se consideraron suficientemente importantes.

# # Eliminación de características recursivas con bosques aleatorios
# 
# Cree un Wrap Eliminador de características recursivas (RFE) alrededor de un modelo de bosque aleatorio para eliminar las características paso a paso. Este método es más conservador en comparación con la selección de características después de aplicar un umbral  único de importancia. Dado que descartar una característica puede influir en las importancias relativas de las otras.

# Cree un eliminador de características recursivas que seleccionará las 2 características más importantes utilizando un modelo de bosque aleatorio.
# 

# In[51]:


# Wrap the feature eliminator around the random forest model
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, verbose=1)


# Ajuste el RFE a los datos de entrenamiento

# In[52]:


# Fit the model to the training data
rfe.fit(X_train, y_train)


# Cree una máscara usando el eliminador ajustado y apliquelo al dataset X

# In[68]:


# Create a mask using an attribute of rfe
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)


# Cambie los parámetros del RFE() para elimnar dos caracterísiticas a cada paso (step)

# In[75]:


# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, step=1, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)


# ¡Excelente! En comparación con el método de umbral rápido del ejercicio anterior, una de las características seleccionadas es diferente.

# # Regresión lineal regularizada
# 
# Ahora utilizaremos el concepto de regularización, para evitar el sobre ajuste del modelo de predicción e igualmente identificar las características suceptibles de ser seleccionadas.

# # Crear un regresor LASSO
# 
# Ahora se trabajará en el conjunto de datos numéricos de mediciones corporales ANSUR para predecir el índice de masa corporal (BMI) de una persona utilizando un regresor Lasso (Least Absolute Shrinkage and Selection Operator)  preimportado. El BMI  es una métrica derivada de la altura y el peso del cuerpo, pero esas dos características se han eliminado del conjunto de datos para forzar al modelo aun mayor análisis.
# 
# Primero, estandarizará los datos utilizando el StandardScaler() que se ha instanciado como scaler para asegurarse de que todos los coeficientes se enfrentan a una fuerza de regularización comparable que intenta reducirlos.
# 
# Todas las funciones y clases necesarias más los conjuntos de datos de entrada X y y cargarán a continuación.

# In[110]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X=pd.read_csv("ANSUR-MALE.csv", encoding='latin-1')#reading a dataset in a dataframe using pandas
y=pd.read_csv("y_ANSUR.csv", encoding='latin-1')#reading a dataset in a dataframe using pandas
y =y['BMI']
X = X.drop(['Age','Branch','Component','DODRace','Date','Ethnicity','Gender','Heightin','Installation','PrimaryMOS','SubjectNumericRace','SubjectsBirthLocation','Weightlbs','WritingPreference','stature',
'subjectid','weightkg'], axis=1)
print(X.shape)
print(y.shape)


# Establezca un conjunto de evaluación del 30%,  estableciendo una relación de conjuntos de entrenamiento-prueba 70% -30%.

# In[114]:


# Set the test size to 30% to get a 70-30% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Ajuste el escalador en las características de entrenamiento y transfórmelas de una vez.

# In[119]:


# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)


# Cree un modelo LASSO

# In[120]:


# Create the Lasso model
la = Lasso()


# Ajúste el modelo LASSO a los datos de entrenamiento escalados

# In[121]:


# Fit it to the standardized training data
la.fit(X_train_std, y_train)


# ¡Buen trabajo! Has ajustado el modelo Lasso a los datos de entrenamiento estandarizados. ¡Ahora veamos los resultados!

# # Resultados del modelo LASSO:
# 
# Ahora que se ha entrenado el modelo LASSO, evaluaremos su capacidad predictiva ( R2 ) en el conjunto de prueba, y enumeremos cuantas características son ignoradas al ser sus coeficientes reducidos a cero.

# In[123]:


# Transform the test set with the pre-fitted scaler
X_test_std = scaler.transform(X_test)

# Calculate the coefficient of determination (R squared) on X_test_std
r_squared = la.score(X_test_std, y_test)
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))

# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0

# Calculate how many features have a zero coefficient
n_ignored = sum(zero_coef)
print("The model has ignored {} out of {} features.".format(n_ignored, len(la.coef_)))


# ¡Que bien! Podemos predecir casi el 85% de la varianza en el valor de IMC usando solo 9 de 91 de las características. Sin embargo, el R ^ 2 podría ser más alto.

# # Ajustando la influencia de la regularización
# 
# El modelo actual de LASSO tiene una puntuación (R2) del 84.7%. Cuando un modelo aplica una regularización demasiado grande, puede sufrir un alto sesgo, lo que perjudica su poder predictivo.
# 
# Mejoremos el equilibrio entre el poder predictivo y la simplicidad del modelo ajustando el parámetro alpha.

# Encontrar el valor más alto para alpha que mantendrá  el valor (R2) por encima de 98% de las opciones: 1,   0.50,   0.1   y  0.01

# In[129]:


# Find the highest alpha value with R-squared above 98%
la = Lasso(alpha=0.1, random_state=0)

# Fits the model and calculates performance stats
la.fit(X_train_std, y_train)
r_squared = la.score(X_test_std, y_test)
n_ignored_features = sum(la.coef_ == 0)

# Print peformance stats 
print("The model can predict {0:.1%} of the variance in the test set.".format(r_squared))
print("{} out of {} features were ignored.".format(n_ignored_features, len(la.coef_)))


# ¡Excelente! Con esta valor para la regularización, podemos predecir el 98% de la varianza en el valor de IMC mientras ignoramos 2/3 de las características.

# # Combinando Selectores de Características
# 
# Ahora encontraremos de manera automática el valor óptimo  de Alpha, mediante el uso de el regresor LassoCV.

# # Creando un regresor LassoCV
# 
# Ahora predeciremos la circunferencia del bíceps en una submuestra del conjunto de datos ANSUR masculino utilizando el regresor LassoCV()  que ajusta automáticamente la intensidad de la regularización (valor alfa) mediante la validación cruzada.
# 
# Primero cargamos los datos. Y laslibrerias necesarias.

# In[133]:


X=pd.read_csv("ANSUR-MALE.csv", encoding='latin-1')#reading a dataset in a dataframe using pandas
mask=['acromialheight','axillaheight','bideltoidbreadth','buttockcircumference','buttockkneelength','buttockpopliteallength','cervicaleheight','chestcircumference','chestheight','earprotrusion','footbreadthhorizontal','forearmcircumferenceflexed','handlength','headbreadth','heelbreadth','hipbreadth','iliocristaleheight','interscyeii','lateralfemoralepicondyleheight','lateralmalleolusheight','neckcircumferencebase','radialestylionlength','shouldercircumference','shoulderelbowlength','sleeveoutseam','thighcircumference','thighclearance','verticaltrunkcircumferenceusa','waistcircumference','waistdepth','wristheight']
X=X.loc[0:999, mask]
BMI=pd.read_csv("y_ANSUR.csv", encoding='latin-1')#reading a dataset in a dataframe using pandas
mask=['BMI']
BMI=BMI.loc[0:999, mask]
X['BMI']=BMI
y=pd.read_csv("y-lassoCV.csv", encoding='latin-1')#reading a dataset in a dataframe using pandas
y =y['Biceps']
print(X.shape)
print(y.shape)
# Set the test size to 30% to get a 70-30% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# Cree y ajuste el modelo LassoCV en el conjunto de entrenamiento.

# In[134]:


from sklearn.linear_model import LassoCV

# Create and fit the LassoCV model on the training set
lcv = LassoCV()
lcv.fit(X_train, y_train)
print('Optimal alpha = {0:.3f}'.format(lcv.alpha_))


# Calculamos el (R2) en el conjunto de prueba

# In[135]:


# Calculate R squared on the test set
r_squared = lcv.score(X_test, y_test)
print('The model explains {0:.1%} of the test set variance'.format(r_squared))


# Creamos una máscara para coeficientes que no sean iguales a cero

# In[137]:


# Create a mask for coefficients not equal to zero
lcv_mask = lcv.coef_ != 0
print('{} features out of {} selected'.format(sum(lcv_mask), len(lcv_mask)))


# ¡Excelente! Obtuvimos una R2 bueno y eliminamos 6 características. Guardaremos la máscara lcv_mask para más adelante.

# # Ensamble de modelos y votación
# 
# El LassoCV()modelo seleccionó 26 de las 32 características. No está mal, pero tampoco es una reducción espectacular de la dimensionalidad. Usemos dos modelos más para seleccionar las 10 características que consideran más importantes utilizando el Eliminador de funciones recursivas (RFE).

# Seleccione 10 funciones con RFE en un GradientBoostingRegressor, descarte 3 características en cada paso.

# In[318]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor

# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=____, 
             n_features_to_select=____, step=____, verbose=1)
rfe_gb.fit(X_train, y_train)


# Calculamos el R2 en el conjunto de pruebas

# In[319]:


# Calculate the R squared on the test set
r_squared = rfe_gb.____
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))


# Asigne la matriz de soporte del modelo ajustado a gb_mask

# In[320]:


# Assign the support array to gb_mask
gb_mask = ____


# Modifique ahora el primer paso para seleccionar 10 características con RFE en un RandomForestRegressor(), elimine 3 características en cada paso

# In[321]:


# Modify the first step to select 10 features with RFE on a RandomForestRegressor() and drop 3 features on each step.

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Select 10 features with RFE on a RandomForestRegressor, drop 3 features on each step
rfe_rf = RFE(estimator=____, 
             n_features_to_select=____, step=____, verbose=1)
rfe_rf.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_rf.score(X_test, y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

# Assign the support array to gb_mask
rf_mask = rfe_rf.support_


# ¡Buen trabajo! Incluyendo el modelo lineal Lasso del ejercicio anterior, ahora tenemos los votos de 3 modelos en los que las características son importantes.

# # Combinando 3 selectores de cartacterísticas
# 
# Combinaremos los votos de los 3 modelos que construyó en los ejercicios anteriores, para decidir qué características son importantes en una máscara global. Luego usaremos esta máscara para reducir la dimensionalidad y ver cómo funciona un regresor lineal simple en el conjunto de datos reducido.

# Sume los votos de los tres modelos usando np.sum()

# In[322]:


# Sum the votes of the three models
votes = ____
print(votes)


# Crear una máscara global para todas las características seleccionadas por los 3 modelos

# In[323]:


# Create a mask for features selected by all 3 models
meta_mask = ____
print(meta_mask)


# Aplique reduccción de la dimesionalidad en X, e imprima que características fueron seleccionadas

# In[325]:


# Apply the dimensionality reduction on X
X_reduced = ____
print(X_reduced.columns)


# Ingrese el conjunto reducido para realizar una regresión lineal simple

# In[328]:


from sklearn.linear_model import LinearRegression
# Plug the reduced dataset into a linear regression pipeline
X_train, X_test, y_train, y_test = train_test_split(____, y, test_size=0.3, random_state=0)
lm.fit(scaler.fit_transform(X_train), y_train)
r_squared = lm.score(scaler.transform(X_test), y_test)
print('The model can explain {0:.1%} of the variance in the test set using {1:} features.'.format(r_squared, len(lm.coef_)))


# ¡Perfecto! ¡Usando los votos obtenidos de 3 modelos, pudimos seleccionar  seleccionar solo 8 características que permitieron que un modelo lineal simple obtuviera una alta precisión!
