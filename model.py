#organize imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping
import math
#import numpy as np
import pandas as pd


#load data from dataser

#Age(Years),Sex(M,F),Pertinent Medical History(1,0),ART,PAP,RAP,Respiratory Rate, Mode of Ventilation,Survived(1,0)
names = ['Age', 'Sex', 'Pertinent_Medical_History', 'ART', 'PAP', 'RAP', 'Respiratory_Rate', 'Ventilation_Mode', 'Survived']
data = pd.read_csv("/home/diego/Documents/Redes 2017-II/KerasModel/patients-vital-signs.csv", names=names)

#functions to parse data
def VentilationToInt(ventType):
    if pd.isnull(ventType):
        return 0
    else:
        return{
            'NaN':0,
            'Controlled':1,
            'Controlled (spontaneous attempts initially. Then sedated)':2,
            'Controlled and ambu ventilation':3,
            'Controlled changed to intermittent mandatory ventilation':4,
            'Controlled with rare spontaneous attempts':5,
            'Controlled with rare spontaneous breaths':6,
            'High frequency jet ventilation':7,
            'Intermittent mandatory ventilation':8,
            'Intermittent mandatory ventilation changed to spontaneous':9,
            'Spontaneous':10,
            'Iron Lung':11,
            'Spontaneous with CPAP':12
        }[ventType]

def genderToInt(gender):
    if gender == 'M':
        return 0
    else:
        return 1

def numDataToInt(data):
    if pd.isnull(data):
        return float('nan')
    else:
        return eval(data)

#separate data from results
predictors = data.drop(['Survived'], axis=1)
target = to_categorical(data.Survived)

#parse data from string to numerical
predictors.Sex = predictors.Sex.apply(genderToInt)
predictors.Ventilation_Mode = predictors.Ventilation_Mode.apply(VentilationToInt)
predictors.ART = predictors.ART.apply(numDataToInt)
predictors.PAP = predictors.PAP.apply(numDataToInt)
predictors.RAP = predictors.RAP.apply(numDataToInt)
predictors.Respiratory_Rate = predictors.Respiratory_Rate.apply(numDataToInt)
predictors = predictors.as_matrix()

#creating model
model = Sequential()
model.add(Dense(60, activation='relu', input_shape = (predictors.shape[1],)))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(2, activation='softmax'))

#my_optimizer= optimizers.SGD(lr=0.01)
#compiling model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
 metrics = ['accuracy'])

#EarlyStopping monitor
early_stopping_monitor = EarlyStopping(patience=3)

#fitting model
model.fit(predictors, target, epochs=10, batch_size=64, validation_split=0.3, callbacks=[early_stopping_monitor])
