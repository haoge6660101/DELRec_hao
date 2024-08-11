from keras import metrics
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import json
import math
import numpy as np
import os
import pandas as pd
import sklearn as sk

myDir = os.path.dirname(os.path.abspath(__file__))

jsonConfig = None
with open(os.path.join(myDir, 'config.json')) as f:
    jsonConfig = json.load(f)


df = pd.read_csv(jsonConfig['inputDataSetFile'], na_values='NOT FOUND', delimiter=';',
                 low_memory=False, decimal='.')


headers = df.columns.tolist()


queriedSet = df[headers].query(jsonConfig['selectionQuery'])
selectedFeaturesNames = jsonConfig['selectedFeatures']
for selectedFeature in selectedFeaturesNames:
    queriedSet[selectedFeature].fillna(0, inplace=True)


selectedFeatureSet = queriedSet[selectedFeaturesNames]


normalizedSet = sk.preprocessing.scale(selectedFeatureSet)


tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)



def createNetwork(featuresLength, activationType='relu', minNeurons = 0, optimizer = "optimizers.SGD(lr=0.01, clipvalue=0.5)"):
    input_vector = Input(shape=(featuresLength,))
    layers = [input_vector]
    for x in range(featuresLength - 1, minNeurons - 1, -1):
        layers.append(Dense(x, activation=activationType)(layers[-1]))
    for x in range(minNeurons + 1, featuresLength, 1):
        layers.append(Dense(x, activation=activationType)(layers[-1]))
    decoded = Dense(featuresLength, activation='linear')(layers[-1])

    autoencoder = Model(input_vector, decoded)
    autoencoder.compile(loss='mean_squared_error',
                        optimizer= eval(optimizer),
                        metrics=['mae'])

    history = autoencoder.fit(x=normalizedSet, y=normalizedSet,
                    epochs=500,
                    verbose=1,
                    callbacks=[tensorboard])

    modelOutput = autoencoder.predict(normalizedSet)

    loFactor = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    loFactor.fit_predict(normalizedSet)
    loScores = loFactor.negative_outlier_factor_

    isoForest = IsolationForest(max_samples=100)
    isoForest.fit(normalizedSet)
    isoScores = isoForest.score_samples(normalizedSet)

    means = np.mean(selectedFeatureSet, axis=0)
    std = np.std(selectedFeatureSet, axis=0)

    data = []
    for i in range(len(normalizedSet)):
        errorRates = []
        line = {}
        for j in range(len(normalizedSet[i])):
            featErr = math.pow(normalizedSet[i][j] - modelOutput[i][j], 2)
            errorRates.append(featErr)
            outputs = modelOutput[i][j]
            line["AE-output-" + selectedFeaturesNames[j]] = modelOutput[i][j]
            line["AE-err-" + selectedFeaturesNames[j]] = featErr
            line["normalized-" + selectedFeaturesNames[j]] = normalizedSet[i][j]
            line["inverse_transform-" + selectedFeaturesNames[j]] = modelOutput[i][j] * std[selectedFeaturesNames[j]] + means[selectedFeaturesNames[j]]
        line["autoEncoderScore"] = sum(errorRates)
        line["loScore"] = loScores[i]
        line["isoScore"] = isoScores[i]
        data.append(line)

    results = pd.DataFrame(data)
    queriedSet.reset_index()
    results.reset_index()
    df_results_01 = pd.concat([queriedSet, results], axis=1)

    df_results_01.to_csv(
        'out_' + jsonConfig['inputDataSetFile'] + "_dim_" + str(minNeurons) + "_loss_" + str(history.history['loss'][-1]) + ".csv", sep=';', float_format='%.15f', encoding='utf-8')


minNeuronsJson = jsonConfig["minNeurons"]
for jsonVal in minNeuronsJson:
    minNeurons = 0
    if jsonVal == "default":
        minNeurons = round(len(selectedFeaturesNames) * 0.66)
    else:
        minNeurons = int(jsonVal)
    createNetwork(len(selectedFeaturesNames), jsonConfig["activationType"], minNeurons, jsonConfig["optimizer"])