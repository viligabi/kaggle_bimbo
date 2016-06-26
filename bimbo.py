#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
col = {
    "Semana": "CW",
    "Agencia_ID":"depo",
    "Canal_ID": "sales_channel",
    "Ruta_SAK": "route_ID",
    "Cliente_ID": "client_ID",
    # "NombreCliente": "client_name",
    "Producto_ID": "product_ID",
    # "NombreProducto": "product_name",
    "Venta_uni_hoy": "sales_units_this_week",
    "Venta_hoy": "sales_this_week",
    "Dev_uni_proxima": "returns_unit_next_week",
    "Dev_proxima": "returns_next_week",
    "Demanda_uni_equil": "adjusted_demand"
}

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5

def prep_products():
    prod = pd.read_csv("data/producto_tabla.csv")
    prod[["weight","w_unit"]] = prod["NombreProducto"].str.extract("([0-9]+)([kKgGml]+)")
    prod.w_unit = prod.w_unit.str.lower() # to unify the unit Kg, G -> kg, g
    prod.weight[prod.w_unit=="g"] = prod.weight.astype(float) / 1000
    prod.weight[prod.w_unit=="ml"] = prod.weight.astype(float) / 1000
    prod.w_unit[prod.w_unit=="g"] = "kg"
    prod.w_unit[prod.w_unit=="ml"] = "l"

    prod.weight = prod.weight.astype(float)
    prod = prod.dropna(axis=0)
    return prod

products = prep_products()

nrows_train = 74180465
nrows_test  =  6999252
start = 1
end = 1*np.power(10,4)
n_lines = 1*np.power(10,4)
train = pd.read_csv("data/train.csv",header =0,skiprows=np.arange(1, end),nrows=n_lines)

target = train.iloc[:,10]
train = train.drop("Demanda_uni_equil",axis=1)

#scaler = preprocessing.StandardScaler().fit(train)
#train = scaler.transform(train)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf.fit(train, target)
from sklearn.cross_validation import cross_val_score
score = cross_val_score(clf, train, target)
score = clf.score(train, target)
print score

pass