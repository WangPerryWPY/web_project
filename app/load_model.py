import joblib
import numpy as np
import os

# data shape:(1, 11)
def _predict(model_path, data):

    rf_model = joblib.load(model_path)

    y_pred = rf_model.predict(data)
    y_pred_proba = rf_model.predict_proba(data)

    return y_pred, y_pred_proba

def predict(data):
  model_path = os.path.dirname(os.path.realpath(__file__))+'/resources/RF_model.pkl'
  return _predict(model_path, data)


# if __name__ == '__main__':

#     # x = np.array([30, 1, 2, 27.28, 3.30, 31.29, 100.00, 1282.50, 0.020000000000000, 0.594931773879142, 0.0]).reshape(1, 11)
#     x = np.array([50, 0, 96, 23.15, 4.00, 20.26, 55.30, 100.39, 0.000000000000000, 0.105986652056980, 6.7]).reshape(1, 11)

#     model_path = './RF_model.pkl'

#     pred_class, prob = _predict(model_path, x)

#     # 0:EAS  1:CD
#     print(pred_class)
#     # predicted probability of 0 and 1
#     print(prob)