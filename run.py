import logging
import pandas as pd
import numpy as np

from qiskit_ibm_provider import IBMProvider
from qiskit.providers.ibmq import least_busy
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin

class QKRR(BaseEstimator, RegressorMixin):

        def __init__(self, quantum_kernel, alpha=1.0):
            self.quantum_kernel = quantum_kernel
            self.alpha = alpha
            self.ridge = Ridge(alpha=alpha)

        def fit(self, X, y):
            K_train = self.quantum_kernel.evaluate(x_vec=X)
            self.ridge.fit(K_train, y)
            return self

        def predict(self, X):
            K_test = self.quantum_kernel.evaluate(x_vec=X)
            return self.ridge.predict(K_test)

class QTSP:

        def __init__(self):
            self.provider = IBMProvider()
            self.qbackend = least_busy(self.provider.backends(filters=lambda x: x.configuration().n_qubits >= 5 and not x.configuration().simulator and x.status().operational == True))

        def read_csv_data(self, csv_path):
            df = pd.read_csv(csv_path)
            return df

        def preprocess_data(self, df, is_training=True):
            if is_training:
                self.feature_names = df.columns[:-1]
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(df.iloc[:, :-1])
                X = scaled_data
                y = df.iloc[:, -1].values
            else:
                X = self.scaler.transform(df)
                y = None
            if is_training:
                return train_test_split(X, y, test_size=1)
            else:
                return X

        def train_quantum_model(self, X_train, y_train):
            quantum_instance = QuantumInstance(self.qbackend, shots=1024, seed_simulator=42, seed_transpiler=42)
            qkernel = QuantumKernel(quantum_instance=quantum_instance)
            qr = QKRR(quantum_kernel=qkernel)
            qr.fit(X_train, y_train)
            self.qridge = qr

        def predict(self, new_data):
            if len(new_data) != len(self.feature_names):
                logging.error(f"Invalid data provided for prediction. Expected {len(self.feature_names)} features.")
                return None
            scaled_data = self.scaler.transform([new_data])
            prediction = self.qridge.predict(scaled_data)
            combined_data = np.hstack((scaled_data, prediction.reshape(-1, 1)))
            final_prediction = prediction[0]
            return final_prediction
        
        def batch_predict(self, batch_data):
            return [self.predict(data_point) for data_point in batch_data]

        def save_output_to_csv(self, actual_new_data, predictions, csv_path):
            df = pd.DataFrame({'input_data': actual_new_data.tolist(), 'predicted_wind_speed': predictions})
            df.to_csv(csv_path, index=False)

if __name__ == '__main__':
        IBMProvider.save_account('', overwrite=True)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        predictor = QTSP()
        df = predictor.read_csv_data('input_data.csv')
        X_train, _, y_train, _ = predictor.preprocess_data(df)
        predictor.train_quantum_model(X_train, y_train)
        df_new = predictor.read_csv_data('new_data.csv')
        X_new = predictor.preprocess_data(df_new, is_training=False)
        predictions = predictor.batch_predict(X_new)
        predictor.save_output_to_csv(X_new, predictions, 'predicted_wind_speeds.csv')
