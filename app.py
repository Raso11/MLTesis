from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Carga de los modelos
gb_model = joblib.load("models/best_parkinsons_model.joblib")  # Gradient Boosting
cnn_model = load_model("models/parkinson_spiral_cnn_82_f1.keras")  # Modelo CNN

# Ruta de predicción
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Datos en formato JSON
        
        # Obtiene características para el modelo Gradient Boosting
        gb_features = np.array(data["gb_features"]).reshape(1, -1)
        
        # Predicción del modelo Gradient Boosting
        gb_prediction = gb_model.predict_proba(gb_features)[0][1]  # Probabilidad de clase positiva
        
        # Obtiene características para el modelo CNN
        cnn_features = np.array(data["cnn_features"])
        cnn_features = pad_sequences([cnn_features], maxlen=128)  # Ajusta según el modelo
        
        # Predicción del modelo CNN
        cnn_prediction = cnn_model.predict(cnn_features)[0][0]  # Probabilidad de clase positiva
        
        # Fusión tardía (por ejemplo, promedio)
        final_prediction = (gb_prediction + cnn_prediction) / 2
        
        # Retorna la predicción
        return jsonify({"gb_prediction": float(gb_prediction),
                        "cnn_prediction": float(cnn_prediction),
                        "final_prediction": float(final_prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
