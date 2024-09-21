import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import pickle
import tempfile  # Biblioteca para crear archivos temporales
import shutil 
from typing import Optional
from typing import ClassVar
from sklearn.linear_model import Ridge
from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

# Crear una instancia de FastAPI
app = FastAPI()

# Cargar el modelo preentrenado desde el archivo pickle
#model_path = "best_model.pkl"
with open("modelo_ridge.pkl", 'rb') as model_file:
    dt2 = pickle.load(model_file)

prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")


# Definir un endpoint para manejar la subida de archivos Excel y hacer predicciones
@app.post("/upload-excel")
def upload_excel(file: UploadFile = File(...)):
    try:
        # Crear un archivo temporal para manejar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)

            # Leer el archivo Excel usando pandas y almacenarlo en un DataFrame
            df = pd.read_excel(temp_file.name)
            
            df_test = df.copy()
            predictions = predict_model(dt2, data=df_test)
            predictions["price"] = predictions["prediction_label"]
            prediction_label = list(predictions["price"])

            return {"predictions": prediction_label}
        
    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}
    

# Ejecutar la aplicación FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)







