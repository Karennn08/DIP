import radiomics
import os

import radiomics.featureextractor

import pandas as pd

#prueba de verificación de features


testCase = "RCOVID"

dataDir = r"C:\Users\karen\OneDrive\Escritorio\DIP\250403\Data"

#print(extractor.kwargs)
#print(extractor.inputImages)
#print(extractor.enabledFeatures)
feat_list = []
print("Sacando features")
paramPath = r"C:\Users\karen\OneDrive\Escritorio\DIP\250403\params.yaml"
#paramPath = os.path.join(os.getcwd(),"params.yaml")
#inicialización dle extractor
extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(paramPath)
print(extractor.enabledFeatures)


for i in range(1,5):
    imagePath = os.path.join(dataDir, "images/" + str(i) + ".png")
    labelPath = os.path.join(dataDir, "masks/" + str(i) + ".png")
    
    #print(imagePath)
    #print(labelPath)
    #print("Parameter File", "absolute path:", os.path.abspath(paramPath))

    featureVector = extractor.execute(imagePath,labelPath,255)
    #print(featureVector['original_firstorder_Median'])
    feat_list.append(featureVector)


df = pd.DataFrame.from_dict(feat_list)

# Agregar una columna con los nombres o identificadores de las imágenes (opcional)
df['imagen_id'] = [f'imagen_{i+1}' for i in range(1,5)]

# Reordenar las columnas para que 'imagen_id' sea la primera columna (opcional)
df = df[['imagen_id'] + [col for col in df.columns if col != 'imagen_id']]
df.to_excel(r'C:\Users\karen\OneDrive\Escritorio\DIP\250403\caracteristicas_tumores.xlsx', index=True)