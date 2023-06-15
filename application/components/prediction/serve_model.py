import numpy as np
from PIL import Image
from io import BytesIO
import joblib as jb

height,width = (216,216)

def load_model():
    model = jb.load('/home/victor/Desktop/chest_xray_project/application/components/prediction/ml_models/second_model.joblib')
    return model

def decode_pred(pred_val):
    if pred_val < 0.5:
        return 'NORMAL'
    else:
        return 'PNEMONIA'
    
def predict(image: Image.Image):
    image = np.asarray(image.resize((height, width)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image * (1/255.)
    
    model = load_model()
    
    result = decode_pred(model.predict(image)[0][0])
    
    return result

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert('RGB')
    return image