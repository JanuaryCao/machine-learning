# save_model.py
import numpy as np
 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
 
# load model
model = ResNet50(weights='imagenet')
 
for i in range(4):
  img_path = './data/img{}.jpg'.format(i)
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
 
  # prediction
  preds = model.predict(x)
 
  print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
 
# save model to resnet50_saved_model path
model.save('resnet50_saved_model') 
