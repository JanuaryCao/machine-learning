# compare_models.py
import time
 
import numpy as np
 
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
 
 
models = ['resnet50_saved_model', 'resnet50_saved_model_TFTRT_FP32']
 
for model in models:
    saved_model = tf.keras.models.load_model(model)
    infer = saved_model.signatures['serving_default']
 
    batch_size = 8
    batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
 
    for i in range(batch_size):
        img_path = 'data/img%d.jpg' % (i % 4)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batched_input[i, :] = x
    batched_input = tf.constant(batched_input)
 
    N_warmup_run = 50
    N_run = 1000
    elapsed_time = []
 
    for i in range(N_warmup_run):
        preds = infer(batched_input)
 
    for i in range(N_run):
        start_time = time.time()
        preds = infer(batched_input)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
 
    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
