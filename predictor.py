# Load Libraries.
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import torch
import preprocessing


# Define / Initialization from Model.
DEVICE = torch.device("cpu")
pretrain_model_name = "indobenchmark/indobert-base-p2"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name )
model_BERT = TFAutoModel.from_pretrained("indobenchmark/indobert-base-p2")
model_path = './models/model_baseline_bert-cnn.h5'
model_BCL = load_model(model_path)

class_names = ['budaya', 'ekonomi', 'kesehatan', 'olahraga', 'otomotif',
               'pertahanan dan keamanan', 'politik', 'teknologi', 'none']

# Function Load Model.
def load_model(path):
  model_loaded = load_model(path)
  return model_loaded 

# Function to transform string to Input ID.
def create_input_ids(sentences):
  # Encode input sentences
  tokenized_sentences = [tokenizer.encode(sentence, add_special_tokens=True)[:512] for sentence in sentences]
  
  # Add padding to input IDs
  max_length = max([len(ids) for ids in tokenized_sentences])
  input_ids = [ids + [0] * (max_length - len(ids)) for ids in tokenized_sentences]

  return input_ids

# Function to transform Input ID to Embedding Vector.
def word_embed(arr_data):
  # equals the length of the input id
  max_length = 32
  x_input_ids = create_input_ids(arr_data)
  x_input_ids_padded = pad_sequences(x_input_ids, maxlen=max_length, padding='post')

  # transform to embedding vector
  with torch.no_grad():
      x_clean_embed = model_BERT(np.array(x_input_ids_padded ))[0]
  return x_clean_embed

# Function to predict 1 data/news title.
def predict_single_data(news_title):
  # preprocessing data
  x_clean = preprocessing.preprocessing_str(news_title)
  x_clean = [x_clean]

  # transform to embedding vector
  x_clean_embed = word_embed(x_clean)

  # predict
  prediksi = model_BCL.predict(x_clean_embed)
  
  topik = "Tidak termasuk ke topik manapun"

  # choose the class by predicted percentage from each class
  for i in range(len(prediksi[0])):
    if (prediksi[0][i]>0.5):
      topik = class_names[i]
  return topik

# Function to predict file that contain > 1 data.
def predict_data_collection(arr_news_title):
  # split the string by whitespace
  elements = arr_news_title.split('\n')

  # convert the elements into a list
  # remove opening and closing quotation marks
  array_data = [element.strip("'") for element in elements]

  # preprocessing data
  x_clean = preprocessing.preprocessing(array_data)

  # transform to embedding vector
  x_clean_embed = word_embed(x_clean)

  # predict 
  predicted_result = model_BCL.predict(x_clean_embed)  

  # get the label from each data
  labels = []
  i = 0
  while (i < len(predicted_result)):
    topik = class_names[8]
    x_arr = array_data[i].split()
    if(not(len(x_arr) < 3) and not(len(x_arr) > 18)):
      for j in range(len(predicted_result[i])):
        if (predicted_result[i][j]>0.5):
          topik = class_names[j]
    labels.append(topik)
    i = i + 1

  print(array_data)

  return str(labels)

# End of Line - Predictor.py #