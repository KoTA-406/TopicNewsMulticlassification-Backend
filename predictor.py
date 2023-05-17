# Load Libraries.
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModel
from ast import literal_eval
import json
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
  # model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
  return model_loaded 

def create_input_ids(sentences):
  # max_length = 32

  # Tokenize input sentences
  tokenized_res = [tokenizer.tokenize(sentence) for sentence in sentences]
  # print("Tokenized Sentences:", tokenized_res)
  
  # Encode input sentences
  tokenized_sentences = [tokenizer.encode(sentence, add_special_tokens=True)[:512] for sentence in sentences]
  
  # Add padding to input IDs
  max_length = max([len(ids) for ids in tokenized_sentences])
  input_ids = [ids + [0] * (max_length - len(ids)) for ids in tokenized_sentences]
  # print("Padded Input IDs:", input_ids)

  return input_ids

def word_embed(arr_data):
  max_length = 32
  x_input_ids = create_input_ids(arr_data)
  x_input_ids_padded = pad_sequences(x_input_ids, maxlen=max_length, padding='post')

  with torch.no_grad():
      x_clean_embed = model_BERT(np.array(x_input_ids_padded ))[0]
  return x_clean_embed

def predict_single_data(news_title):
  x_clean = preprocessing.preprocessing_str(news_title)
  x_clean = [x_clean]
  x_clean_embed = word_embed(x_clean)
  prediksi = model_BCL.predict(x_clean_embed)
  
  topik = "Tidak termasuk ke topik manapun"
  # Cetak presentase hasil prediksi untuk setiap kelas
  for i in range(len(prediksi[0])):
    if (prediksi[0][i]>0.5):
      topik = class_names[i]
  return topik

def predict_data_collection(arr_news_title):
  # print(arr_news_title)
  # arr_news_title = json.loads(arr_news_title)
  # Remove the opening and closing brackets and split the string by whitespace
  elements = arr_news_title.split('\n')

  # Convert the elements into a list
  array_data = [element.strip("'") for element in elements]
  x_clean = preprocessing.preprocessing(array_data)
  x_clean_embed = word_embed(x_clean)

  predicted_result = model_BCL.predict(x_clean_embed)  

  labels = []
  i = 0
  while (i < len(predicted_result)):
    topik = class_names[8]
    for j in range(len(predicted_result[i])):
      if (predicted_result[i][j]>0.5):
        topik = class_names[j]
    labels.append(topik)
    i = i + 1

  print(array_data)

  return str(labels)


