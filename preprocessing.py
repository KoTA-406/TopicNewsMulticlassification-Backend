# Load Libraries
import re

def clean(text):     
  text = text.strip()              
  text = re.sub(r'https?://\S+|www\.\S+', '', text)           
  text = re.sub(r'[^\w\s]','', text) #hapus simbol
  text = ''.join([x for x in text if not x.isdigit()]) #hapus angka
  return text

def preprocessing(data):
  for i in range (len(data)): 
    data[i] = str(data[i]).lower() 
    data[i] = str(clean(data[i]))
  return data

def preprocessing_str(data):
  data = data.lower() 
  data = str(clean(data))
  return data