from langchain_nvidia_ai_endpoints import ChatNVIDIA
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time
from datetime import datetime

output_file = "result_file.txt"
time_file = "time_output.txt"
input_file = "NVIDIA/Sample_Prompts_100.csv"

def getNVIDIAModelAccess():
  client = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key="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", 
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
  )
  return client

def read_csv(filename):
  df = pd.read_csv(filename)
  data_dict = dict(zip(df.iloc[:,0], df.iloc[:,1]))
  return data_dict

def get_inference_with_nvidia(num, question):
  print("Get inference with NVIDIA", num)
  audit_time = ""
  try:
    model = getNVIDIAModelAccess()
    timestamp_before_inference = time.time()
    response = model.invoke(question)
    timestamp_after_inference = time.time()
    delta_time = timestamp_after_inference - timestamp_before_inference
    audit_time += "timestamp_before_inference :: " + str(timestamp_before_inference)
    audit_time += " timestamp_after_inference :: " + str(timestamp_after_inference)
    audit_time += " total inference time :: " + str(delta_time)
  except Exception as e:
    print("error in getting answer of question :: ", question)
    print(e)
    return num, "error in getting answer of question :: ", question
  return num, audit_time, response

def start_parallel_inferece(file_name):
  questions_dict = read_csv(filename=file_name)
  futures = []
  timeseries = {}
  
  pool = ThreadPoolExecutor(max_workers=10)
  if questions_dict is not None:
    for num, question in questions_dict.items():
      result = pool.submit(get_inference_with_nvidia, num, question)
      timeseries[num] = datetime.now().time()
      futures.append(result)

    for future in as_completed(futures):
      num, audit_time, response = future.result()
      content = "" + audit_time + " Qno. " + str(num) + " Qsn. " + questions_dict.get(num) 
      with open(output_file, "a") as f:
        f.write(content)
        f.write("\n")
      f.close()
  else:
    print("File Is not readbale")
  if timeseries is not None:
    for key, value in timeseries.items():
      print(str(key) + "  " + str(value))
      with open(time_file, "a") as t:
        t.write(str(key) + "  " + str(value))
        t.write("\n")
    t.close()


start_parallel_inferece(file_name=input_file)


