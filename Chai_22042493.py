########## INSTRUCTIONS ##########
# 1. Only add or modify codes within the blocks enclosed with
#    ########## student's code ##########
#
#    ####################################
##################################


########## student's code ##########
# If you need to import any library, you may do it here
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
####################################

def student_details():
  ########## student's code ##########
  ##########    Task 1     ##########
  # 1. Update the name and id to your name and student id
  studentName = "Chai Yi Xiang"
  studentId = "22042493"
  ####################################
  return studentName, studentId

def load_file():
  ########## student's code ##########
  ##########    Task 2     ##########
  # 1. load the csv file to be a pandas DataFrame with vairable name: "df"
  #    (note that the csv file has no header)
  #    add/set the headers of the columns (from left to right) to be
  #        cache, channelmin, channelmax, publishedperformance, estimatedperformance
  try:
    df = pd.read_csv('dataset.csv', header=None)
    df.columns = ['cache', 'channelmin', 'channelmax', 'publishedperformance', 'estimatedperformance']
    print("Data loaded")
    return df
  except FileNotFoundError:
    print("File loading error")
    return None
  ####################################  
  return df

def train_clustering_model(df):
  ########## student's code ##########
  ##########    Task 3     ##########  
  # 1. initialise a kmeans model with 5 clusters using variable name: "kmModel"
  # 2. train the kmeans model using the following columns from df
  #      publishedperformance, estimatedperformance
  kmModel = KMeans(n_clusters=5, random_state=42)
  kmModel.fit(df[['publishedperformance', 'estimatedperformance']])
  print("Cluster model training successful")
  ####################################
  return kmModel

def test_clustering_model(df, kmModel):
  ########## student's code ##########
  ##########    Task 4     ##########  
  # 1. use any 10 rows from df and identify/predict their clusters
  # 2. save the identified cluster index with variable name: "outcome"
  sample = df.sample(n=10)
  X = sample[['publishedperformance', 'estimatedperformance']]
  outcome = kmModel.predict(X)  
  print(f"Clustering outcome: {outcome}")
  ####################################
  return outcome

def add_clustering_result_to_data(df, kmModel):
  ########## student's code ##########
  ##########    Task 5     ##########  
  # 1. predict the clusters of every row in df
  # 2. convert the cluster numbers (0,1,2,3,4,5) to alphabets (a,b,c,d,e)
  # 3. add the cluster outcome as a new column called "cresult"
  clusters = kmModel.predict(df[['publishedperformance', 'estimatedperformance']])
  mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}
  df['cresult'] = [mapping.get(c, 'e') for c in clusters]
  print("Cluster number converted and added as 'cresult'")
  ####################################
  return df

def train_decision_tree(df):
  ########## student's code ##########
  ##########    Task 6     ##########  
  # 1. initialise a decision tree model with maximum depth of 5 as variable: dtModel
  # 2. train the decision tree model to classify based on inputs of
  #    a. channelmin    
  #    b. channelmax
  #    to identify the output of "estimatedperformance"
  X = df[['channelmin', 'channelmax']]
  y = df['estimatedperformance']
  dtModel = DecisionTreeClassifier(max_depth=5)
  dtModel.fit(X, y)  
  print("Decision tree training successful")
  ####################################
  return dtModel

def test_decision_tree(df, dtModel):
  ########## student's code ##########
  ##########    Task 7     ##########  
  # 1. predict the class using the trained decision tree
  # 2. add the predicted outcome as a new column called "dresult"
  X = df[['channelmin', 'channelmax']]
  df['dresult'] = dtModel.predict(X)
  print("Decision tree trained")
  ####################################
  return df

def save_to_file(df):
  ########## student's code ##########
  ##########    Task 8     ##########  
  # 1. save the dataframe "df" to a csv file with the name of "finalresults.csv"
  try:
      df.to_csv("finalresults.csv", index=False)
      print("File saved to finalresults.csv")
  except PermissionError:
      print("Unable to save, no file permission")
      print("CLose the file if it is opened")
  ####################################

if __name__ == "__main__": 
  print("Only add or modify codes within the blocks enclosed with")
  print("########## student's code ##########")
  print("")
  print("####################################")
  print("DO NOT REMOVE OR MODIFY CODES FROM OTHER SECTIONS")
  print("")

  sname,sid = student_details()
  print(f"You are {sname} with student ID {sid}")  

  
  ########## student's code ##########
  # you do not need to change the code of this section
  # but you may modify the code for debugging purpose
  df = load_file()
  kmModel = train_clustering_model(df)
  results = test_clustering_model(df, kmModel)
  df = add_clustering_result_to_data(df, kmModel)

  dtModel = train_decision_tree(df)
  df = test_decision_tree(df, dtModel)
  save_to_file(df)
  ####################################
