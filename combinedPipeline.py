from NaiveBayes import NaiveBayes
from RandomForest import RandomForestOutput
from LogisticRegressionFunc import logisticRegressionValue
from SVM import SVMoutput
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from createdf import is_english


df = pd.read_excel('concatenated_file3.xlsx')


df['is_english'] =df['tweets'].apply(is_english)
df = df[df['is_english']].drop(columns=['is_english'])
df.reset_index(drop=True, inplace=True)
df['tweets'] = df['tweets'].astype(str)

def returnActionStatus(string):
    
    a = NaiveBayes(string)
    b = RandomForestOutput(string)
    c = logisticRegressionValue(string)
    d = SVMoutput(string)
    
   # print(a,b,c,d)
    avg = (a+b+c+d)/4
    
    #print(avg, string)
    if (avg>=0.5):
        return 1
    if (avg<0.5):
        return 0
    


df['predicted_action'] = df['tweets'].apply(returnActionStatus)


false_positives = df[(df['Action'] == 0) & (df['predicted_action'] == 1)]
false_negatives = df[(df['Action'] == 1) & (df['predicted_action'] == 0)]



accuracy = accuracy_score(df['Action'], df['predicted_action'])
print(f"Accuracy: {accuracy}")
f1 = f1_score(df['Action'], df['predicted_action'])
print(f'F1 Score: {f1}')
conf_matrix = confusion_matrix(df['Action'], df['predicted_action'])
print('Confusion Matrix:')
print(conf_matrix)
class_report = classification_report(df['Action'], df['predicted_action'])
print('Classification Report:')
print(class_report)

#df.to_excel('testResults.xlsx', index = False)

#I have made complaint on 112 for accident case of my father at 4:04 AM with complaint number 9198792 no pcr patrol van even visited yet.This is Delhi I cannot even believe @DelhiPolice [0,0,1,1] > 1
# am so fed up, I am not getting any solution since last 2 months mob:- 8708715041 [1, 0, 1, 1] > 1
#Dear a fraud call has Been circulating thru mumbai High court. Kindly act asap. Phone nos 9048191468.. follow up call comes from 9642761458. Request to pls act asap. [0,1,1,1] > 1
#Absolute nuisance of loud music from chembur camp area from a band being heard in all surrounding areas. It's a working day. People have a right to sleep. Children and senior citizens are disturbed. How is the police allowing this nuisance. [1,0,0,0] >1
#Check accuracy of NB and Random Forest - optimize them

