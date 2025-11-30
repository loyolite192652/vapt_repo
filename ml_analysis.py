# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# FILE : ml_analysis . py
# PURPOSE : Nmap XML Parsing , Mock Supervised Prediction ,
# Unsupervised Anomaly Detection , and Final Reporting .
# EXECUTION : python3 ml_analysis . py -- xml scan_results . xml
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
import xmltodict
import pandas as pd
from sklearn . ensemble import IsolationForest
from sklearn . preprocessing import LabelEncoder
import numpy as np
import os
import argparse
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# STEP 1: DATA INGESTION AND FEATURE ENGINEERING
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def step_ 1_ da ta_ ingestion ( xml_file_path ) :
" " " Parses XML , extracts features , structures data , and encodes it . " " "
print ( " \n - - - Step 1: Data Ingestion and Feature Engineering ---" )
try :
with open ( xml_file_path , ’r ’) as f :
data_dict = xmltodict . parse ( f . read () )
except Exception as e :
print ( f " [ FATAL ERROR ] Could not read or parse XML file : { e } " )
return pd . DataFrame () , pd . DataFrame ()
extracted_data = []
ports = (
data_dict . get ( ’ nmaprun ’ , {})
. get ( ’ host ’ , {})
. get ( ’ ports ’ , {})
. get ( ’ port ’ , [])
)
if not isinstance ( ports , list ) :
ports = [ ports ]
for port in ports :
# Include only open ports
if port . get ( ’ state ’ , {}) . get ( ’ @state ’) == ’ open ’:
service = port . get ( ’ service ’ , {})
extracted_data . append ({
’ Port_ID ’: int ( port . get ( ’ @portid ’) ) ,
’ Protocol ’: port . get ( ’ @protocol ’) ,
’ Service_Name ’: service . get ( ’ @name ’ , ’ unknown ’) ,
’ Service_Version ’: service . get ( ’ @version ’ , ’ unknown ’) ,
})
df_features = pd . DataFrame ( extracted_data )
if df_features . empty :
print ( " [ INFO ] No open ports found for analysis . " )
return df_features , pd . DataFrame ()
df_encoded = pd . get_dummies (
df_features ,
columns =[ ’ Service_Name ’ , ’ Protocol ’] ,
prefix =[ ’ Service ’ , ’ Proto ’]
)
le = LabelEncoder ()
df_encoded [ ’ Service_Version_Encoded ’] = le . fit_transform (
df_features [ ’ Service_Version ’ ]. astype ( str )
)
X_test = df_encoded . select_dtypes ( include =[ np . number ]) . drop ( columns =[ ’ Port_ID ’ ])
print ( f " [ SUCCESS ] DataFrame created with { len ( df_features ) } records . " )
print ( f " [ SUCCESS ] Feature Matrix ( X_test ) ready with shape : { X_test . shape } " )
return df_features , X_test
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# STEP 2: MOCK SUPERVISED PREDICTIVE SCORING
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def s t e p _ 2 _ p r e d i c tive_s coring ( df_features , X_test ) :
" " " Simulated vulnerability scoring . " " "
print ( " \n - - - Step 2: Predictive Vulnerability Scoring ( MOCK ) ---" )
def mock_predict ( row ) :
service = row [ ’ Service_Name ’ ]. lower ()
version = row [ ’ Service_Version ’]
if ’ ftp ’ in service or ’ telnet ’ in service or ’ 0.93 ’ in version :
return 3
elif ( ’ http ’ in service and ’ 2. ’ in version ) or ( ’ ssh ’ in service and ’ 7. ’ in version ) :
return 2
elif ’ https ’ in service and ’ unknown ’ in version :
return 1
else :
return 0
df_features [ ’ Risk_Score ’] = df_features . apply ( mock_predict , axis =1)
risk_map = {0: ’ Low ’ , 1: ’ Medium ’ , 2: ’ High ’ , 3: ’ CRITICAL ’}
df_features [ ’ Risk_Level ’] = df_features [ ’ Risk_Score ’ ]. map ( risk_map )
print ( " [ SUCCESS ] Simulated Supervised Learning prediction completed . " )
return df_features
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# STEP 3: ANOMALY DETECTION & PRIORITIZATION
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def s t e p _ 3 _ a n o m a ly_detection ( df_features , X_test ) :
" " " Runs Isolation Forest to identify anomalies . " " "
print ( " \n - - - Step 3: Anomaly Detection and Prioritization ---" )
if len ( X_test ) < 2:
print ( " [ WARNING ] Not enough data points for anomaly detection . Skipping . " )
df_features [ ’ Anomaly_Flag ’] = 1
df_features [ ’ Anomaly_Status ’] = ’ Normal ’
else :
contamination_rate = max (0.01 , min (0.49 , 1 / len ( X_test ) ) )
iso = IsolationForest ( contamination = contamination_rate , random_state =42)
df_features [ ’ Anomaly_Flag ’] = iso . fit_predict ( X_test )
df_features [ ’ Anomaly_Status ’] = df_features [ ’ Anomaly_Flag ’ ]. apply (
lambda x : ’ ANOMALY ’ if x == -1 else ’ Normal ’

)
final_results = df_features [
( df_features [ ’ Risk_Level ’] == ’ CRITICAL ’) |
( df_features [ ’ Anomaly_Status ’] == ’ ANOMALY ’)
]
print ( " [ SUCCESS ] Unsupervised Anomaly Detection completed . " )
print ( f " [ SUCCESS ] Identified { len ( final_results ) } priority items . " )
return final_results
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# STEP 4: OVERALL VULNERABILITY SCORE
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def c a l c u l a t e _ v u l n e r a b i l i t y _ p e r c e n t a g e ( df_features ) :
" " " Calculates overall vulnerability percentage . " " "
if df_features . empty :
return 0.0 , 0
risk_weights = { ’ CRITICAL ’: 3 , ’ High ’: 2 , ’ Medium ’: 1 , ’ Low ’: 0}
df_features [ ’ Weighted_Score ’] = df_features [ ’ Risk_Level ’ ]. map ( risk_weights )
total_open_ports = len ( df_features )
total_risk = df_features [ ’ Weighted_Score ’ ]. sum ()
max_score = total_open_ports * 3
vuln_percent = ( total_risk / max_score ) * 100 if max_score > 0 else 0
high_priority_count = len ( df_features [ df_features [ ’ Risk_Score ’] >= 2])
return round ( vuln_percent , 2) , high_priority_count
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# MAIN EXECUTION
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
if __name__ == " __main__ " :
parser = argparse . ArgumentParser ()
parser . add_argument ( " -- xml " , help = " Path to Nmap XML file " , required = True )
args = parser . parse_args ()
xml_filename = args . xml
if not os . path . exists ( xml_filename ) :
print ( f " [ FATAL ERROR ] XML file ’{ xml_filename } ’ not found . " )
exit ()
df_features , X_test = step_1_data_ingestion ( xml_filename )
if df_features . empty :
print ( " \ nAnalysis terminated : No open ports found . " )
exit ()
df_features = s tep_2_ predic tive_s coring ( df_features , X_test )
final_report = step_3_anomaly_detection ( df_features , X_test )
vuln_percent , high_priority_count = c a l c u l a t e _ v u l n e r a b i l i t y _ p e r c e n t a g e ( df_features )
print ( " \ n \ n # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # " )
print ( " ### FINAL AI - ASSISTED VULNERABILITY REPORT ### " )
print ( " # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # " )
print ( f " OVERALL VULNERABILITY SCORE : { vuln_percent }% ( Weighted ) " )
print ( f " HIGH / CRITICAL ITEMS IDENTIFIED : { high_priority_count } out of { len ( df_features ) } open ports " )
print ( " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " )
if not final_report . empty :
print ( " HIGH PRIORITY ITEMS ( CRITICAL RISK OR ANOMALY ) : " )
report_display = final_report [[
’ Port_ID ’ ,
’ Service_Name ’ ,
’ Service_Version ’ ,
’ Risk_Level ’ ,
’ Anomaly_Status ’
]]. sort_values ( by =[ ’ Risk_Level ’ , ’ Anomaly_Status ’] , ascending =[ False , True ])
print ( report_display . to_markdown ( index = False ) )
else :
print ( " [ REPORT ] No critical or anomalous services detected . " )
 
