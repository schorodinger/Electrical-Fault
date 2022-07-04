import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('C:/Users/LUCAS/electrical_fault/data/raw/classData.csv')
df_og = df.copy()

df['faultCode'] = df['G'].astype('str') + df['C'].astype('str') + df['B'].astype('str') + df['A'].astype('str')
#

df = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc', 'faultCode']]


nf = df[df['faultCode']=='0000'].reset_index(drop=True)
lg = df[df['faultCode']=='1001'].reset_index(drop=True)
ll = df[df['faultCode']=='0110'].reset_index(drop=True)
llg = df[df['faultCode']=='1011'].reset_index(drop=True)
lll = df[df['faultCode']=='0111'].reset_index(drop=True)
lllg = df[df['faultCode']=='1111'].reset_index(drop=True)

dfs_fault = [nf, lg, ll, llg, lll, lllg]
c = 0

for val in dfs_fault:
  if c in range(3, 6):
    val.drop(range(1000, len(val)), inplace=True)
  c += 1
  val.drop(range(0, 200), inplace=True)


new_df = pd.concat(dfs_fault)
new_df['faultCode'] = new_df['faultCode'].replace('1111', '0111')
new_df.to_csv('data/processed/df_cut_unif.csv', index=False)