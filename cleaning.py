import pandas as pd

'''metadata cleaning'''
df1 = pd.read_excel('QG_DS_ML_hiringChallenge.xlsx', sheet_name=[1], nrows=46, usecols="K:MX")
p1_df = pd.DataFrame(list(df1.items())[0][1], index=None)
p1_df = p1_df.T
header = p1_df.iloc[0]
p1_df = p1_df[1:]
p1_df.columns = header
p1_df = p1_df.drop(['SEQUENCE COUNTS'], axis=1)
# print(p1_df.columns)

'''Gene sparse data cleaning'''
df2 = pd.read_excel('QG_DS_ML_hiringChallenge.xlsx', sheet_name=[1], skiprows=47, index_col=0, usecols="A:MX",
                    skipfooter=48)
p2_df = pd.DataFrame(list(df2.items())[0][1], index=None)
drop_labels = [*range(0, 10)]
p2_df = p2_df.drop(p2_df.columns[drop_labels], axis=1)
p2_df = p2_df.T
# print(p2_df.columns)

# df3 = pd.read_excel('QG_DS_ML_hiringChallenge.xlsx', sheet_name=[2], index_col=0, usecols="A:NC",header=None)
# p3_df = pd.DataFrame(list(df3.items())[0][1],index=None)
# p3_df = p3_df.drop(p3_df.columns[drop_labels], axis=1)
# p3_df = p3_df.T
#
# p3_df.set_index('Gene',append=True,inplace=True)
# p3_df = p3_df.loc[(p3_df!=0).all(axis=1)]
# p3_df.set_index(p3_df.iloc[:,0],append=True,inplace=True)
# p3_df.reset_index(level=0,inplace=True)
# p3_df.index.names=['ID','Batch number']
# p3_df = p3_df.reorder_levels(['Batch number', 'ID'])
# p3_df.drop(p3_df.iloc[:,0:2],axis=1, inplace=True)
# p3_df.dropna(how='any',inplace=True)
#
# print(p3_df)


df = pd.concat([p1_df, p2_df], axis=1)
df.index.names = ['ID']
df.set_index('Batch number', append=True, inplace=True)
df = df.reorder_levels(['Batch number', 'ID'])
df = df[(df['Cancer type']=='Daisy') | (df['Clinical diagnosis']=='CONTROL' )]
#

# df = df.truediv(p3_df, fill_value=0)

df = pd.get_dummies(df, columns=['Gender', 'Ancestry', 'Clinical diagnosis', 'Cancer type'],
                    prefix=['Gender', 'Ancestry', 'Clinical diagnosis', 'Cancer type'], dummy_na=True)
df.rename(columns={"Cancer type_nan": "Cancer type_control"}, inplace=True)
drop_cols = ['Source', 'Date collected', 'Date collected', 'Date sequenced', 'Date of extraction',
             'Location of extraction', 'Flow cell name', 'Bucket version', 'Bucket number', 'Gender_nan',
             'Ancestry_nan', 'Clinical diagnosis_nan']
df.drop(drop_cols, axis=1, inplace=True)
df.to_csv('dataset.csv', index=False)

