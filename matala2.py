import pandas as pd 
import numpy as np
import seaborn as sns 
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO  
from IPython.display import Image  
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from datetime import date
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
pip install mlxtend
pip install WordCloud
pip freeze requiments
df= pd.read_csv("corona_india.csv", encoding='cp1252')
d=pd.read_csv("corona_india.csv", encoding='cp1252') 
#***********************info and cleaning the data **********************************************
df.info()
df.describe()
df.shape
df.drop_duplicates(inplace=True)

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df.drop(cols_to_drop, axis=1,inplace=True)

pct_null = df.isnull().sum() / len(df)
missing_features = pct_null[pct_null > 0.60].index
df.drop(missing_features, axis=1, inplace=True)
    
df.dropna(inplace=True)
#***** drop rows without full text like(????,Z,R)****
df = df.drop(labels=[245,265,270,291,314,322,330,389,428,485,533,570,381,419,475], axis=0)
df = df.drop(labels=[6,8,9], axis=0)

#*******************************normalization************************************************
#***normalization to the column-Out of the following, choose 5 tools which you use most often these days **
df[['Tool_1', 'Tool_2', 'Tool_3','Tool_4','Tool_5']] = df['Out of the following, choose 5 tools which you use most often these days.'].str.split(',', expand=True)
#***normalization to the column-Other than attending online classes for your course, what are the 5 activities that you mostly indulge in during this period? **
df[['activity_1', 'activity_2', 'activity_3','activity_4','activity_5']] = df['Other than attending online classes for your course, what are the 5 activities that you mostly indulge in during this period? '].str.split(',', expand=True)
df[['Date', 'time']] = df['Timestamp'].str.split(' ', expand=True)
## drop the original columns*********
df.drop(['Out of the following, choose 5 tools which you use most often these days.' ,'Other than attending online classes for your course, what are the 5 activities that you mostly indulge in during this period? ','Timestamp'], axis=1, inplace=True)

#**** moved y to the front*****
col = df.pop("Your One Line message to the World during the Lockdown !!")
df.insert(0, col.name, col)

#************order the avarege of updating on covid********
df['Average number of hours per day that you spend on updating yourself on Covid-19 related news?'].unique()
df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"]=df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"].replace(["30 minutes to 2 hrs"],"1.25")#avg im hrs
df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"]=df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"].replace(["Less than 30 minutes"],"0.5")
df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"]=df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"].replace(["More than 4 hrs"],"4")
df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"]=df["Average number of hours per day that you spend on updating yourself on Covid-19 related news?"].replace(["2 to 4 Hrs"],"3")

df['Average number of hours per day that you spend on updating yourself on Covid-19 related news?'].unique()
#************order the avarege of time in the media********
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?'].unique()
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["30 minutes to 2 hrs"],"1.25")
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["2 to 4 Hrs"],"3")
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["More than 12 hrs"],"12")
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["6 to 8 Hrs"],"7")
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["8 to 10 Hrs"],"9")
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["10 to 12 Hrs"],"11")
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["4 to 6 Hrs"],"5")
df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?']=df["Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?"].replace(["Less than 30 minutes"],"0.5")

df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?'].unique()

#************order the avarege of social media********
df['Other than for academic activities, how much time on an average do you spend per day on social media apps these days?'].unique()
df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"]=df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"].replace(["30 minutes to 2 hrs"],"1.25")#avg im hrs
df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"]=df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"].replace(["Less than 30 minutes"],"0.5")
df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"]=df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"].replace(["More than 6 hrs"],"6")
df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"]=df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"].replace(["2 to 4 Hrs"],"3")
df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"]=df["Other than for academic activities, how much time on an average do you spend per day on social media apps these days?"].replace(["4 to 6 Hrs"],"5")

df['Other than for academic activities, how much time on an average do you spend per day on social media apps these days?'].unique()


#******calculate the age *********
today = date.today()
df["age"]=today.year-df["Year of Birth"]
df['age'].unique()

#*****cor check ***
x=df.iloc[:,1:45]
cor= x.corr().abs()
sns.heatmap(x.corr(), annot=True)
plt.show()
upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.975)]
df.drop(to_drop, axis=1, inplace=True)



csvname='corona_india'
csvname=csvname[:]+"_cleaned.csv"
df.to_csv(csvname,index=False)

Tool=df[["Tool_1","Tool_2","Tool_3","Tool_4","Tool_5"]]
Tv= Tool.values
Tl= Tv.tolist()

te = TransactionEncoder()
te_ary = te.fit(Tl).transform(Tl)
df_te = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df_te, min_support=0.6, use_colnames=True)
frequent_itemsets = apriori(df_te, min_support=0.6, use_colnames=True)
frequent_itemsets = fpmax(df_te, min_support=0.6, use_colnames=True)
frequent_itemsets

a=association_rules(frequent_itemsets, metric="confidence" , min_threshold=0.7)

Activ=df[["activity_1","activity_2","activity_3","activity_4","activity_5"]]
Ac_val=Activ.values
Ac_l=Ac_val.tolist()
te = TransactionEncoder()
te_ary = te.fit(Ac_l).transform(Ac_l)
df_te = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df_te, min_support=0.6, use_colnames=True)
association_rules(frequent_itemsets, metric="confidence" , min_threshold=0.7)

d["Other than attending online classes for your course, what are the 5 activities that you mostly indulge in during this period? "].unique()


#***********************************Vizualization**********************************************************************

sns.regplot(x="Gender", y="Do you find these classes too much burden in these difficult times?", data=df)
sns.regplot(x="Gender", y="Do you find these classes too much burden in these difficult times?", data=df)

sns.lmplot(x="Gender", y="Do you find these classes too much burden in these difficult times?", data=df, x_estimator=np.mean)
sns.lmplot(x="Gender", y="Do you verify the authenticity of the messages you forward on social media groups i.e. they are correct or have fake information?", data=df, x_estimator=np.mean)
sns.lmplot(x="Gender", y="Average number of hours per day that you spend on updating yourself on Covid-19 related news?", data=df, x_estimator=np.mean)
sns.catplot(x="Gender", y="Other than for academic activities, how much time on an average do you spend per day on social media apps these days?", kind="box", data=df)

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.bar(df['age'],df['Average number of hours per day that you spend on updating yourself on Covid-19 related news?'])
ax.set_title('age and avg of hs - conid 19')

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.bar(df['age'],df['Other than for academic activities, how much time on an average do you spend per day on social media apps these days?'])
ax.set_title('age and social media')

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.bar(df['age'],df['Other than for academic activities, how much time on an average do you spend per day on social media apps these days?'])
ax.set_title('age and time on smartphone and computer')

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.bar(df['Gender'],df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?'])
ax.set_title('avg')

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.bar(df['Studying in '],df['age'])
ax.set_title('avg')

sizes=[df['Gender'].tolist().count('Female'),df['Gender'].tolist().count('Male')]
labels=["Female","Male"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes_age=[df['age'].tolist().count(21),df['age'].tolist().count(25),df['age'].tolist().count(24),df['age'].tolist().count(23),df['age'].tolist().count(22),df['age'].tolist().count(26),df['age'].tolist().count(17),df['age'].tolist().count(16),df['age'].tolist().count(19),df['age'].tolist().count(18)]
age=["21","25","24","23","22","26","17","16","19","18"]
plt.pie(sizes_age, labels = age, autopct = None)
plt.axes().set_aspect("equal")
plt.show()


df['During this period do you have a smartphone/ computer for your exclusive use?'].unique()
sizes=[df['During this period do you have a smartphone/ computer for your exclusive use?'].tolist().count('Yes'),df['During this period do you have a smartphone/ computer for your exclusive use?'].tolist().count('No')]
labels=["Yes","No"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

df['Average number of hours per day that you spend on smartphone or computer (your screen time) during this lockdown?'].hist(bins=12, alpha=0.5)
df['During the period of the current COVID-19 lockdown, you are:'].unique()
sizes=[df['During the period of the current COVID-19 lockdown, you are:'].tolist().count('At Home with Family'),df['During the period of the current COVID-19 lockdown, you are:'].tolist().count('At Home but Alone'),df['During the period of the current COVID-19 lockdown, you are:'].tolist().count('In a Paying Guest'),df['During the period of the current COVID-19 lockdown, you are:'].tolist().count('Accommodation'),df['During the period of the current COVID-19 lockdown, you are:'].tolist().count('Other'),df['During the period of the current COVID-19 lockdown, you are:'].tolist().count('Stuck in Transit to Home')]
labels=["At Home with Family","At Home but Alone","In a Paying Guest","Accommodation","Other","Stuck in Transit to Home"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

df['Number of people you are staying with during this period'].unique()
df['Do you miss seeing your college and school friends in person? '].unique()
sizes=[df['Do you miss seeing your college and school friends in person? ']['Gender'='Female'].tolist().count(1),df['Do you miss seeing your college and school friends in person? '].tolist().count(2),df['Do you miss seeing your college and school friends in person? '].tolist().count(3),df['Do you miss seeing your college and school friends in person? '].tolist().count(4),df['Do you miss seeing your college and school friends in person? '].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you spend more time chatting online with your friends now than when there was no lockdown? '].tolist().count(1),df['Do you spend more time chatting online with your friends now than when there was no lockdown? '].tolist().count(2),df['Do you spend more time chatting online with your friends now than when there was no lockdown? '].tolist().count(3),df['Do you spend more time chatting online with your friends now than when there was no lockdown? '].tolist().count(4),df['Do you spend more time chatting online with your friends now than when there was no lockdown? '].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you feel disappointed for events and opportunities  (e.g. annual festival, farewell etc.) in college/ school you had been looking forward to?'].tolist().count(1),df['Do you feel disappointed for events and opportunities  (e.g. annual festival, farewell etc.) in college/ school you had been looking forward to?'].tolist().count(2),df['Do you feel disappointed for events and opportunities  (e.g. annual festival, farewell etc.) in college/ school you had been looking forward to?'].tolist().count(3),df['Do you feel disappointed for events and opportunities  (e.g. annual festival, farewell etc.) in college/ school you had been looking forward to?'].tolist().count(4),df['Do you feel disappointed for events and opportunities  (e.g. annual festival, farewell etc.) in college/ school you had been looking forward to?'].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you miss shopping in a market or mall etc. ? '].tolist().count(1),df['Do you miss shopping in a market or mall etc. ? '].tolist().count(2),df['Do you miss shopping in a market or mall etc. ? '].tolist().count(3),df['Do you miss shopping in a market or mall etc. ? '].tolist().count(4),df['Do you miss shopping in a market or mall etc. ? '].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Is the online teaching-learning helping you feel connected as a group? '].tolist().count(1),df['Is the online teaching-learning helping you feel connected as a group? '].tolist().count(2),df['Is the online teaching-learning helping you feel connected as a group? '].tolist().count(3),df['Is the online teaching-learning helping you feel connected as a group? '].tolist().count(4),df['Is the online teaching-learning helping you feel connected as a group? '].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Is the online teaching-learning helping you in maintaining a routine? '].tolist().count(1),df['Is the online teaching-learning helping you in maintaining a routine? '].tolist().count(2),df['Is the online teaching-learning helping you in maintaining a routine? '].tolist().count(3),df['Is the online teaching-learning helping you in maintaining a routine? '].tolist().count(4),df['Is the online teaching-learning helping you in maintaining a routine? '].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Is the online teaching-learning providing some hours of structured learning? '].tolist().count(1),df['Is the online teaching-learning providing some hours of structured learning? '].tolist().count(2),df['Is the online teaching-learning providing some hours of structured learning? '].tolist().count(3),df['Is the online teaching-learning providing some hours of structured learning? '].tolist().count(4),df['Is the online teaching-learning providing some hours of structured learning? '].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you look forward to these online teaching sessions? '].tolist().count(1),df['Do you look forward to these online teaching sessions? '].tolist().count(2),df['Do you look forward to these online teaching sessions? '].tolist().count(3),df['Do you look forward to these online teaching sessions? '].tolist().count(4),df['Do you look forward to these online teaching sessions? '].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you find these classes too much burden in these difficult times?'].tolist().count(1),df['Do you find these classes too much burden in these difficult times?'].tolist().count(2),df['Do you find these classes too much burden in these difficult times?'].tolist().count(3),df['Do you find these classes too much burden in these difficult times?'].tolist().count(4),df['Do you find these classes too much burden in these difficult times?'].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you believe all content you get on social media groups?'].tolist().count(1),df['Do you believe all content you get on social media groups?'].tolist().count(2),df['Do you believe all content you get on social media groups?'].tolist().count(3),df['Do you believe all content you get on social media groups?'].tolist().count(4),df['Do you believe all content you get on social media groups?'].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you forward most content you get on social media groups?'].tolist().count(1),df['Do you forward most content you get on social media groups?'].tolist().count(2),df['Do you forward most content you get on social media groups?'].tolist().count(3),df['Do you forward most content you get on social media groups?'].tolist().count(4),df['Do you forward most content you get on social media groups?'].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()

sizes=[df['Do you verify the authenticity of the messages you forward on social media groups i.e. they are correct or have fake information?'].tolist().count(1),df['Do you verify the authenticity of the messages you forward on social media groups i.e. they are correct or have fake information?'].tolist().count(2),df['Do you verify the authenticity of the messages you forward on social media groups i.e. they are correct or have fake information?'].tolist().count(3),df['Do you verify the authenticity of the messages you forward on social media groups i.e. they are correct or have fake information?'].tolist().count(4),df['Do you verify the authenticity of the messages you forward on social media groups i.e. they are correct or have fake information?'].tolist().count(5)]
labels=["1","2","3","4","5"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.axes().set_aspect("equal")
plt.show()


stopwords_set = set(STOPWORDS)
wordcloud = WordCloud(background_color='black',stopwords = stopwords_set, max_words = 300, max_font_size = 60, random_state=42,).generate(str(df['Your One Line message to the World during the Lockdown !!']))
print(wordcloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

stopwords_set = set(STOPWORDS)
wordcloud = WordCloud(background_color='black',stopwords = stopwords_set, max_words = 300, max_font_size = 60, random_state=42,).generate(str(df['Which among the following are your prime concerns due to changes in academic schedule because of  COVID-19?']))
print(wordcloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


stopwords_set = set(STOPWORDS)
wordcloud = WordCloud(background_color='black',stopwords = stopwords_set, max_words = 300, max_font_size = 60, random_state=42,).generate(str(df['Which of the following best describes your overall mood these days?']))
print(wordcloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

stopwords_set = set(STOPWORDS)
wordcloud = WordCloud(background_color='black',stopwords = stopwords_set, max_words = 300, max_font_size = 60, random_state=42,).generate(str(df['In your opinion, what are the things in classroom teaching that cannot be substituted by online teaching?']))
print(wordcloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


stopwords_set = set(STOPWORDS)
wordcloud = WordCloud(background_color='black',stopwords = stopwords_set, max_words = 300, max_font_size = 60, random_state=42,).generate(str(df['What is the one thing you will want do on the first day when lockdown is lifted?']))
print(wordcloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

stopwords_set = set(STOPWORDS)
wordcloud = WordCloud(background_color='black',stopwords = stopwords_set, max_words = 300, max_font_size = 60, random_state=42,).generate(str(d['Out of the following, choose 5 tools which you use most often these days.']))
print(wordcloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

stopwords_set = set(STOPWORDS)
wordcloud = WordCloud(background_color='black',stopwords = stopwords_set, max_words = 300, max_font_size = 60, random_state=42,).generate(str(d['How has the pandemic COVID-19 changed you as a person? How has it changed your thinking process?']))
print(wordcloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()




