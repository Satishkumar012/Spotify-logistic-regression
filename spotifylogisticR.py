import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import math


class logisticRegression:
    
    def __init__(self,x,y):
        self.frame=pd.DataFrame(x)
        self.output=pd.DataFrame(y)
        self.frame.insert(0,'ones',1)
     
        
    @staticmethod
    def label_to_value(data,column_name):
        new_values={}
        for cols in column_name:
            col_new_values={}
            col=data[cols].value_counts()
            List=list(col.index)
            j=len(col)-1
            
            for val in List:
                data[cols]=data[cols].replace(val,j)
                col_new_values[val]=j
                j-=1 
            new_values[cols]=col_new_values
                           
        return data,new_values
     
        
    def encoded_input(self,my_data,mappings):
        
        encoded_data = {}
        for key, value in mydata.items():
         if key in mappings and value in mappings[key]:
             encoded_data[key] = mappings[key][value]
         else:
             encoded_data[key] = value
      
        
        feature_order=[]
        for i in range(len(mydata)):
             feature_order.append(list(mydata.keys())[i])

        encoded_list = [encoded_data[feature] for feature in feature_order]
        return(encoded_list)
    
    
    def training_testing(self):          
        total_rows = len(self.frame)
        seventy_per = int(total_rows * 0.7)
        
        self.training = self.frame.iloc[:seventy_per]
        self.testing = self.frame.iloc[seventy_per:]
        self.y_train = self.output.iloc[:seventy_per]
        self.y_test = self.output.iloc[seventy_per:]
        
        return self.training, self.testing, self.y_train, self.y_test
     
       
    def fit(self):

        _1st_step=(self.training).T@(self.training)
        _2nd_step=pd.DataFrame(np.linalg.inv(_1st_step),columns=list(self.training.columns))
        _3rd_step=(_2nd_step)@(self.training).T
        _4th_step=(_3rd_step@(self.y_train))
        List=_4th_step.values.flatten().tolist()
        
        self.intercepts=float(List[0])
        self.slopes=[float(i) for i in List[1:]]
        
    
    def intercept(self):
        return self.intercepts
    
    
    def slope(self):
        return self.slopes
       
        
    def predict(self, test): 
      self.X = np.array(test)

      z = self.intercepts + np.dot(self.X, np.array(self.slopes))

      self.sigmoid = 1 / (1 + np.exp(-z))

      return self.sigmoid
    
        
    def predict_train(self):
        
        self.testing=self.testing.drop('ones',axis=1)
        
        z = self.intercepts + np.dot(self.testing, np.array(self.slopes))
        self.sigmoid_predict = 1 / (1 + np.exp(-z))
        return self.sigmoid_predict
     
     
    def plot_realvspredict(self):
        plt.scatter(range(len(self.sigmoid_predict)),self.sigmoid_predict,label='predicted values')
        plt.scatter(range(len(self.y_test)),self.y_test,label='Actual Values')
        plt.title("Scatter plot between Actual values and predicted values")
        plt.plot()
        plt.show()
        
        
    def plot_listening_time_vs_age(self):
        
        a=self.frame.head(5)
        b=self.frame.tail(5)
        plt.scatter(a['listening_time'],a['age'])
        plt.scatter(b['listening_time'],b['age'])
        plt.xlabel('listening_time in minutes')
        plt.ylabel('age')
        plt.title("listening time as compared to age")
        plt.plot()  
        plt.show()


    def plot_Device_usage(self,x):
        
        a=x['device_type'].value_counts()
        labels=a.index
        plt.pie(a,labels=labels,autopct='%1.1f%%')
        plt.ylabel("")
        plt.title("Device usage")
        plt.show()
     
        
    def plot_listen_by_user(self,x):
        a=x[['listening_time','subscription_type']]
        avg_listening=a.groupby('subscription_type')['listening_time'].mean()
        avg_listening.plot(kind='bar')
        plt.xlabel('subscription Type')
        plt.ylabel("Average listening time  in minutes")
        plt.title("Average listening time by the subscription_type")
        plt.show()
     
      
    def plot_premiumvsfree(self,x):
        a=x['subscription_type'].value_counts()
        y=a[['Free','Premium']]
        labels=y.index
        plt.pie(y,labels=labels,autopct='%1.1f%%')
        plt.ylabel("")
        plt.title("Subscription distribution of Premium and Free users")
        plt.show()
    
        
    def plot_user_country(self,x):
        a=x['country'].value_counts()
        a.plot(kind='bar')
        plt.xlabel('country name')
        plt.ylabel('NO. of users')
        plt.title("Compare popularity by country")
        plt.show()
    
    
    def plot_heat(self,x):
        a=x[['country','subscription_type']]
        b=a.groupby('country')['subscription_type'].value_counts()
        heatmap_data=b.reset_index(name='count').pivot(index='country',columns='subscription_type',values='count').fillna(0)
        # print(heatmap_data)
        sns.heatmap(heatmap_data)
        plt.title("count by country and subscription type")
        plt.ylabel('country')
        plt.xlabel('Subscription Type')
        plt.show()
        
        
data=pd.read_csv("C:/Users/asus/OneDrive/Desktop/Data Science and Data Analytics/ML ppt/data for spotify logistic.csv")
data1,mappings=logisticRegression.label_to_value(data,['gender','country','device_type','subscription_type'])

x=data1.loc[:,'gender':'offline_listening']
y=data1['is_churned']


LOR=logisticRegression(x,y)

mydata = {
    'gender': input("Enter the gender of the user: "),
    'age': float(input("Enter the age of the user: ")),
    'country': input("Enter the country of the user: "),
    'subscription_type': input("Enter the subscription type of the user: "),
    'listening_time': float(input("Enter the listening time of the user: ")),
    'songs_played_per_day': float(input("Enter how many songs played per day: ")),
    'skip_rate': float(input("Enter the skip rate: ")),
    'device_type': input("Enter the device type: "),
    'ads_listened_per_week': float(input("Enter ads listened per week: ")),
    'offline_listening': float(input("Enter the offline listening: "))
}
d=LOR.encoded_input(mydata,mappings)

v=LOR.training_testing()
z=LOR.fit()
b=LOR.intercept()
print("intercept of this given data",b)
c=LOR.slope()
print("slope of this code",c)
s=LOR.predict_train()
print("Prediction of the given training data",s)
a=LOR.predict(d)
print("given output according to the input",a)

t=LOR.plot_realvspredict()

e=LOR.plot_listening_time_vs_age()

x=pd.read_csv("C:/Users/asus/OneDrive/Desktop/Data Science and Data Analytics/ML ppt/data for spotify logistic.csv")
f=LOR.plot_Device_usage(x)

g=LOR.plot_listen_by_user(x)
h=LOR.plot_premiumvsfree(x)
i=LOR.plot_user_country(x)
j=LOR.plot_heat(x)