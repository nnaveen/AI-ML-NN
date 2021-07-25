'''
Predicting Student Admissions with Neural Networks
We predict student admissions to graduate school at UCLA based on three pieces of data:

GRE Scores (Test)
GPA Scores (Grades)
Class rank (1-4)
The dataset originally came from here: http://www.ats.ucla.edu/
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Loading the data
# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('~/Desktop/student_data.csv')
# Printing out the first 10 rows of our data
data[:10]

'''Plotting the data
First let's make a plot of our data to see how it looks. 
In order to have a 2D plot, let's ingore the rank.'''
# Importing matplotlib
import matplotlib.pyplot as plt

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    
# Plotting the points
plot_points(data)
plt.show()