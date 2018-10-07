from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
import random

data = pd.read_csv("cars.csv")
data = data.values;

k = 3

colors = ['#FF0000','#00FF00','#0000FF','#FFFF00','#00FFFF','#FF00FF']
def lazy_plot(centroids,labels):
	plt.clf()
	for i in range(k):
		points = np.array([data[j] for j in range(len(data)) if labels[j] == i])
		plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='#550055')
	plt.title("Current positions of centroids")
	plt.xlabel("hp")
	plt.ylabel("time-to-60")
	plt.pause(0.5)

def compute(k_i):
	k = k_i
	colors = ["#"+''.join([random.choice('23456789ABCD') for j in range(6)]) for i in range(k)]

	centroids = data[0:k]

	old_centroids = np.zeros(centroids.shape)

	correction = np.linalg.norm(centroids - old_centroids, axis=1)
	print("Centroids:",centroids)

	while correction.all() != 0:

		labels = np.argmin(dist.cdist(data, centroids),axis=1)

		lazy_plot(centroids,labels)

		old_centroids = deepcopy(centroids)
		#Resetting centroids
		for i in range(k):
			points = [data[j] for j in range(len(data)) if labels[j] == i]
			centroids[i] = np.mean(points, axis=0)
		correction = np.linalg.norm(centroids - old_centroids, axis=1)
		if(correction.all() != 0):
			print("Correction:",centroids)
	print("Final Centroids::",centroids)
	lazy_plot(centroids,labels)

def showPlot():
	pass
	plt.show()
compute(3)
showPlot()
