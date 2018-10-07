import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=300, n_features=3, centers=4,
                  cluster_std=2.6, random_state=60)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
points = data[0]

print(points)
model = KMeans(n_clusters=3)
model.fit(points)
print(model.cluster_centers_)
y_km = model.fit_predict(points)
ax.scatter(points[y_km == 0, 0], points[y_km == 0, 1], points[y_km == 0, 2], c='green')
ax.scatter(points[y_km == 2, 0], points[y_km == 2, 1], points[y_km == 2, 2], c='blue')
ax.scatter(points[y_km == 1, 0], points[y_km == 1, 1], points[y_km == 2, 2], c='orange')
plt.show()
