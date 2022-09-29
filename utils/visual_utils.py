import matplotlib.pyplot as plt


def visualize_feature_scatter(features, cls=None, labels_=None, cluster_centers=None, file_path=None):
    colors = ['#7FFFD4', '#000000','#0000FF', '#A52A2A','#DEB887', '#00FFFF', '#FFD700','#808080', '#000080','#FFA500','#FF0000', '#FFFF00']
    fig = plt.figure()

    if labels_ is not None:
        if features.shape[-1] <= 2:
            ax = plt.axes()
        else:
            ax = plt.axes(projection='3d')
        for k, col in zip(range(10), colors):
            my_members = labels_ == k
            cluster_center = cluster_centers[k]
            if features.shape[-1] <= 2:
                ax.plot(features[my_members, 0], features[my_members, 1], "w", markerfacecolor=col, marker=".")
                ax.plot(
                    cluster_center[0],
                    cluster_center[1],
                    "o",
                    markerfacecolor=col,
                    markeredgecolor=col,
                    markersize=6,
                )
                # ax.text(x=cluster_center[0], y=cluster_center[1], s=str(k))
            else:
                ax.scatter3D(features[my_members, 0], features[my_members, 1], features[my_members, 2], c=col)
                ax.scatter3D(cluster_center[0], cluster_center[1], cluster_center[2], c=col, s=40)
    else:
        if features.shape[-1] > 2:
            ax = plt.axes(projection='3d')
            ax.scatter3D(features[:, 0], features[:, 1], features[:, 2], "w")
        else:
            plt.scatter(features[:, 0], features[:, 1])

    if cls is not None:
        title = "Clustering Result for cls " + str(cls)
        plt.title(title)

    if file_path:
        plt.savefig(file_path)
        print(f"Save Clustering Vis to {file_path}")
    else:
        plt.show()
