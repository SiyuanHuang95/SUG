import numpy as np
import torch


def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc / pc_L_max
    return pc


def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')
    

def rotation_point_cloud(pc):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    # rotation_matrix = np.array([[cosval, 0, sinval],
    #                             [0, 1, 0],
    #                             [-sinval, 0, cosval]])
    # rotation_matrix = np.array([[1, 0, 0],
    #                             [0, cosval, -sinval],
    #                             [0, sinval, cosval]])
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_by_angle(pc, rotation_angle):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :param rotation_angle: angle of rotation
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. jittering is per point.
    :param pc: B X N X 3 array, original batch of point clouds
    :param sigma:
    :param clip:
    :return:
    """
    jittered_data = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
    jittered_data += pc
    return jittered_data


def shift_point_cloud(pc, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    pc += shifts
    return pc


def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
    """
    N, C = pc.shape
    scales = np.random.uniform(scale_low, scale_high, 1)
    pc *= scales
    return pc


def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    shape_pc = pc
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def pc_augment(pc):
    pc = rotation_point_cloud(pc)
    pc = jitter_point_cloud(pc)
    # pc = random_scale_point_cloud(pc)
    #    pc = rotate_perturbation_point_cloud(pc)
    # pc = shift_point_cloud(pc)
    return pc
    

def random_sample_pc(pts, num_points):
    point_idx = np.arange(0, pts.shape[0])
    np.random.shuffle(point_idx)
    pts = pts[point_idx[:num_points]]
    return pts


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]


def farthest_point_sample_np(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, C, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
    return centroids, centroids_vals


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = torch.device("cuda:" + str(xyz.get_device()))

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # B x npoint
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, C, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals