'''
Contains a generic algorithm object which can do vanilla ICP
Then, create plugin functions which can do naive, simple, and fast PFH

ref: https://github.com/stevenliu216/Point-Feature-Histogram/blob/master/pfh/pfh.py

Usage:
icp = FPFH(et=0.1, div=2, nneighbors=8, rad=0.03)
result_cloud = icp.solve(source_pc, target_pc)

'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ast import Div
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from multiprocessing import Pool
from copy import deepcopy
import pickle

data_root = "/point_dg/data"
dataset_list = ["scannet", "shapenet", "modelnet"]

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


def random_sample_pc(pts, num_points):
    point_idx = np.arange(0, pts.shape[0])
    np.random.shuffle(point_idx)
    pts = pts[point_idx[:num_points]]
    return pts

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


def process_pts(pts, pt_num=500):
    process_pts = []
    print(f"Convert the pts {pts.shape[0]}")
    for i in range(pts.shape[0]): 
        cur_pt = fps(normal_pc(pts[i][:, :3]), pt_num)
        process_pts.append(cur_pt)
    return np.array(process_pts)


def get_pfh_descriptor_worker(worker_config):
    descriptor_ = worker_config["descriptor_"]
    pc_ = worker_config["pc"]
    idx = worker_config["idx"]
    try:
        cur_pc_norm, inds, used_idx = descriptor_.calc_normals(pc=pc_)
        cur_pc_hist = descriptor_.calcHistArray(pc_[used_idx], cur_pc_norm, inds, used_idx)
        return {"hist" : cur_pc_hist, "idx": idx, "status": True}
    except:
        print(f" {idx} Error to skip")
        pass

def get_pfh_descriptor(pcs:np.array, dataset_type, method="PFH"):
    descriptor = PFH(e=0.1, div=2, nneighbors=8, rad=0.2)
    hists = []
    work_pool = Pool(processes=24)
    pfh_worker_configs = []
    for i in range(pcs.shape[0]):
        pfh_worker_configs.append(
            {
                "pc": pcs[i],
                "descriptor_": deepcopy(descriptor),
                "idx": i
            }
        )
    
    hists = work_pool.map(get_pfh_descriptor_worker, pfh_worker_configs)
    failure_counter = 0
    for hist_item in hists:
        if hist_item is None:
            failure_counter += 1
    print(f"Number of failures: {failure_counter} among total {len(pfh_worker_configs)}")

    npy_path = os.path.join(data_root, dataset_type, "downsample_normed.npy")
    np.save(npy_path, pcs)

    hist_res = os.path.join(data_root, dataset_type, "hist_res.npy")
    with open(hist_res, "wb") as f:
        pickle.dump(hists, f)
    return hists


def pfh_hist_distance(histS, histT):
    distance = []
    dist = []
    for i in range(histS.shape[0]):
        for j in range(histT.shape[0]):
            #appending the l2 norm and j
            dist.append((np.linalg.norm(histS[i]-histT[j]),j))
        dist.sort(key=lambda x:x[0]) #To sort by first element of the tuple
        distance.append(dist[0][0])
        # HSY: Only need to keep the distance information
        dist = []
        # HSY: only keep the median as the scalar metric
    dis = np.median(distance)
    return dis


class PFH(object):

    """Parent class for PFH
    should set the rad paramter carefully
    """

    def __init__(self, e, div, nneighbors, rad):
        """Pass in parameters """
        self._e = e
        self._div = div
        self._nneighbors = nneighbors
        self._radius = rad

        self._error_list = []
        self._Rlist = []
        self._tlist = []

    def solve(self, P, Q):
        """Main solver
        :P: Source point cloud
        :Q: Target point cloud
        :e: Threshold to stop iterating
        :div: Number of divisions for binning PFH
        :nneighbors: Number of k neighbors for surface normal estimation and PFH neighbors
        :returns: R_list, t_list, Cp, error_list
        """
        iterations = 0
        done = False
        error_o = 100
        Cp = P

        print("...ICP started... \n")
        while not done:
            start = time.process_time()
            # Find correspondences p_i <-> q_i
            # Matching is done via histogram signatures
            matched_dist_inds = self.findMatches(Cp, Q)
            matchInd, _ = self.extractIndices(matched_dist_inds)
            Cq = []
            for i in range(len(Cp)):
                q_near = Q[matchInd[i]]
                Cq.append(q_near)

            if done:
                # When finished, move the points according to the centroids
                print("final move")
                pbar = sum(P)/len(P)
                qbar = sum(Q)/len(Q)

                R_init = np.matrix([[1,0,0], [0,1,0], [0,0,1]])
                t_init = qbar - R_init.dot(pbar)
                R = R_init
                t = t_init
            else:
                # Get transforms
                R, t = self.getTransform(Cp, Cq)

            # Keep track of the R and t for final transforms
            self._Rlist.append(R)
            self._tlist.append(t)

            # Terminate based on error < e, lack of progress, or iterations
            error = self.getError(Cp, Cq, R, t)
            self._error_list.append(error)
            #if abs(error - error_o) < (0.02*e):
            #    ctr += 1
            if error < self._e or abs(error - error_o) < 0.02*self._e or iterations == 30:
                if error < self._e:
                    print('Found low error solution\n')
                elif abs(error - error_o) < .02*self._e:
                    print('Lack of progress, terminating\n')
                elif iterations == 30:
                    print('Reached max iterations\n')
                done = True
            error_o = error

            # Update all P
            new_p_list = []
            for p in Cp:
                new_p_list.append(R.dot(p)+t)
            Cp = new_p_list
            print("===============================")
            print("iteration: ", iterations)
            print("error: ", error)
            end = time.process_time()
            print("Time per iteration: ", end - start)
            print("===============================\n\n")
            iterations = iterations + 1

        return Cp

    def getNeighbors(self, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius
        :pq: TODO
        :pc: TODO
        :returns: TODO
        """
        k = self._nneighbors
        neighbors = []
        for i in range(len(pc)):
            dist = np.linalg.norm(pq-pc[i])
            if dist <= self._radius: #0.005 default, 0.15 for our usage
                neighbors.append((dist, i))
        # print("Found {} neighbors".format(len(neighbors)))
        neighbors.sort(key=lambda x:x[0])
        neighbors.pop(0)
        return neighbors[:k]

    def calc_normals(self, pc):
        """TODO: Docstring for calc_normals.
        :pc: TODO
        :returns: TODO
        """
        normals = []
        ind_of_neighbors = []
        N = len(pc)
        used_idx = []
        for i in range(N):
            # Get the indices of neighbors, it is a list of tuples (dist, indx)
            indN = self.getNeighbors(pc[i], pc) #<- old code

            if len(indN) == 0:
                continue
            used_idx.append(i)
            # Breakout just the indices
            indN = [indN[i][1] for i in range(len(indN))] #<- old code
            ind_of_neighbors.append(indN)            
            # PCA
            X = PFH.convert_pc_to_matrix(pc)[:, indN]
            X = X - np.mean(X, axis=1)
            cov = np.matmul(X, X.T)/(len(indN))
            _, _, Vt = np.linalg.svd(cov)
            normal = Vt[2, :]

            # Re-orient normal vectors
            if np.matmul(normal, -1.*(pc[i])) < 0:
                normal = -1.*normal
            normals.append(normal)

        return normals, ind_of_neighbors, used_idx

    def calcHistArray(self, pc, norm, indNeigh, used_idx):
        """override this function with custom Histogram"""
        N = len(pc)
        histArray = np.zeros((N, self._div**3))
        for i in range(N):
            u = np.asarray(norm[i].T).squeeze()
            k = self._nneighbors
            n = k + 1
            N_features = sp.comb(n, 2)
            features = []
            p_list = [i] + indNeigh[i]
            p_list_copy = [i] + indNeigh[i]
            for z in p_list:
                p_list_copy.pop(0)
                for p in p_list_copy:
                    pi = pc[p]
                    pj = pc[z]
                    if np.arccos(np.dot(norm[p], pj - pi)) <= np.arccos(np.dot(norm[z], pi - pj)):
                        ps = pi
                        pt = pj
                        ns = np.asarray(norm[p]).squeeze()
                        nt = np.asarray(norm[z]).squeeze()
                    else:
                        ps = pj
                        pt = pi
                        ns = np.asarray(norm[z]).squeeze()
                        nt = np.asarray(norm[p]).squeeze()

                    u = ns
                    difV = pt - ps
                    dist = np.linalg.norm(difV)
                    difV = difV/dist
                    difV = np.asarray(difV).squeeze()
                    v = np.cross(difV, u)
                    w = np.cross(u, v)

                    alpha = np.dot(v, nt)
                    phi = np.dot(u, difV)
                    theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))

                    features.append(np.array([alpha, phi, theta]))
                
            features = np.asarray(features)
            pfh_hist, bin_edges = self.calc_pfh_hist(features)
            histArray[i, :] = pfh_hist / (N_features)
        return histArray

    def findMatches(self, pcS, pcT):
        """Find matches from source to target points
        :pcS: Source point cloud
        :pcT: Target point cloud
        :returns: TODO
        """
        print("...Finding correspondences. \n")
        numS = len(pcS)
        numT = len(pcT)
        
        print("...Processing source point cloud...\n")
        normS,indS = self.calc_normals(pcS)
        ''' TODO: implement the different histograms '''
        #histS = calcHistArray_naive(pcT, normS, indS, div, nneighbors)
        #histS = calcHistArray_simple(pcT, normS, indS, div, nneighbors)
        histS = self.calcHistArray(pcS, normS, indS)
        
        print("...Processing target point cloud...\n")
        ''' TODO: implement the different histograms '''
        normT,indT = self.calc_normals(pcT)
        #histT = calcHistArray_naive(pcT, normT, indT, div, nneighbors)
        #histT = calcHistArray_simple(pcT, normT, indT, div, nneighbors)
        
        histT = self.calcHistArray(pcT, normT, indT)
        
        distance = []
        dist = []
        for i in range(numS):
            for j in range(numT):
                #appending the l2 norm and j
                dist.append((np.linalg.norm(histS[i]-histT[j]),j))
            dist.sort(key=lambda x:x[0]) #To sort by first element of the tuple
            distance.append(dist[0][0])
            # HSY: Only need to keep the distance information
            dist = []
        return distance

    def extractIndices(self, DistIndices):
        """
        :DistIndices: TODO
        :returns: TODO
        """
        matchInd = []
        distances = []
        for i in range(len(DistIndices)):
            #always pull the lowest distance result's index
            matchInd.append(DistIndices[i][0][1])
            distances.append(DistIndices[i][0][0])
        return matchInd, distances

    def getTransform(self, Cp, Cq):
        """Calculate the transforms based on correspondences
        :Cp: Source point cloud
        :Cq: Target point cloud
        :returns: R and t matrices
        """
        # Get the centroids
        pbar = sum(Cp)/len(Cp)
        qbar = sum(Cq)/len(Cq)

        # Subtract mean from data
        X = np.matrix(np.zeros((3, len(Cp))))
        Y = np.matrix(np.zeros((3, len(Cq))))
        Cp = PFH.convert_pc_to_matrix(Cp)
        X = Cp - pbar
        Cq = PFH.convert_pc_to_matrix(Cq)
        Y = Cq - qbar
        
        # SVD
        # To use SVD, the value should be larger than zero
        U, _, Vt = np.linalg.svd(X.dot(Y.T))
        V = Vt.T
        det = np.linalg.det(V.dot(U.T))
        anti_reflect = np.matrix([[1,0,0],
                          [0,1,0],
                          [0,0,det]])
        R = V.dot(anti_reflect).dot(U.T)
        t = qbar - R.dot(pbar)
        return R, t

    def getError(self, Cp, Cq, R, t):
        """
        Calculate the transformation error. Assume Cp and Cq have 1-to-1 correspondences.
        """
        err = 0
        for i in range(len(Cp)):
            q_near = Cq[i]
            tmp = np.linalg.norm(R.dot(Cp[i]) + t - q_near)
            err = err + tmp**2
        return err

    def step(self, si, fi):
        """Helper function for calc_pfh_hist. Depends on selection of div
        :si: TODO
        :fi: TODO
        :returns: TODO
        """
        if self._div==2:
            if fi < si[0]:
                result = 0
            else:
                result = 1
        elif self._div==3:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            else:
                result = 2
        elif self._div==4:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            else:
                result = 3
        elif self._div==5:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            elif fi >= si[1] and fi < si[2]:
                result = 2
            elif fi >= si[2] and fi < si[3]:
                result = 3
            else:
                result = 4
        return result

    def calc_thresholds(self):
        """
        :returns: 3x(div-1) array where each row is a feature's thresholds
        """
        delta = 2./self._div
        s1 = np.array([-1+i*delta for i in range(1,self._div)])
        
        delta = 2./self._div
        s3 = np.array([-1+i*delta for i in range(1,self._div)])
        
        delta = (np.pi)/self._div
        s4 = np.array([-np.pi/2 + i*delta for i in range(1,self._div)])
        
        s = np.array([s1,s3,s4])
        return s 

    def calc_pfh_hist(self, f):
        """Calculate histogram and bin edges.
        :f: feature vector of f1,f3,f4 (Nx3)
        :returns:
            pfh_hist - array of length div^3, represents number of samples per bin
            bin_edges - range(0, 1, 2, ..., (div^3+1)) 
        """
        # preallocate array sizes, create bin_edges
        pfh_hist, bin_edges = np.zeros(self._div**3), np.arange(0,self._div**3+1)
        
        # find the division thresholds for the histogram
        s = self.calc_thresholds()
        
        # Loop for every row in f from 0 to N
        for j in range(0, f.shape[0]):
            # calculate the bin index to increment
            index = 0
            for i in range(1,4):
                index += self.step(s[i-1, :], f[j, i-1]) * (self._div**(i-1))
            
            # Increment histogram at that index
            pfh_hist[index] += 1
        
        return pfh_hist, bin_edges


    @staticmethod
    def convert_pc_to_matrix(pc):
        """
            Coverts a point cloud to a numpy matrix.
        Inputs:
            pc - a list of 3 by 1 numpy matrices.
            pts_list = [pts[1][i].reshape(3,1) for i in range(pts[1].shape[0])]
        outputs:
            numpy_pc - a 3 by n numpy matrix where each column is a point.
        """
        numpy_pc = np.matrix(np.zeros((3, len(pc))))

        for index, pt in enumerate(pc):
            numpy_pc[0:3, index] = pt.reshape(3, 1)

        return numpy_pc


class SPFH(PFH):

    """Child class of PFH to implement a different calcHistArray"""

    def calcHistArray(self, pc, norm, indNeigh):
        """Overriding base PFH to SPFH"""
        print("\tCalculating histograms simple method \n")
        N = len(pc)
        histArray = np.zeros((N, self._div**3))
        distArray = np.zeros((self._nneighbors))
        distList = []
        for i in range(N):
            u = np.asarray(norm[i].T).squeeze()
            
            features = np.zeros((len(indNeigh[i]), 3))
            for j in range(len(indNeigh[i])):
                pi = pc[i]
                pj = pc[indNeigh[i][j]]
                if np.arccos(np.dot(norm[i], pj - pi)) <= np.arccos(np.dot(norm[j], pi - pj)):
                    ps = pi
                    pt = pj
                    ns = np.asarray(norm[i]).squeeze()
                    nt = np.asarray(norm[indNeigh[i][j]]).squeeze()
                else:
                    ps = pj
                    pt = pi
                    ns = np.asarray(norm[indNeigh[i][j]]).squeeze()
                    nt = np.asarray(norm[i]).squeeze()
                
                u = ns
                difV = pt - ps
                dist = np.linalg.norm(difV)
                difV = difV/dist
                difV = np.asarray(difV).squeeze()
                v = np.cross(difV, u)
                w = np.cross(u, v)

                alpha = np.dot(v, nt)
                phi = np.dot(u, difV)
                theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))
                
                features[j, 0] = alpha
                features[j, 1] = phi
                features[j, 2] = theta
                distArray[j] = dist

            distList.append(distArray)
            pfh_hist, bin_edges = self.calc_pfh_hist(features)
            histArray[i, :] = pfh_hist / (len(indNeigh[i]))

        return histArray

class FPFH(PFH):

    """Child class of PFH to implement a different calcHistArray"""

    def calcHistArray(self, pc, norm, indNeigh):
        """Overriding base PFH to FPFH"""

        print("\tCalculating histograms fast method \n")
        N = len(pc)
        histArray = np.zeros((N, self._div**3))
        distArray = np.zeros((self._nneighbors))
        distList = []
        for i in range(N):
            u = np.asarray(norm[i].T).squeeze()
            
            features = np.zeros((len(indNeigh[i]), 3))
            for j in range(len(indNeigh[i])):
                pi = pc[i]
                pj = pc[indNeigh[i][j]]
                if np.arccos(np.dot(norm[i], pj - pi)) <= np.arccos(np.dot(norm[j], pi - pj)):
                    ps = pi
                    pt = pj
                    ns = np.asarray(norm[i]).squeeze()
                    nt = np.asarray(norm[indNeigh[i][j]]).squeeze()
                else:
                    ps = pj
                    pt = pi
                    ns = np.asarray(norm[indNeigh[i][j]]).squeeze()
                    nt = np.asarray(norm[i]).squeeze()
                
                u = ns
                difV = pt - ps
                dist = np.linalg.norm(difV)
                difV = difV/dist
                difV = np.asarray(difV).squeeze()
                v = np.cross(difV, u)
                w = np.cross(u, v)
                
                alpha = np.dot(v, nt)
                phi = np.dot(u, difV)
                theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))
                
                features[j, 0] = alpha
                features[j, 1] = phi
                features[j, 2] = theta
                distArray[j] = dist

            distList.append(distArray)
            pfh_hist, bin_edges = self.calc_pfh_hist(features)
            histArray[i, :] = pfh_hist / (len(indNeigh[i]))

        fast_histArray = np.zeros_like(histArray)
        for i in range(N):
            k = len(indNeigh[i])
            for j in range(k):
                spfh_sum = histArray[indNeigh[i][j]]*(1/distList[i][j])
            
            fast_histArray[i, :] = histArray[i, :] + (1/k)*spfh_sum
        return fast_histArray


if __name__ == "__main__":
    
    for dataset_type in dataset_list:
        print(f"Current Process: {dataset_type}")
        npy_path = os.path.join(data_root, dataset_type, "train_pts.npy")
        pts = np.load(npy_path)
        pts = process_pts(pts, pt_num=500)
        hists = get_pfh_descriptor(pts, dataset_type)