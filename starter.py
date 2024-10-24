import numpy as np
import pandas as pd


dist = 0
MIN = 0
MAX = 28
# returns Euclidean distance between vectors and b
def euclidean(a,b):
    
    return(dist)
        
# returns Cosine Similarity between vectors and b
def cosim(a,b):
    #Change to vectors
    a = np.array(a)
    b = np.array(b)
    #Generalize to higher dimensions
    dist = np.dot(a, b) / np.sqrt(np.sum(a**2)) / np.sqrt(np.sum(b**2))

    return(dist)

def reduce(examples, r):
    # Step 1: Center the data (subtract the mean)
    X_centered = examples - np.mean(examples, axis=0)

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]

    # Step 5: Select the top r eigenvectors
    # Number of components to retain
    top_eigenvectors = eigenvectors[:, :r]

    # Step 6: Project the data onto the top k eigenvectors
    X_reduced = np.dot(X_centered, top_eigenvectors)
    
    return X_reduced

def initialize_centroids(k):
    
    clusters = []
    
    for i in range(k):
        randx = np.random.randint(MIN, MAX)
        randy = np.random.randint(MIN, MAX)
        clusters.append((randx, randy))
        
    return clusters

def update_centroids():
    pass
            
# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    k = 5
    centroids = initialize_centroids(k)
    
    cluster_assignments = {}
    
    for iter, example in enumerate(train):
        print("Iteration ", iter)
        assigned_centroid = None
        
        for i in range(len(example)):
            pixel_x = i % MAX
            pixel_y = np.floor(i / 28)
            pixel_coords = (pixel_x, pixel_y)
            
            min_dist = float('inf')
            
            for centroid in centroids:
                dist = cosim(pixel_coords, centroid)
                if dist < min_dist:
                    min_dist = dist
                    assigned_centroid = centroid
                    
            if assigned_centroid != None:       
                if assigned_centroid not in cluster_assignments:
                    cluster_assignments[assigned_centroid] = [pixel_coords]
                else:
                    cluster_assignments[assigned_centroid].append(pixel_coords)
                
        #Return after first example to test
        print("Intialized Centroids: ", centroids)
        print("Cluster assignments for first example (not reduced): ", cluster_assignments)
        print("Number of keys in cluster assignments: ", len(cluster_assignments.keys()))
        return cluster_assignments
    #return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    #show('valid.csv','pixels')
    r = 250
    k = 5
    
    #Testing distance metrics
    a = np.array([1,2])
    b = np.array([3,4])
    cosim(a, b)
    
    data = pd.read_csv("mnist_train.csv")
    
    print("Shape of data: ", data.shape)
    
    X = data.drop(data.columns[0], axis=1) #first column is class
    X = data.drop(data.columns[-1], axis=1) #last column is nan?
    y = data[data.columns[0]] #labels
    
    X = np.array(X)
    
    print("Shape of original X: ", X.shape)
    print("Shape of original y: ", y.shape)
    
    cluster_assignments_1 = kmeans(X[:500], None, None) #Only the first 500 examples
    
    
    
    

    
    # X_reduced = reduce(X, r)

    # print("X Reduced shape:", X_reduced.shape)
    
    
if __name__ == "__main__":
    main()
    