import numpy as np
import math

dist = 0

# Returns Euclidean distance between vectors and b
def euclidean(a,b):
    
    return(dist)

def pearson_corelation(a,b,list):
    '''
    input: a(Point1), b(Point2), List of pair of (x,y) points
    output: pearson corellation coefficient [1 or -1], The stronger the linear relationship.
            Direction of relationship: Positive indicate that as one variable increases, 
            the other tends to increase and vice-versa.
    '''

    sumX,sumY = 0, 0
    for item in list:
        sumX += item[0]
        sumY += item[1]
    meanX = sumX/len(list)
    meanY = sumY/len(list)
    Num = (a[0]-meanX)*(a[1]-meanY) + (b[0]-meanX)*(b[1]-meanY)
    Denm = math.sqrt( (a[0]-meanX)**2 + (b[0]-meanX)**2 ) * math.sqrt( (a[1]-meanY)**2 + (b[1]-meanY)**2 )
    r_xy = Num / Denm   
    return(r_xy)

# returns Cosine Similarity between vectors and b
def cosim(a,b):
    #Generalize to higher dimensions
    dist = np.dot(a, b) / np.sqrt(np.sum(a**2)) / np.sqrt(np.sum(b**2))

    print(dist)

    return(dist)

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
    return(labels)

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
    a = np.array([1,2])
    b = np.array([3,4])
    cosim(a, b)

    list = [(1,2), (2,3), (3,4)]
    a = (1,2)
    b = (2,3)
    r = pearson_corelation(a=a,b=b,list=list)
    
if __name__ == "__main__":
    main()
    