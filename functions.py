import numpy as np
import math

def convWeights():
    filters = np.asarray([[[[ 1.23197868e-01,  1.21013328e-01,  5.55894114e-02,
           1.22546211e-01, -1.13924950e-01,  1.63308624e-02,
           7.11815208e-02, -1.79373309e-01, -7.33664185e-02,
          -1.78919733e-01,  2.87411772e-02,  3.13604474e-01,
          -7.38753751e-02, -3.51157575e-03, -1.69863924e-01,
           2.97689904e-03,  1.36900201e-01, -9.75987408e-03,
           7.64718279e-02, -4.44247983e-02, -3.57660800e-02,
           7.04894215e-02, -1.34449244e-01,  3.61628145e-01,
           2.20158324e-01, -3.74784805e-02,  9.02241841e-02,
           1.33106619e-01]],

        [[ 1.34669662e-01,  1.39181912e-01, -2.00249329e-01,
           8.32050964e-02,  2.55689830e-01, -8.74001533e-02,
          -6.25811666e-02,  1.03548355e-01,  1.29270881e-01,
           1.17336243e-01,  7.15075433e-02,  5.02449982e-02,
           7.83289373e-02,  7.35287741e-02,  3.57163921e-02,
           1.35013342e-01, -1.07945085e-01,  6.52265996e-02,
          -8.27328712e-02, -2.48917341e-02,  8.78830627e-02,
          -2.14274451e-01, -9.54130962e-02, -5.36020063e-02,
          -1.67288139e-01, -1.45073146e-01, -1.20265849e-01,
           8.77386332e-02]],

        [[-2.03397632e-01, -6.98120818e-02,  1.92361027e-01,
          -2.78055370e-01, -8.65041688e-02,  2.77441442e-02,
           1.48318764e-02, -1.56550169e-01, -8.21324512e-02,
          -1.33350149e-01, -1.00128660e-02,  3.03311467e-01,
           2.30857700e-01, -7.76780993e-02, -8.57873037e-02,
          -1.33485168e-01,  6.61595389e-02,  2.88855173e-02,
           3.87611687e-02,  1.03204258e-01,  8.21961537e-02,
           2.95181815e-02, -2.12194055e-01, -6.03742041e-02,
           9.99485999e-02,  9.62157547e-02,  1.90097004e-01,
           3.50952148e-01]]],


       [[[-9.26949456e-02, -1.81710169e-01,  2.14355841e-01,
          -1.21761888e-01,  9.33905039e-03, -3.65641825e-02,
          -1.95963353e-01,  2.11966094e-02, -8.70793611e-02,
           1.56878769e-01, -8.36760327e-02,  1.17640374e-02,
           2.51611322e-01,  2.44480614e-02,  1.46383852e-01,
          -2.68228471e-01, -1.22409724e-01,  1.20129570e-01,
           9.46300402e-02,  4.89647724e-02,  6.40809834e-02,
           9.83199924e-02, -1.45534389e-02, -3.48517835e-01,
           8.88029411e-02,  5.62829562e-02, -1.94121704e-01,
           3.80887315e-02]],

        [[-2.79713899e-01,  1.86545104e-01,  1.19675972e-01,
          -6.43393323e-02, -2.57329047e-01,  6.99831769e-02,
           1.88752711e-01,  1.15586378e-01,  8.03610682e-02,
          -2.86427766e-01,  1.28067032e-01, -1.69207066e-01,
          -2.33089440e-02,  3.73774096e-02, -4.05135229e-02,
           2.97349542e-01,  1.12967290e-01, -1.76789537e-01,
          -1.77275762e-02, -5.26154675e-02,  1.36285454e-01,
          -2.74531573e-01, -8.76444578e-02,  2.86760211e-01,
           1.76616356e-01,  1.02616340e-01,  1.06655411e-01,
           1.31054431e-01]],

        [[ 3.27885538e-01,  1.82210609e-01, -2.49361053e-01,
          -8.22947547e-03,  2.72016883e-01, -6.74307942e-02,
          -7.09627122e-02, -2.85547644e-01,  6.38959184e-02,
           1.31156087e-01,  7.00197145e-02,  9.63320732e-02,
          -1.09148845e-01, -3.58609147e-02, -2.69000791e-02,
          -1.77067928e-02, -7.22070113e-02,  9.31690112e-02,
          -1.01348693e-02, -3.45227830e-02, -7.56589621e-02,
          -1.13439746e-02,  7.00330511e-02,  2.76339017e-02,
          -1.03042342e-01, -8.79302546e-02, -1.45018265e-01,
           6.20377064e-02]]],


       [[[ 1.87558755e-01,  9.76180956e-02, -7.57730082e-02,
           1.84656933e-01, -1.28318638e-01, -5.06559154e-04,
          -8.16791598e-03,  2.71143734e-01,  8.50609094e-02,
          -1.72460467e-01, -2.41475087e-02, -1.38608024e-01,
          -2.69374758e-01,  1.73315275e-02,  1.93353314e-02,
           2.17706740e-01, -6.60839155e-02, -1.38610080e-01,
           1.49489865e-01, -7.04622194e-02,  4.30011712e-02,
          -6.30701482e-02,  9.66540873e-02, -5.79412095e-02,
          -1.59380466e-01, -8.73005092e-02,  3.59879918e-02,
          -2.92522609e-01]],

        [[ 9.43545476e-02, -2.06278190e-01, -2.27437377e-01,
           1.36626363e-01,  9.29863006e-02, -1.12979664e-02,
          -4.36420515e-02,  1.00281031e-05, -3.36728729e-02,
           2.22770974e-01,  3.10836383e-03, -2.91987598e-01,
           2.20956489e-01,  4.90549579e-02, -1.50953174e-01,
          -1.97139218e-01,  1.04808524e-01,  2.63976365e-01,
          -2.38684788e-01, -2.98483483e-02, -1.67186096e-01,
           2.50295848e-01,  1.39594197e-01,  1.54975265e-01,
          -1.53669594e-02,  4.64403117e-03, -6.91256151e-02,
          -1.36238858e-01]],

        [[-3.09300691e-01, -3.34686041e-02,  2.60633290e-01,
          -1.81505717e-02, -3.69318202e-02, -2.16381848e-02,
          -4.14582975e-02,  1.27186686e-01,  6.93331063e-02,
           5.65112848e-03,  5.70804859e-03, -1.82262853e-01,
          -1.94020137e-01,  1.89259835e-02,  1.43617094e-01,
           9.70994607e-02, -6.67568296e-03, -1.41669512e-01,
           2.18719784e-02, -4.92307432e-02,  3.59067582e-02,
          -5.48286177e-02,  2.25167442e-02, -1.92230850e-01,
          -1.86511099e-01,  1.15473740e-01, -5.42746149e-02,
          -1.04113542e-01]]]])

    filfersR = []        #filters reshaped
    for i in range(28):
    # get the filter
        f = filters[:, :, :, i]
        filfersR.append(f)
    filfersR = np.asarray(filfersR)
    return filfersR
    

#apply one convolution for one image with one filter
#Tested
def convolution (image, flt):
    h, w, rgb = image.shape      #image height, width
    fh, fw, rgb = flt.shape   #filter heigth, width
    oh = h-fh+1             #output height, witdth
    ow = w-fw+1
    output = np.zeros((oh, ow))
    for x in range (oh):
        for y in range (ow):
            conv = 0
            for i in range (fh):
                for j in range(fw):
                    conv += image[x+i, y+j] * flt[i,j]
            output[x,y] = conv
    return output

# perform convolutional layer
def convLayer (image, filters):
    output = list()
    for flt in filters:
        result = convolution (image, flt)
        output.append(result)
    return result

#Tested a bit
def flatten(images):
    return np.reshape(images, -1)

def dense(X, W, activation):
    # weights shape: (x,y), x - lef side neurons, y - right side
    output = np.zeros(W.shape[1])  
    for i in range (W.shape[1]):
        for j in range(len(X)):        # len(X) = len(W.shape[0])
            output[i] += X[j]*W[j][i]

    if(activation.lower() == "relu"):
        for i in range(len(output)):
            if (output[i] < 0):
                output[i] = 0
    elif(activation.lower() == "softmax"):
        output = np.exp(output) / np.sum(np.exp(output)) 

    return output

def predict(probablilities):
    maxProb = 0
    pred = 0
    for i in range(10):
        if (probablilities[i] > maxProb):
            maxProb = probablilities[i]
            pred = i
    return (pred, maxProb)

#pred [0,1], actual {0,1}
def CrossEntropy(prediction, label):
    loss = 0
    for i in range(10):
        if i == label:
            loss += 1*(-1 * math.log(prediction[i]))
        else:
            loss += 0*(-1 * math.log(prediction[i]))
    return loss

# d2outpyt = y predicted
def d1Bacprop(w, labels, d1output, d2output):
    # entropyDerr = d1output[i] - labels[i]
    # #if d1output index == d2 neuron index
    # softmaxDerr = d1output[i]*(1-d1output[i])
    # #else
    # softmaxDerr = d1output[i]*d1output[#d2 neuron index]
    grad = np.zeros((w.shape[0], w.shape[1]))
    for i in range (w.shape[1]):
        for j in range (w.shape[0]):
            grad[j][i] = (d2output[i]-labels[i]) * (d2output[i]*(1-d2output[i])) * d1output[j]
    return grad

def d2Bacprop(w, d2w, labels, d1output, d2output, inputt):

    grad = np.zeros((w.shape[0], w.shape[1]))
    for i in range (w.shape[1]):
        for j in range (w.shape[0]):
            for x in range (d2w.shape[1]):
                if(d1output[x] == 0):   # due to ReLU activation, it's derrivatie either 0 or 1
                    grad[j][i] +=0
                else:
                    grad[j][i] += (d2output[x]-labels[x]) * (d2output[x]*(1-d2output[x])) * d2w[i][x] * inputt[j]
    return grad

