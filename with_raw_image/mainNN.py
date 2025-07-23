import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns

def imgDataLoader(path,imgNum):
    imgDataList = []
    imgLblList = []
    for i in range(imgNum):
        targetImage = cv2.imread(f"{path}/{str(i)}.jpg",cv2.IMREAD_GRAYSCALE)
        imgDataList.append(targetImage)
        imgLblList.append(i%10)
    return np.array(imgDataList), np.array(imgLblList)
    
trainingData,trainingLabel = imgDataLoader("E:/Git/Handwritten-digit-recognition/with_raw_image/ProcessedImage/trainingImage",500)  # Must change the path of (trainingImage and testingImage) folder with the actual path of (trainingImage and testingImage) folder according to your pc
testingData,testingLabel = imgDataLoader("E:/Git/Handwritten-digit-recognition/with_raw_image/ProcessedImage/testingImage",100)     # You may have backslash (\) in you path. If yes, replace all backslash(\) wtih forward slash (/)


trainingData = trainingData.reshape(500,784)/255.0
testingData = testingData.reshape(100,784)/255.0
trainingLabel = trainingLabel.reshape(500,1)
testingLabel = testingLabel.reshape(100,1)

encoder = OneHotEncoder(sparse_output=False)

trainLabelEn = encoder.fit_transform(trainingLabel)
testLavelEn = encoder.fit_transform(testingLabel)

inputLyr = 784
hiddenLyr1 = 256
hiddenLyr2 = 128
hiddenLyr3 = 64
outputLyr = 10

np.random.seed(42)
w1 = np.random.randn(inputLyr,hiddenLyr1)*0.01
b1 = np.zeros((1,hiddenLyr1))

w2 = np.random.randn(hiddenLyr1,hiddenLyr2)*0.01
b2 = np.zeros((1,hiddenLyr2))

w3 = np.random.rand(hiddenLyr2,hiddenLyr3)*0.01
b3 = np.zeros((1,hiddenLyr3))

w4 = np.random.randn(hiddenLyr3,outputLyr)*0.01
b4 = np.zeros((1,outputLyr))

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def loss(actualLabel,pradictedLabel):
    return -np.mean(np.sum(actualLabel*np.log(pradictedLabel + 1e-8), axis=1))

def accu():
    z1_test = np.dot(testingData, w1) + b1
    r1_test = relu(z1_test)
       
    z2_test = np.dot(r1_test, w2) + b2
    r2_test = relu(z2_test)
        
    z3_test = np.dot(r2_test,w3) + b3
    r3_test = relu(z3_test)
        
    z4_test = np.dot(r3_test,w4) + b4
    r4_test = softmax(z4_test)

    predictions = np.argmax(r4_test, axis=1)
    true_labels = testingLabel.flatten()
    accuracy = np.mean(predictions == true_labels)
    
    return accuracy * 100,predictions,true_labels

learnRate = 0.1
epochNum = 150
batchSize = 10
batchesNum = trainingData.shape[0] // batchSize

accuRecord = []
lossRecord = []
acu = None
predi = None
t_labl = None

print("Traing start:")
for epoch in range(epochNum):
    totalLoss = 0

    reOrder = np.random.permutation(trainingData.shape[0])
    trainingDataReorder = trainingData[reOrder]
    trainingLabelEnReorder = trainLabelEn[reOrder]
    
    for i in range(batchesNum):
        start = i * batchSize
        end = start + batchSize
        batchData = trainingDataReorder[start:end]
        batchLabel = trainingLabelEnReorder[start:end]

        z1 = np.dot(batchData, w1) + b1
        r1 = relu(z1)
       
        z2 = np.dot(r1, w2) + b2
        r2 = relu(z2)
        
        z3 = np.dot(r2,w3) + b3
        r3 = relu(z3)
        
        z4 = np.dot(r3,w4) + b4
        r4 = softmax(z4)

        batchLoss = loss(batchLabel, r4)
        totalLoss += batchLoss

        dz4 = r4 - batchLabel
        dw4 = np.dot(r3.T, dz4) / batchSize
        db4 = np.mean(dz4, axis= 0, keepdims= True)

        da3 = np.dot(dz4,w4.T)
        dz3 = da3 * relu_derivative(z3) 
        dw3 = np.dot(r2.T, dz3) / batchSize
        db3 = np.mean(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, w3.T)
        dz2 = da2 * relu_derivative(z2)
        dw2 = np.dot(r1.T,dz2) / batchSize
        db2 = np.mean(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2,w2.T)
        dz1 = da1 * relu_derivative(z1)
        dw1 = np.dot(batchData.T,dz1)/ batchSize
        db1 = np.mean(dz1,axis=0, keepdims=True)
        
       
        w4 -= learnRate * dw4
        b4 -= learnRate * db4

        w3 -= learnRate * dw3
        b3 -= learnRate * db3

        w2 -= learnRate * dw2
        b2 -= learnRate * db2

        w1 -= learnRate * dw1
        b1 -= learnRate * db1

    avg_loss = totalLoss / batchesNum
    acu,predi,t_labl = accu()
    print(f"epoch no: {epoch+1} and Loss: {avg_loss:.4f}  Accuracy: {acu}%")  
    accuRecord.append(acu)
    lossRecord.append(avg_loss)

print("Training End")
print(f"final Accuracy: {acu}% ")

np.savez('E:/Git/Handwritten-digit-recognition/with_raw_image/model_params.npz', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, w4=w4,b4=b4) # Provide acutal path for the file with the file name and change all backslash with forward slash


x = np.arange(0,epochNum,1)
cm = confusion_matrix(t_labl, predi)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.set_title("Training Accuracy and loss")
ax1.plot(x,accuRecord,'g', label = "Accuracy")
ax1.plot(x,lossRecord,'r', label = "Loss")
ax1.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy & Loss")
ax1.legend()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title(f"Confusion Matrix (Accuracy: {acu:.2f}%)")
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("True Label")

# Show both plots together
plt.tight_layout()
plt.show()


