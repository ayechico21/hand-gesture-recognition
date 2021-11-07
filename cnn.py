import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2

convnet = input_data(shape=[None, 240, 250, 1], name='input')
# Conv Layer 1
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
# Conv Layer 2
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
# Conv Layer 3
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
# Conv Layer 4
convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
# Conv Layer 5
convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
# Conv Layer 6
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
# Conv Layer 7
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
# Conv Layer 8
convnet = fully_connected(convnet, 1000, activation='relu')
convnet = dropout(convnet, 0.75)
# Fully Connected Layer with SoftMax as Activation Function
convnet = fully_connected(convnet, 9, activation='softmax')

# Regression for ConvNet with ADAM optimizer
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='regression')
model = tflearn.DNN(convnet, tensorboard_verbose=0)

X_train = []
Y_train = []
X_test = []
Y_test = []

for i in range(75):
    img = cv2.imread('Images\\one\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
for i in range(75):
    img = cv2.imread('Images\\two\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
for i in range(75):
    img = cv2.imread('Images\\three\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
for i in range(75):
    img = cv2.imread('Images\\four\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
for i in range(75):
    img = cv2.imread('Images\\five\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
for i in range(75):
    img = cv2.imread('Images\\hello\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
for i in range(75):
    img = cv2.imread('Images\\fist\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
for i in range(75):
    img = cv2.imread('Images\\thumbsUp\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
for i in range(75):
    img = cv2.imread('Images\\blank\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(gray.reshape(240, 250, 1))
    Y_train.append([1, 0, 0, 0, 0, 0, 0, 0, 0])


for i in range(75, 101):
    img = cv2.imread('Images\\one\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
for i in range(75, 101):
    img = cv2.imread('Images\\two\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
for i in range(75, 101):
    img = cv2.imread('Images\\three\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
for i in range(75, 101):
    img = cv2.imread('Images\\four\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
for i in range(75, 101):
    img = cv2.imread('Images\\five\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
for i in range(75, 101):
    img = cv2.imread('Images\\hello\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
for i in range(75, 101):
    img = cv2.imread('Images\\fist\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
for i in range(75, 101):
    img = cv2.imread('Images\\thumbsUp\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
for i in range(75, 101):
    img = cv2.imread('Images\\blank\\img_' + str(i) + '.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(gray.reshape(240, 250, 1))
    Y_test.append([1, 0, 0, 0, 0, 0, 0, 0, 0])

print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))

model.fit(X_train, Y_train, n_epoch=50, validation_set=(X_test, Y_test), snapshot_step=100, show_metric=True, run_id='convnet')
model.save("Model\\cnnModel.tfl")