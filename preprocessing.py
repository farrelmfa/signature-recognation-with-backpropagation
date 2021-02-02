import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import io
from skimage.filters import threshold_otsu   # For finding the threshold for grayscale to binary conversion
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
from time import time
import keras
from tensorflow.python.framework import ops
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog 


genuine_image_paths = "real"
forged_image_paths = "forged"

##PREPROCESSING IMAGE##

def rgbgrey(img):
    # Converts rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg

def greybin(img):
    # Converts grayscale to binary
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
    #img = ndimage.binary_erosion(img).astype(img.dtype)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg

def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgbgrey(img) #rgb to grey
    if display:
        plt.imshow(grey, cmap = matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey) #grey to binary
    if display:
        plt.imshow(binimg, cmap = matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg==1)
    # Now we will make a bounding box with the boundary as the position of pixels on extreme.
    # Thus we will get a cropped image with only the signature part.
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap = matplotlib.cm.Greys_r)
        plt.show()
    return signimg

##FEATURE EXTRACTION##

def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                a = a+1
    total = img.shape[0] * img.shape[1]
    return a/total

def Centroid(img):
    numOfWhites = 0
    a = np.array([0,0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                b = np.array([row,col])
                a = np.add(a,b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a/numOfWhites
    centroid = centroid/rowcols
    return centroid[0], centroid[1]

def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity

def SkewKurtosis(img):
    h,w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    #calculate projections along the x and y axes
    xp = np.sum(img,axis=0)
    yp = np.sum(img,axis=1)
    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(img))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(img))
    
    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(img) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(img) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(img) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(img) * sy**4) - 3

    return (skewx , skewy), (kurtx, kurty)

def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal

def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features

##SAVING FEATURES##

def makeCSV():
    if not(os.path.exists('Features')):
        os.mkdir('Features')
        print('New folder "Features" created')
    if not(os.path.exists('Features/Training')):
        os.mkdir('Features/Training')
        print('New folder "Features/Training" created')
    if not(os.path.exists('Features/Testing')):
        os.mkdir('Features/Testing')
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = genuine_image_paths
    # forged signatures path
    fpath = forged_image_paths
    for person in range(41,51):
        per = ('00'+str(person))[-3:]
        print('Saving features for person id-',per)
        
        with open('Features/Training/training_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(0,5):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(0,5):
                source = os.path.join(fpath, '000'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
        
        with open('Features/Testing/testing_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Testing set
            for i in range(5, 10):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(5,10):
                source = os.path.join(fpath, '000'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
    print("DONE!")

# makeCSV()

def testing(path):
    feature = getCSVFeatures(path)
    if not(os.path.exists('TestFeatures')):
        os.mkdir('TestFeatures')
    with open('TestFeatures/testcsv.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature))+'\n')

n_input = 9


# train_person_id = input("Enter person's id : ")
# train_path = 'Features/Training/training_'+train_person_id+'.csv'
# # test_path = 'TestFeatures/testcsv.csv'
# test_path = 'Features/Testing/testing_'+train_person_id+'.csv'

def readCSV(train_path, test_path, type2=False):
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.to_categorical(correct,2)      # Converting to one hot
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not(type2):
        df = pd.read_csv(test_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = keras.utils.to_categorical(correct,2)      # Converting to one hot
    if not(type2):
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input

ops.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 1000
display_step = 1

# Network Parameters
n_hidden_1 = 7 # 1st layer number of neurons
n_hidden_2 = 49  # 2nd layer number of neurons
n_hidden_3 = 98 # 3rd layer
n_classes = 2 # no. of classes (genuine or forged)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes], seed=4))
}


# Create model
def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer

loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# For accuracies
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Initializing the variables
init = tf.global_variables_initializer()

epoch_loss = []
epoch_num = []

root = Tk()
root.title("Signature Recognation")
root.geometry("400x400")

def evaluate(test_image_path, train_path, test_path, type2=False):   
    if not(type2):
        train_input, corr_train, test_input, corr_test = readCSV(train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(train_path, test_path, type2)
    ans = 'Random'
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, loss = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
            epoch_loss.append(loss)
            epoch_num.append(epoch)
            if loss<0.00001:
                break
            # # Display logs per epoch step
            # if epoch % 10 == 0:
            #     print("Epoch:", '%04d' % (epoch+1), "loss={:.9f}".format(loss))
            # print("Optimization Finished!")
        
        # Finding accuracies
        accuracy1 =  accuracy.eval({X: train_input, Y: corr_train})
        if type2 is False:
            accuracy2 =  accuracy.eval({X: test_input, Y: corr_test})
            print(accuracy2)
            return accuracy1, accuracy2
        else:
            prediction = pred.eval({X: test_input})
            if prediction[0][1]>prediction[0][0]:
                print('Genuine Image')
                img = PhotoImage(file=test_image_path)   
                root.img = img
                canvas = Canvas(root, width = 400, height = 400)          
                canvas.pack()
                canvas.create_text(200, 50,fill="black",font="Times 20",text="Genuine Image", anchor=N)
                canvas.create_image((200, 100), anchor=N, image=img)
                return True
            else:
                print('Forged Image')
                img = PhotoImage(file=test_image_path)  
                root.img = img
                canvas = Canvas(root, width = 400, height = 400)         
                canvas.pack()  
                canvas.create_text(200,50,fill="black",font="Times 20",text="Forged Image", anchor=N)
                canvas.create_image(200, 100, anchor=N, image=img)
                return False


def trainAndTest(rate=0.001, epochs=1000, neurons=4, display=True):    
    start = time()

    # Parameters
    global learning_rate, training_epochs, n_hidden_1
    learning_rate = rate
    training_epochs = epochs

    # Network Parameters
    n_hidden_1 = neurons # 1st layer number of neurons
    n_hidden_2 = 16 # 2nd layer number of neurons
    n_hidden_3 = 32 # 3rd layer

    train_avg, test_avg = 0, 0
    n = 50
    for i in range(1,n+1):
        if display:
            print("Running for Person id",i)
        temp = ('0'+str(i))[-2:]
        train_score, test_score = evaluate(train_path.replace('01',temp), test_path.replace('01',temp))
        train_avg += train_score
        test_avg += test_score
    if display:
        print("Number of neurons in Hidden layer-", n_hidden_1)
        print("Training average-", train_avg/n)
        print("Testing average-", test_avg/n)
        print("Time taken-", time()-start)
    return train_avg/n, test_avg/n, (time()-start)/n

train_person_id = ""
test_image_path = ""

def openfilename(): 

	# open file dialog box to select image 
	# The dialogue box has a title "Open" 
	filename = filedialog.askopenfilename(title ='"pen') 
	return filename

def open_img(): 
	# Select the Imagename from a folder 
	test_image_path = openfilename() 

	# opens the image 
	img = Image.open(test_image_path) 
	
	# resize the image and apply a high-quality down sampling filter 
	img = img.resize((250, 250), Image.ANTIALIAS) 

	# PhotoImage class is used to add image to widgets, icons etc 
	img = ImageTk.PhotoImage(img) 

	# create a label 
	panel = Label(root, image = img) 
	
	# set the image as img 
	panel.image = img 
	panel.grid(row = 2) 

def Submit():

    train_person_id = e1.get()
    # test_image_path = e2.get() + ".png"
    test_image_path = openfilename()
    # test_image_path = input("Enter path of signature image : ")
    if train_person_id == "Syarifah":
        train_person_id = "001"
    elif train_person_id == "Winda":
        train_person_id = "002"
    elif train_person_id == "Verrel":
        train_person_id = "003"
    elif train_person_id == "Utin":
        train_person_id = "004"
    elif train_person_id == "Irvan":
        train_person_id = "005"
    elif train_person_id == "Ressya":
        train_person_id = "006"
    elif train_person_id == "Fauzi":
        train_person_id = "007"
    elif train_person_id == "Rani":
        train_person_id = "008"
    elif train_person_id == "Nadia":
        train_person_id = "009"
    elif train_person_id == "Arif":
        train_person_id = "010"
    elif train_person_id == "Tommy":
        train_person_id = "011"
    elif train_person_id == "Evan":
        train_person_id = "012"
    elif train_person_id == "Andrew":
        train_person_id = "013"
    elif train_person_id == "Angel":
        train_person_id = "014"
    elif train_person_id == "Giry":
        train_person_id = "015"
    elif train_person_id == "Laura":
        train_person_id = "016"
    elif train_person_id == "Vita":
        train_person_id = "017"
    elif train_person_id == "Nabiila":
        train_person_id = "018"
    elif train_person_id == "Ranu":
        train_person_id = "019"
    elif train_person_id == "Rizky O":
        train_person_id = "020"
    elif train_person_id == "Nanda":
        train_person_id = "021"
    elif train_person_id == "Ricky":
        train_person_id = "022"
    elif train_person_id == "Carmel":
        train_person_id = "023"
    elif train_person_id == "Kharis":
        train_person_id = "024"
    elif train_person_id == "Rasya":
        train_person_id = "025"
    elif train_person_id == "Afrina":
        train_person_id = "026"
    elif train_person_id == "Vindy":
        train_person_id = "027"
    elif train_person_id == "Phillip":
        train_person_id = "028"
    elif train_person_id == "Dhea":
        train_person_id = "029"
    elif train_person_id == "Vanessa":
        train_person_id = "030"
    elif train_person_id == "Girar":
        train_person_id = "031"
    elif train_person_id == "Puca":
        train_person_id = "032"
    elif train_person_id == "Atthoriq":
        train_person_id = "033"
    elif train_person_id == "Max":
        train_person_id = "034"
    elif train_person_id == "Dio":
        train_person_id = "035"
    elif train_person_id == "Monica":
        train_person_id = "036"
    elif train_person_id == "Rafli":
        train_person_id = "037"
    elif train_person_id == "Farhan":
        train_person_id = "038"
    elif train_person_id == "Ardhan":
        train_person_id = "039"
    elif train_person_id == "Joandi":
        train_person_id = "040"
    elif train_person_id == "Aughest":
        train_person_id = "041"
    elif train_person_id == "Rakha":
        train_person_id = "042"
    elif train_person_id == "Sulthan":
        train_person_id = "043"
    elif train_person_id == "Jayadi":
        train_person_id = "044"
    elif train_person_id == "Zhafran":
        train_person_id = "045"
    elif train_person_id == "Dio S":
        train_person_id = "046"
    elif train_person_id == "Erika":
        train_person_id = "047"
    elif train_person_id == "Alnodi":
        train_person_id = "048"
    elif train_person_id == "Intan":
        train_person_id = "049"
    elif train_person_id == "Farrel":
        train_person_id = "050"
    elif train_person_id == "":
        messagebox.showerror("Error message", "NAME MUST BE FILLED!")
    else :
    	messagebox.showerror("Error message","INVALID NAME")
    train_path = 'Features/Training/training_'+train_person_id+'.csv'
    testing(test_image_path)
    # test_path = 'Features/Testing/testing_'+train_person_id+'.csv'
    test_path = 'TestFeatures/testcsv.csv'
    evaluate(test_image_path, train_path, test_path, type2=True)

    # root = Tk()
    # root.title("Login Admin!")
    # root.geometry("800x800")    
    # canvas.pack()
    # if bol == True:
    #     print("test")
    #     img = PhotoImage(file=test_image_path)   
    #     canvas = Canvas(root, width = 400, height = 400)          
    #     canvas.pack()
    #     canvas.create_image((200, 50), anchor=N, image=img)
    #     canvas.create_text(100, 100,fill="black",font="Times 20",text="Genuine Image", anchor=N)
    # else:
    #     print("else")
    #     img = PhotoImage(file=test_image_path)  
    #     root.img = img
    #     canvas = Canvas(root, width = 400, height = 400)         
    #     canvas.pack()  
    #     canvas.create_text(200,50,fill="black",font="Times 20",text="Forged Image", anchor=N)
    #     canvas.create_image(200, 100, anchor=N, image=img)

# trainAndTest()

# root = Tk()
# root.title("Signature Recognation")
# root.geometry("400x400")

# membuat label
label = Label(root, text="Signature Recognation")
label.pack()
label1 = Label(root, text="Insert Name:")
label1.pack()

# membuat entry atau inputan
e1 = Entry(root)
e1.pack()
e1.insert(0, "")

# e2 = Entry(root)
# e2.pack()
# e2.insert(0, "")

# membuat button di tkinter
tombol = Button(root, text="Load Image and Submit", command=Submit)
# btn = Button(root, text ='open image', command = open_img)
tombol.pack()
# btn.pack()


root.mainloop()
# plt.plot(epoch_loss)

# plt.show()


