from genericpath import isfile
from flask import *
from numpy.core.arrayprint import TimedeltaFormat
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import sys
import shutil
from os import listdir
from os.path import isdir
import cv2
# example of loading the keras facenet model
from keras.models import load_model
from PIL import Image
import numpy as np
# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot 



from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
  
app = Flask(__name__) #creating the Flask class object   
app.config["UPLOAD_FOLDER"] = './static/videos/'
@app.route('/') #decorator drfines the   
def home():  
    return (render_template('index.html',page='home'))
@app.route('/choose',methods=['POST']) #decorator drfines the   
def choose():
    try:
        shutil.rmtree('./data-test')
    except:
        print("file doesnt exist")
    file = request.files['file_name']
    if file:
        filename = 'sample.mp4'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        return render_template('index.html',page='process')
    return render_template('index.html',page='upload-error')

@app.route('/about/<name>') #decorator drfines the 
def about(name):
    return "<h1>hello, this is our about page "+name+"</h1>"
@app.route('/contact') #decorator drfines the 
def contact():  
    return "<h1>hello, this is our Contact Page</h1>"

@app.route('/login') #decorator drfines the 
def future():  
    return render_template("login.html") 

# @app.route('/voice',methods=['GET']) #decorator drfines the 
# def voice():  
#     username = request.args.get('name') 
#     passwrd=request.form['pass']  
#     if username!="":
#         os.system("python voice.py "+username)
#         return render_template("index.html",page=)   
#     else:
#         return render_template("404.html") 

@app.route('/process') #decorator drfines the 
def process():
    cam = cv2.VideoCapture(".\\static\\videos\\sample.mp4")
    try:    
        # creating a folder named data
        if not os.path.exists('data-test'):
            os.makedirs('data-test')
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    # frame
    currentframe = 0
    while(True):    
        # reading from frame
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = './data-test/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
            if currentframe==1:
                val_data_folder = "data-test/"
                # load test dataset
                print("Load dataset")
                testX = load_dataset(val_data_folder)
                # save arrays to one file in compressed format
                print("Start compressing")
                savez_compressed('data-test/result.npz', testX)
                print("creted compressed file")
                cam.release()
                cv2.destroyAllWindows()
                ############################################

                # load the face dataset
                data = load('data-test/result.npz')
                testX = data['arr_0']
                print('Loaded: ', testX.shape)
                # load the facenet model
                print('Loading Model')
                model = load_model('facenet_keras.h5',compile=False)
                print('Loaded Model')
                # convert each face in the test set to an embedding
                newTestX = list()
                for face_pixels in testX:
                    embedding = get_embedding(model, face_pixels)
                    newTestX.append(embedding)
                newTestX = asarray(newTestX)
                print(newTestX.shape)
                # save arrays to one file in compressed format
                savez_compressed('data-test/result-embedding.npz',newTestX)
                return render_template('index.html',page='predict')
        else:
            return "error occured"
            break
            
    # Release all space and windows once done



# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    path = directory
    # get face
    face = extract_face(path)
    # store
    faces.append(face)    
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X = list()
	# enumerate folders, on per class
    
	for file in listdir(directory):
		# path
		path = directory + file
		# skip any files that might be in the dir
		if not isfile(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		
		X.extend(faces)
	return asarray(X)
# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]



@app.route('/predict') #decorator drfines the 
def predict():
    data = load('data-test/result.npz')

    testX_faces = data['arr_0']
    data = load('data-test/result-embedding.npz')

    testX_faces_new = data['arr_0']

    # load face embeddings
    # data = load('drive/MyDrive/s8-ucek-assets/5-celebrity-faces-embeddings.npz')
    data = load('teachers-faces-embeddings.npz')

    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # test model on a random example from the test dataset
    selection = choice([i for i in range(0,1)])

    random_face_pixels = testX_faces[selection]
    random_face_emb = testX_faces_new[selection]
    # random_face_class = testy[selection]
    # random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    # print('Expected: %s' % random_face_name[0])

    # call voice script
   

    # plot for fun
    # pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    print(title)
    # pyplot.title(title)
    # pyplot.show()
    if class_probability<86:
        predict_names[0]='unknown'
    os.system("python voice.py "+predict_names[0])
    return render_template('index.html',page='result',name=predict_names[0],accuracy=class_probability)
     



if __name__ =='__main__':  
    app.run(debug = True)