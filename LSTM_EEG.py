import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py, time
import datetime

def plot_results_():
    nothing=0

def load_data(fileName, numClasses ,numImage, subjectID,**options):
    # Jerarquia del Dataset
    # ImageNet
    #   - ImageNet_CLASS
    #       - ImageNet_IMAGE_number
    #            - Subject_id
    #
    # DATASET INFO:
    # 40 IMAGENET_classes
    # 50 IMAGES per class
    # 6 Subjects
    #
    # SAMPLE INFO
    # EEG recorded is a vector 128(electrodes) x 500 ms(not all EEG has the full time)
    #
    # PROBLEM WITH classes size
    # Info classes Subjects 6 -> 0-29 5(1,3,4,5,6)-> 30-38, 4(3,4,5,6) -> 39
    #
    # PROBLEM WITH id_images_eeg size
    # 50 ->all | 9, 19, 29, 39

    f = h5py.File(fileName, "r")
    classes=f['/ImageNet/']

    train_porcentage = 0.9

    #Creo un radom para saber que clases escoger al azar con max numClass
    if (numClasses > 30 and numClasses < 40):
        tmp = np.arange(0, 39)
    elif numClasses == 40:
        tmp = np.arange(0, 40)
    else:
        tmp = np.arange(0, 30)
    #tmp = np.arange(0,40)
    np.random.shuffle(tmp)
    index_classes=tmp[:numClasses]

    print("Index Classes " + str(len(index_classes)) + "/40")
    print(index_classes)

    #Creo un random para escoger el numero de imagenes

    # PROBLEM WITH id_images_eeg size
    # 50 ->all | 9, 19, 29, 39
    if numImage==50:
        numImage=49

    tmp = np.arange(0, 49)
    np.random.shuffle(tmp)
    index_images = tmp[:numImage]
    print("Index Images " + str(len(index_images)) + "/" + str(len(tmp)))
    print(index_images)

    # PROBLEM WITH classes size
    # Info classes Subjects 6 -> 0-29 5(1,3,4,5,6)-> 30-38, 4(3,4,5,6) -> 39
    if options.get("imagesKeys"):
        imagesKeys= options.get("imagesKeys")
        tmp = np.arange(0, subjectID)
    else:
        imagesKeys=None

    if imagesKeys==None:
        if (numClasses > 30 and numClasses < 40):
            tmp = np.arange(0, 5)
            #gettingImages.keys()
            imagesKeys=['1','3','4','5','6']
            subjectID=5
        elif numClasses == 40:
            tmp = np.arange(0, 4)
            # gettingImages.keys()
            imagesKeys=['3','4','5','6']
            subjectID=4
        else:
            # gettingImages.keys()
            imagesKeys=['1','2','3','4','5','6']
            tmp = np.arange(0,6)


    np.random.shuffle(tmp)
    index_subjects = tmp[:subjectID]
    print("Index Subjects "+str(len(index_subjects))+"/6")
    print(index_subjects)
    print("Id Subjects: " + str(imagesKeys))

    #Modo de selecionar la secuencia Ventanas de V numeros, numero de E electrodos, no se pondran random para dar importantcia al orden espacial, offset donde se empiza con la ventana
    result=[]
    label=[]
    #imagen [inice][indice2][subject1]ordeno electrodos
    for i in range(0, numClasses):
        gettingClass=classes[classes.keys()[index_classes[i]]]
        for x in range(0, numImage):
            gettingImages=gettingClass[gettingClass.keys()[index_images[x]]]
            for z in range(0, subjectID):
                samples = gettingImages[imagesKeys[index_subjects[z]]]
                samples = np.array(samples)
                samples = samples[int(offset):int(window+offset), :int(electrodes)]
                samples = np.reshape(samples,(1,samples.shape[0], samples.shape[1]))
                if ((x==0) & (i==0) & (z==0)):
                    result =samples
                    labeling=np.zeros(40)
                    labeling[i]=1
                    label = np.reshape(labeling, (1, labeling.shape[0]))

                else:
                    result = np.append(result, samples, axis=0)
                    labeling=np.zeros(40)
                    labeling[i]=1
                    labeling = np.reshape(labeling, (1, labeling.shape[0]))
                    label = np.append(label, labeling, axis=0)

    #Randomize de data
    p = np.random.permutation(result.shape[0])
    result=result[p]
    label = label[p]

    row = round(train_porcentage * result.shape[0])

    x_train = result[:int(row), :, :]
    y_train = label[:int(row), :]
    x_test = result[int(row):, :]
    y_test = label[int(row):, :]

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()
    model.add(LSTM(
        input_shape=(window, layers[0]),
        output_dim=layers[1],
        return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(
        input_dim=layers[1]-layers[1]*0.2,
        output_dim=layers[2],
        return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(

        output_dim=layers[3]))

    model.add(Activation("softmax"))

    return model

#Variables
window=200
offset=50
electrodes=128
classes = 29
images = 50
subjects = 4
# For next experiments
imagesKeys=['3','4','5','6']

start = time.time()
X_train, y_train, X_test, y_test = load_data("EEG_Dataset_ImageNet.hdf5", classes, images, subjects,imagesKeys=imagesKeys)
print 'obtaning data time : ', time.time() - start

print "X_train shape"
print X_train.shape
print "y_train shape"
print y_train.shape
print "X_test shape"
print X_test.shape
print "y_test shape"
print y_test.shape

print 'Loaded data'
model=build_model([128, 500, 1000, 40])
start = time.time()
#model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print 'compilation time : ', time.time() - start

print model.summary()


# checkpoints
date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
filepath="weights/weights-improvement-"+str(classes)+"_"+str(images)+"_"+str(subjects)+"-{epoch:02d}-{val_acc:.2f}"+str(date)+".hdf5"
checkpoint_save_model = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_earlystop =  EarlyStopping(monitor='val_loss', patience=2, verbose=0)
callbacks_list = [checkpoint_save_model,checkpoint_earlystop]

#Step 3 Train the model
model.fit(
    X_train,
    y_train,
    batch_size=window,
    epochs=20,
    callbacks=callbacks_list,
    validation_split=0.05)

print 'Finish fit'

#Step 4 - Plot the predictions!
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Step 5 - Save Model!
date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
model_yaml = model.to_yaml()
with open("model_"+str(classes)+"-"+str(images)+"-"+str(subjects)+"_"+str(date)+".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model_weights_"+str(classes)+"-"+str(images)+"-"+str(subjects)+"_"+str(date)+".h5")

print("Saved model to disk")
print("Save model to "+"model_"+str(classes)+"-"+str(images)+"-"+str(subjects)+"_"+str(date)+".yaml")
print("Save model+weights to file:"+"model_weights_"+str(classes)+"-"+str(images)+"-"+str(subjects)+"_"+str(date)+".h5")
