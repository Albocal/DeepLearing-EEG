from os import listdir
import scipy.io as sio
import h5py

#Jerarquia del Dataset
#ImageNet
#   - ImageNet_CLASS
#       - ImageNet_IMAGE_number
#            - Subject_id

fileName = "EEG_Dataset_ImageNet.hdf5"
f = h5py.File(fileName, "w")
f.attrs['file_name'] = fileName
f.attrs['creator']          = 'Alberto Bozal'
f.attrs['HDF5_Version']     = h5py.version.hdf5_version
f.attrs['h5py_version']     = h5py.version.version

#Six subjects matlab folders:
# Subject_id(Ej 1,2,3,4,5,6)
#      -  n02106662_1152.JPEG.eeg.mat ( class= n02106662, IdImage=1152)
#      ... save all info in x (name variable)
for i in range(1, 7):
    for listfiles in listdir("../../eeg_dataset/eeg_matlab/"+str(i)+"/"):
        nombre1 = listfiles.split('_', 1)
        nombre2 = nombre1[1].split('.', 1)
        print(str(i) + " " + nombre1[0] + " " + nombre2[0])
        x = sio.loadmat('../../eeg_dataset/eeg_matlab/' + str(i) + '/' + nombre1[0]+"_"+nombre2[0]+".JPEG.eeg.mat")
        f.create_dataset("/ImageNet/"+nombre1[0]+'/'+nombre2[0]+"/"+str(i), data=x['x'])

f.close()