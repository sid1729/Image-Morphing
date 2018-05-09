
import numpy as np
import scipy as scipy
import PIL as pillow
import imageio
from scipy.spatial import Delaunay
from PIL import ImageDraw,Image
from scipy.interpolate import RectBivariateSpline
import os
import subprocess as sp


class Affine:
    def __init__(self,source,destination):
        if not (source.shape[0] == 3 and source.shape[1] == 2 and destination.shape[0] == 3 and destination.shape[1] == 2) :
            raise ValueError('Arguments are not of the correct dimensions')
        if not (source.dtype == np.float64 and destination.dtype == np.float64):
            raise ValueError('Arguments are not of the correct type')
        self.source = source
        self.destination = destination
        self.get_matrix()

    def get_interpolation_func(self,interpolation_func):
        self.interpolation_func = interpolation_func


    def get_matrix(self):
        A = np.array([[self.source[0][0],self.source[0][1],1,0,0,0] ,[0,0,0,self.source[0][0],self.source[0][1],1]],dtype=np.float64)
        A = np.append(A,np.array([[self.source[1][0],self.source[1][1],1,0,0,0] ,[0,0,0,self.source[1][0],self.source[1][1],1]],dtype=np.float64),axis=0)
        A = np.append(A, np.array([[self.source[2][0], self.source[2][1], 1, 0, 0, 0],[0, 0, 0, self.source[2][0], self.source[2][1], 1]], dtype=np.float64), axis=0)

        B = np.array([[self.destination[0][1]],[self.destination[0][0]],[self.destination[1][1]],[self.destination[1][0]],[self.destination[2][1]],[self.destination[2][0]]],dtype=np.float64)
        h = np.linalg.solve(A,B)
        self.matrix = np.array([[h[0],h[1],h[2]], [h[3], h[4], h[5]],[0,0,1]],dtype=np.float64)

    def transform(self,sourceImage,destinationImage):
        if not (isinstance(sourceImage,np.ndarray) and isinstance(destinationImage,np.ndarray)):
            raise TypeError ('parameters are not Numpy array')


        max_x = (int) (np.max(np.transpose(self.source)[1])) + 1
        min_x = np.floor(np.min(np.transpose(self.source)[1])).astype('int')
        min_y = np.floor(np.min(np.transpose(self.source)[0])).astype('int')
        max_y = (int) (np.max(np.transpose(self.source)[0])) + 1

        bounded_image = sourceImage[min_x:max_x,min_y:max_y]


        self.interpolation_func = RectBivariateSpline(np.arange(min_x,max_x), np.arange(min_y,max_y), bounded_image, kx=1, ky=1)
        image = Image.new('L', (sourceImage.shape[1], sourceImage.shape[0]), 0)
        ImageDraw.Draw(image).polygon(((self.destination[0][0], self.destination[0][1]),
                                       (self.destination[1][0], self.destination[1][1]),
                                       (self.destination[2][0], self.destination[2][1])), outline=255,
                                      fill=255)
        mask = np.array(image)
        inverted_matrix = np.linalg.inv(self.matrix)
        list = np.nonzero(mask)
        x = np.ones(len(list[0]))
        matrix_list = np.array([list[0],list[1],x])
        source_matrix = np.matmul(inverted_matrix,matrix_list)
        destinationImage[list[0],list[1]] = np.round(self.interpolation_func.ev(source_matrix[1],source_matrix[0]))


class Blender:

    def __init__(self,startImage,startPoints,endImage,endPoints):
        if not (isinstance(startImage,np.ndarray) and isinstance(startPoints,np.ndarray) and isinstance(endImage,np.ndarray) and isinstance(endPoints,np.ndarray)):
            raise TypeError ('parameters are not Numpy array')

        self.startPoints = startPoints
        self.endPoints = endPoints
        self.startImage = startImage
        self.endImage = endImage
        tri = Delaunay(self.startPoints)
        self.first_tri = self.startPoints[tri.simplices]
        self.second_tri = self.endPoints[tri.simplices]

    def getBlendedImage(self,alpha):
        self.middle_triangle = np.array(self.first_tri,dtype=np.float64)
        for index in range(0,len(self.first_tri)):
            for index_2 in range(0,len(self.first_tri[index])):
                for index_3 in range(0,len(self.first_tri[index][index_2])):
                    self.middle_triangle[index][index_2][index_3] =  (1 - alpha) * self.first_tri[index][index_2][index_3] + (alpha) *  self.second_tri[index][index_2][index_3]

        img_1 = Image.new('L', (self.startImage.shape[1], self.startImage.shape[0]), 0)
        image_morphed_1 = np.array(img_1)
        img_2 = Image.new('L', (self.endImage.shape[1], self.endImage.shape[0]), 0)
        image_morphed_2 = np.array(img_2)

        for i in range(0,len(self.middle_triangle)):
            object_2 = Affine(self.first_tri[i],self.middle_triangle[i])
            object_3 = Affine(self.second_tri[i],self.middle_triangle[i])
            object_2.transform(self.startImage,image_morphed_1)
            object_3.transform(self.endImage,image_morphed_2)
        image_morphed = (1 - alpha) * image_morphed_1 + (alpha) * image_morphed_2
        return image_morphed

    def generateMorphVideo(self,targetFolderPath,sequenceLength,includeReversed=True):
        count = sequenceLength
        i = 0
        if os.path.isdir(targetFolderPath) == False:
            os.mkdir(targetFolderPath)
        if includeReversed == True:
            array = np.linspace(0,1,num=count)
            for alpha in array:
                i = i + 1
                if alpha == 0:
                    image_result = Image.fromarray(self.startImage)
                elif alpha == 1:
                    image_result = Image.fromarray(self.endImage)
                else:
                    image_result_array = self.getBlendedImage(alpha)
                    image_result = Image.fromarray(image_result_array)
                    image_result = image_result.convert('L')
                save_string = targetFolderPath + '/frame{0:03d}.jpg'.format(i)
                image_result.save(save_string)
                save_string = targetFolderPath + '/frame{0:03d}.jpg'.format(2 * count - i + 1)
                image_result.save(save_string)

        else:
            array = np.linspace(0,1,num=count)
            i = i + 1
            for alpha in array:
                if alpha == 0:
                    image_result = Image.fromarray(self.startImage)
                elif alpha == 1:
                    image_result = Image.fromarray(self.endImage)
                else:
                    image_result_array = self.getBlendedImage(alpha)
                    image_result = Image.fromarray(image_result_array)
                    image_result = image_result.convert('L')
                save_string = targetFolderPath + '/frame{0:03d}.jpg'.format(i)
                image_result.save(save_string)
        sp.call("ffmpeg -y -framerate 5 -i " + targetFolderPath + "/frame%03d.jpg -r 5 " + targetFolderPath + "/morph.mp4",shell=True)

class ColorAffine:
    def __init__(self,source,destination):
        if not (source.shape[0] == 3 and source.shape[1] == 2 and destination.shape[0] == 3 and destination.shape[1] == 2) :
            raise ValueError('Arguments are not of the correct dimensions')
        if not (source.dtype == np.float64 and destination.dtype == np.float64):
            raise ValueError('Arguments are not of the correct type')
        self.source = source
        self.destination = destination
        self.get_matrix()

    def get_interpolation_func(self,interpolation_func):
        self.interpolation_func = interpolation_func


    def get_matrix(self):
        A = np.array([[self.source[0][0],self.source[0][1],1,0,0,0] ,[0,0,0,self.source[0][0],self.source[0][1],1]],dtype=np.float64)
        A = np.append(A,np.array([[self.source[1][0],self.source[1][1],1,0,0,0] ,[0,0,0,self.source[1][0],self.source[1][1],1]],dtype=np.float64),axis=0)
        A = np.append(A, np.array([[self.source[2][0], self.source[2][1], 1, 0, 0, 0],[0, 0, 0, self.source[2][0], self.source[2][1], 1]], dtype=np.float64), axis=0)

        B = np.array([[self.destination[0][1]],[self.destination[0][0]],[self.destination[1][1]],[self.destination[1][0]],[self.destination[2][1]],[self.destination[2][0]]],dtype=np.float64)
        h = np.linalg.solve(A,B)
        self.matrix = np.array([[h[0],h[1],h[2]], [h[3], h[4], h[5]],[0,0,1]],dtype=np.float64)

    def transform(self,sourceImage,destinationImage):
        if not (isinstance(sourceImage,np.ndarray) and isinstance(destinationImage,np.ndarray)):
            raise TypeError ('parameters are not Numpy array')

        image = Image.new('L', (sourceImage.shape[1], sourceImage.shape[0]), 0)
        ImageDraw.Draw(image).polygon(((self.destination[0][0], self.destination[0][1]),
                                       (self.destination[1][0], self.destination[1][1]),
                                       (self.destination[2][0], self.destination[2][1])), outline=255,
                                      fill=255)
        mask = np.array(image)
        inverted_matrix = np.linalg.inv(self.matrix)
        max_x = (int)(np.max(np.transpose(self.source)[1])) + 1
        min_x = np.floor(np.min(np.transpose(self.source)[1])).astype('int')
        min_y = np.floor(np.min(np.transpose(self.source)[0])).astype('int')
        max_y = (int)(np.max(np.transpose(self.source)[0])) + 1
        destinationImage = np.transpose(destinationImage)

        for i in range(0, 3):
            bounded_image = np.transpose(sourceImage)[i][min_y:max_y,min_x:max_x]
            self.interpolation_func = RectBivariateSpline(np.arange(min_y,max_y),
                                                          np.arange(min_x,max_x), bounded_image,
                                                          kx=1, ky=1)
            list = np.nonzero(mask)
            x = np.ones(len(list[0]))
            matrix_list = np.array([list[0], list[1], x])
            source_matrix = np.matmul(inverted_matrix, matrix_list)
            destinationImage[i][list[1], list[0]] = np.round(self.interpolation_func.ev(source_matrix[0], source_matrix[1]))


class ColorBlender:

    def __init__(self,startImage,startPoints,endImage,endPoints):
        if not (isinstance(startImage,np.ndarray) and isinstance(startPoints,np.ndarray) and isinstance(endImage,np.ndarray) and isinstance(endPoints,np.ndarray)):
            raise TypeError ('parameters are not Numpy array')

        self.startPoints = startPoints
        self.endPoints = endPoints
        self.startImage = startImage
        self.endImage = endImage
        #self.interpolate_func_1 = RectBivariateSpline(np.arange(startImage.shape[0]), np.arange(startImage.shape[1]),startImage, kx=1, ky=1)
        #self.interpolate_func_2 = RectBivariateSpline(np.arange(startImage.shape[0]), np.arange(startImage.shape[1]),endImage, kx=1, ky=1)
        tri = Delaunay(self.startPoints)
        self.first_tri = self.startPoints[tri.simplices]
        self.second_tri = self.endPoints[tri.simplices]

    def getBlendedImage(self,alpha):
        self.middle_triangle = np.array(self.first_tri,dtype=np.float64)
        for index in range(0,len(self.first_tri)):
            for index_2 in range(0,len(self.first_tri[index])):
                for index_3 in range(0,len(self.first_tri[index][index_2])):
                    self.middle_triangle[index][index_2][index_3] =  (1 - alpha) * self.first_tri[index][index_2][index_3] + (alpha) *  self.second_tri[index][index_2][index_3]

        img_1 = Image.new('RGB', (self.startImage.shape[1], self.startImage.shape[0]), 0)
        image_morphed_1 = np.array(img_1)
        img_2 = Image.new('RGB', (self.endImage.shape[1], self.endImage.shape[0]), 0)
        image_morphed_2 = np.array(img_2)

        for i in range(0,len(self.middle_triangle)):
            object_2 = ColorAffine(self.first_tri[i],self.middle_triangle[i])
            #object_2.get_interpolation_func(self.interpolate_func_1)
            object_3 = ColorAffine(self.second_tri[i],self.middle_triangle[i])
            #object_3.get_interpolation_func(self.interpolate_func_2)
            object_2.transform(self.startImage,image_morphed_1)
            object_3.transform(self.endImage,image_morphed_2)
        image_1 = Image.fromarray(image_morphed_1)
        image_2 = Image.fromarray(image_morphed_2)
        image_morphed = Image.blend(image_1,image_2,alpha)
        return np.array(image_morphed)

    def generateMorphVideo(self,targetFolderPath,sequenceLength,includeReversed=True):
        count = sequenceLength
        i = 0
        if os.path.isdir(targetFolderPath) == False:
            os.mkdir(targetFolderPath)
        if includeReversed == True:
            array = np.linspace(0,1,num=count)
            for alpha in array:
                i = i + 1
                if alpha == 0:
                    image_result = Image.fromarray(self.startImage)
                elif alpha == 1:
                    image_result = Image.fromarray(self.endImage)
                else:
                    image_result_array = self.getBlendedImage(alpha)
                    image_result = Image.fromarray(image_result_array)
                save_string = targetFolderPath + '/frame{0:03d}.jpg'.format(i)
                image_result.save(save_string)
                save_string = targetFolderPath + '/frame{0:03d}.jpg'.format(2 * count - i + 1)
                image_result.save(save_string)

        else:
            array = np.linspace(0,1,num=count)
            i = i + 1
            for alpha in array:
                if alpha == 0:
                    image_result = Image.fromarray(self.startImage)
                elif alpha == 1:
                    image_result = Image.fromarray(self.endImage)
                else:
                    image_result_array = self.getBlendedImage(alpha)
                    image_result = Image.fromarray(image_result_array)
                save_string = targetFolderPath + '/frame{0:03d}.jpg'.format(i)
                image_result.save(save_string)
        sp.call("ffmpeg -y -framerate 5 -i " + targetFolderPath + "/frame%03d.jpg -r 5 " + targetFolderPath + "/morph.mp4",shell=True)


if __name__ == "__main__":
    f = open('tiger2.jpg.txt')
    f1 = open('wolf.jpg.txt')
    counter = 0

    for i in f.readlines():
        import re
        counter = counter + 1
        pattern = r'[0-9]+'
        m = re.findall(pattern,i)
        d = np.array([[m[0],m[1]]],dtype=np.float64)
        e = np.array([[m[1],m[0]]],dtype=np.float64)
        if counter == 1:
            startPoints = np.array([[m[0],m[1]]],dtype=np.float64)
            startPoints_2 = np.array([[m[1],m[0]]],dtype=np.float64)
        else:
            startPoints = np.append(startPoints,d,0)
            startPoints_2 = np.append(startPoints_2,e,0)

    counter = 0
    for i in f1.readlines():
        import re

        counter = counter + 1
        pattern = r'[0-9]+'
        m = re.findall(pattern, i)
        d = np.array([[m[0], m[1]]], dtype=np.float64)
        e = np.array([[m[1], m[0]]], dtype=np.float64)
        if counter == 1:
            endPoints = np.array([[m[0], m[1]]], dtype=np.float64)
            endPoints_2 = np.array([[m[1], m[0]]], dtype=np.float64)
        else:
            endPoints = np.append(endPoints, d, 0)
            endPoints_2 = np.append(endPoints_2,e,0)

    startImage = imageio.imread('Tiger2Gray.jpg')
    startImage_color = imageio.imread('morph_start.jpeg')
    endImage = imageio.imread('WolfGray.jpg')
    endImage_color = imageio.imread('morph_end.jpg')
    startImage_color_2 = imageio.imread('morph_5.jpg')
    endImage_color_2 = imageio.imread('morph_end.jpg')
    object = ColorBlender(startImage_color,startPoints,endImage_color,endPoints)
    image_1 = Image.fromarray(object.getBlendedImage(0.5))
    image_1.show()









