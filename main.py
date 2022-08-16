from mainwindow import Ui_MainWindow
from FeatureMatching import *
from Harris import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import numpy as np
import sys

class App(QtWidgets.QMainWindow):
    
    first_path = ""
    second_path = ""
    third_path = ""
 
    
    
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.Import_H.clicked.connect(self.load_data)
        self.ui.Feature_import.clicked.connect(self.load_data2)
        self.ui.Template_Import.clicked.connect(self.feature_matching)
        self.ui.Show_Harris.clicked.connect(self.Apply_Harris)

        
     
        
    def load_data(self):
        filepath = QFileDialog.getOpenFileName(self)
        if filepath[0]:
            self.first_path = filepath[0]
        
        img = cv2.imread(self.first_path)
        self.ui.Import.show()
        self.ui.Import.setImage(np.rot90(img,1))

    def Apply_Harris (self):
        Response_Mat= Harris_Edge_Detector(self.first_path,3,0.05)
        src = cv2.imread(self.first_path)
        max_corner = np.max(Response_Mat)
        corner = np.array(Response_Mat > (max_corner * 0.01), dtype="int8")
        src = src[:corner.shape[0], :corner.shape[1]]
        src[corner == 1] = [255,0,0]

        self.ui.Harris.show()
        self.ui.Harris.setImage(np.rot90(src,1))


    def load_data2(self):
        filepath = QFileDialog.getOpenFileName(self)
        if filepath[0]:
            self.second_path = filepath[0]
        img = cv2.imread(self.second_path)
        self.ui.img.show()
        self.ui.img.setImage(np.rot90(img,1))
        
        
    def feature_matching(self):
        filepath = QFileDialog.getOpenFileName(self)
        if filepath[0]:
            self.third_path = filepath[0]
        img = cv2.imread(self.third_path)
        self.ui.temp.show()
        self.ui.temp.setImage(np.rot90(img,1))

        original_img = cv2.imread(self.second_path)
        template_img =cv2.imread(self.third_path)

        original_img = cv2.resize(original_img,(256,256))
        template_img = cv2.resize(template_img,(256,256))

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create() 
        kp1, descriptor1 = sift.detectAndCompute(original_img, None)
        kp2, descriptor2 = sift.detectAndCompute(template_img, None)

        matched_features = feature_matching_temp(descriptor1, descriptor2,"SSD")
        matched_features = sorted(matched_features, key=lambda x: x.distance, reverse=True)
        matched_image_SSD = cv2.drawMatches(original_img, kp1, template_img, kp2,matched_features[:30], template_img, flags=2)
        
        self.ui.SSD.show()
        self.ui.SSD.setImage(np.rot90(matched_image_SSD,1))

        
        matched_features = feature_matching_temp(descriptor1, descriptor2,"NCC")
        matched_features = sorted(matched_features, key=lambda x: x.distance, reverse=True)
        matched_image_NCC = cv2.drawMatches(original_img, kp1, template_img, kp2,matched_features[:30], template_img, flags=2)

        self.ui.NCC.show()
        self.ui.NCC.setImage(np.rot90(matched_image_NCC,1))
   


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = App()
    application.show()
    app.exec_()
    


if __name__ == "__main__":
    main()
