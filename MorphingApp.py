import os
import re
import sys
import imageio
import Morphing
import numpy as np
from MorphingGUI import *
from PySide.QtGui import *
from PySide.QtCore import *

class Consumer(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(Consumer, self).__init__(parent)
        self.setupUi(self)
        self.trueSim = None
        self.trueEim = None
        self.sImage = None
        self.eImage = None
        self.sPoints = None
        self.ePoints = None
        self.holdStart = None
        self.holdEnd = None
        self.extensioner = None
        self.trueRatio = None
        self.startFile = None
        self.endFile = None
        self.curSceneS = None
        self.curSceneE = None
        self.backStart = None
        self.backEnd = None

        self.slider = 0
        self.widther = 0
        self.heightr = 0

        self.indicateLoad = 0
        self.indicateDela = 0
        self.indicateBlend = 0
        self.indicateP1 = 0
        self.indicateP2 = 0
        self.indicatePersist = 0
        self.indicateNoPrior = 0
        self.sx = 0
        self.sy = 0
        self.ex = 0
        self.ey = 0

        self.cS = 0
        self.eS = 0

        self.horizontalSlider.setEnabled(False)
        self.lineEdit.setEnabled(False)
        self.blendButton.setEnabled(False)
        self.checkBox.setEnabled(False)
        self.startButton.setEnabled(True)
        self.endButton.setEnabled(True)

        self.startButton.clicked.connect(self.loadDataStart)
        self.endButton.clicked.connect(self.loadDataEnd)

    def loadImageStartEnd(self, filePath, sorePoints):
        beginer = QPixmap(filePath)
        Alt = QGraphicsScene()

        myPainter = QPainter()
        myPainter.begin(beginer)
        myPainter.setPen(Qt.red)

        for i in sorePoints:
            x = QPoint(i[0], i[1])
            myPainter.drawEllipse(x, 5, 5)
            myPainter.setBrush(Qt.red)

        myPainter.end()
        Alt.addPixmap(self.myScaler(beginer))

        unAlt = beginer.copy()
        copyPix = beginer.copy()

        return Alt, unAlt, copyPix

    def loadImageNoPrior(self, filePath):
        beginer = QPixmap(filePath)
        Alt = QGraphicsScene()
        Alt.addPixmap(self.myScaler(beginer))

        unAlt = beginer.copy()
        copyPix = beginer.copy()

        return Alt, unAlt, copyPix

    def loadDataFromFileStart(self, filePath):
        imger = imageio.imread(filePath)
        self.sImage = imger
        self.sPoints = self.getPoints(filePath)
        if len(self.sPoints) > 0:
            temp, self.trueSim, self.holdStart = self.loadImageStartEnd(filePath, self.sPoints)
        else:
            temp, self.trueSim, self.holdStart = self.loadImageNoPrior(filePath)
        self.startFrame.setScene(temp)
        self.startFrame.setSceneRect(QRectF(self.startFrame.viewport().rect()))

        if self.indicateLoad == 0:
            self.indicateLoad = 1
        elif self.indicateLoad == 2:
            self.loadedState()

        self.extensioner = filePath.split('.')[1]
        self.startFile = filePath + ".txt"

        if len(self.sPoints) == 0:
            self.indicatePrior = -1

    def loadDataFromFileEnd(self, filePath):
        imger = imageio.imread(filePath)
        self.eImage = imger
        self.ePoints = self.getPoints(filePath)
        if len(self.ePoints) > 0:
            temp, self.trueEim, self.holdEnd = self.loadImageStartEnd(filePath, self.ePoints)
        else:
            temp, self.trueEim, self.holdEnd = self.loadImageNoPrior(filePath)
        self.endFrame.setScene(temp)
        self.endFrame.setSceneRect(QRectF(self.endFrame.viewport().rect()))

        if self.indicateLoad == 0:
            self.indicateLoad = 2
        elif self.indicateLoad == 1:
            self.loadedState()

        self.endFile = filePath + ".txt"

        if len(self.ePoints) == 0:
            self.indicatePrior = -1

        tempDim = QPixmap(filePath)
        self.widther = tempDim.width()
        self.heightr = tempDim.height()
        if tempDim.width() / 400 > tempDim.height() / 300:
            self.trueRatio = tempDim.width() / 400
        else:
            self.trueRatio = tempDim.height() / 300

    def loadDataStart(self):
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open image file ...', filter="JPG/PNG files (*.jpg *.png)")

        if not filePath:
            return

        self.loadDataFromFileStart(filePath)

    def loadDataEnd(self):
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open image file ...', filter="JPG/PNG files (*.jpg *.png)")

        if not filePath:
            return

        self.loadDataFromFileEnd(filePath)

    def loadedState(self):
        self.horizontalSlider.setEnabled(True)
        self.horizontalSlider.sliderMoved.connect(self.slideFrankOcean)

        self.lineEdit.setEnabled(True)
        self.lineEdit.setReadOnly(True)

        self.blendButton.setEnabled(True)
        self.blendButton.clicked.connect(self.blendaSplenda)

        self.checkBox.setEnabled(True)
        self.checkBox.clicked.connect(self.showDelauney)

        self.startButton.setEnabled(True)
        self.endButton.setEnabled(True)

        self.startFrame.installEventFilter(self)
        self.endFrame.installEventFilter(self)
        self.installEventFilter(self)

    def blendaSplenda(self):
        if len(self.sImage.shape) > 2:
            blendy = Morphing.ColorBlender(self.sImage, self.sPoints, self.eImage, self.ePoints)
        else:
            blendy = Morphing.Blender(self.sImage, self.sPoints, self.eImage, self.ePoints)
        tempImg = blendy.getBlendedImage(self.slider)
        tempImg = Morphing.Image.fromarray(tempImg)
        tempImg.save('tempBlendShow.' + self.extensioner)

        # holdr = QPixmap(Morphing.Image.fromarray(QImage(tempImg)))
        holdr = QPixmap('tempBlendShow.' + self.extensioner)
        blendScene = QGraphicsScene()
        blendScene.addPixmap(self.myScaler(holdr))
        self.blendFrame.setScene(blendScene)
        self.blendFrame.setSceneRect(QRectF(self.blendFrame.viewport().rect()))

    def slideFrankOcean(self):
        self.slider = self.horizontalSlider.sliderPosition() / 20
        self.lineEdit.setText(str(self.slider))

    def blendItUp(self):
        if self.indicateBlend == 0:
            if not os.path.isdir("AlphaImages"):
                os.mkdir("AlphaImages")

            leny = range(0, 21, 1)
            if len(self.sImage.shape) > 2:
                blendy = Morphing.ColorBlender(self.sImage, self.sPoints, self.eImage, self.ePoints)
            else:
                blendy = Morphing.Blender(self.sImage, self.sPoints, self.eImage, self.ePoints)

            for alpha in leny:
                waitey = blendy.getBlendedImage(alpha * 0.05)
                imgr = Morphing.Image.fromarray(waitey)
                if len(waitey.shape) > 2:
                    imgr = imgr.convert('RGB')
                else:
                    imgr = imgr.convert('L')

                holup = "AlphaImages/frame{0}.{1}" .format(alpha, self.extensioner)
                imgr.save(holup)

            self.indicateBlend = 1
        else:
            numer = self.horizontalSlider.tickInterval()
            holdIm = QPixmap("AlphaImages/frame" + str(numer / 5) + "." + self.extensioner)
            self.blendFrame.setPixmap(self.myScaler(holdIm))

    def getPoints(self, filePath):
        if os.path.isfile(filePath + ".txt"):
            with open(filePath + ".txt", 'r') as filer:
                lines = filer.readlines()

            temp1 = re.search("\D+(\d+)\D+(\d+)", lines[0])
            arrs = np.array([[temp1.group(1), temp1.group(2)]], dtype=np.float64)

            for i in range(1, len(lines)):
                x = re.search("\D+(\d+)\D+(\d+)", lines[i])
                y = np.array([[x.group(1), x.group(2)]], dtype=np.float64)
                arrs = np.append(arrs, y, axis=0)

            return arrs
        else:
            self.indicateNoPrior = 1
            return np.array([], dtype=np.float64)

    def showDelauney(self):
        if self.indicateDela == 0:
            if len(self.sPoints) >= 3 and len(self.ePoints) >= 3:
                witer = Morphing.Delaunay(self.sPoints)
                tringles = self.sPoints[witer.simplices]
                etringles = self.ePoints[witer.simplices]

                temp = QGraphicsScene()
                tempPainter = QPainter()
                tempPainter.begin(self.holdStart)
                tempPainter.setPen(Qt.cyan)

                for i in tringles:
                    x = [QPoint(i[0][0], i[0][1]), QPoint(i[1][0], i[1][1]), QPoint(i[2][0], i[2][1])]
                    y = QPolygon(x)
                    tempPainter.drawPolygon(y)

                tempPainter.end()
                temp.addPixmap(self.myScaler(self.holdStart))
                self.startFrame.setScene(temp)
                self.startFrame.setSceneRect(QRectF(self.startFrame.viewport().rect()))

                temp = QGraphicsScene()
                tempPainter = QPainter()
                tempPainter.begin(self.holdEnd)
                tempPainter.setPen(Qt.cyan)

                for i in etringles:
                    x = [QPoint(i[0][0], i[0][1]), QPoint(i[1][0], i[1][1]), QPoint(i[2][0], i[2][1])]
                    y = QPolygon(x)
                    tempPainter.drawPolygon(y)

                tempPainter.end()
                temp.addPixmap(self.myScaler(self.holdEnd))
                self.endFrame.setScene(temp)
                self.endFrame.setSceneRect(QRectF(self.endFrame.viewport().rect()))

                self.indicateDela = 1
        else:
            temp = QGraphicsScene()
            temp.addPixmap(self.myScaler(self.trueSim))
            self.startFrame.setScene(temp)
            self.startFrame.setSceneRect(QRectF(self.startFrame.viewport().rect()))

            temp = QGraphicsScene()
            temp.addPixmap(self.myScaler(self.trueEim))
            self.endFrame.setScene(temp)
            self.endFrame.setSceneRect(QRectF(self.endFrame.viewport().rect()))

            self.indicateDela = 0

    def myScaler(self, toBeScaled, x=400, y=300):
        return toBeScaled.scaled(x, y, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def getFileLen(self):
        if os.path.isfile(self.startFile) and os.path.isfile(self.endFile):
            with open(self.startFile, 'r') as filer:
                lenStart = filer.readlines()
            with open(self.endFile, 'r') as holdr:
                lenEnd = holdr.readlines()
            return len(lenStart), len(lenEnd)
        else:
            with open(self.startFile, 'a') as filer:
                lenStart = 0 #filer.readlines()
            with open(self.endFile, 'a') as holdr:
                lenEnd = 0 #holdr.readlines()
            return lenStart, lenEnd

    def eventFilter(self, curr_widget, event):
        flaggy = 0
        if event.type() == QEvent.MouseButtonPress:
            x = event.pos().x()
            y = event.pos().y()
            lenStart, lenEnd = self.getFileLen()

            if lenStart == lenEnd and curr_widget == self.startFrame:
                if self.indicatePersist == 1:
                    self.persist()
                else:
                    self.startPress(x, y)
            elif lenStart > lenEnd and curr_widget == self.endFrame:
                self.endPress(x, y)
            elif lenStart == lenEnd and curr_widget != self.endFrame and self.indicatePersist == 1:
                self.persist()
            flaggy = 1

        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Backspace:
                lenStart, lenEnd = self.getFileLen()

                if lenEnd != len(self.ePoints):
                    self.endFrame.setScene(self.backEnd)
                    self.endFrame.setSceneRect(QRectF(self.endFrame.viewport().rect()))
                    self.trueEim = self.curSceneE
                    self.holdEnd = self.myScaler(self.curSceneE, self.widther, self.widther)
                    self.indicatePersist = 0

                    with open(self.endFile, 'r') as tempR:
                        waitR = tempR.readlines()
                        waitR = waitR[:-1]
                        with open(self.endFile, 'w') as tempW:
                            for i in range(len(waitR)):
                                if i != len(waitR) - 1:
                                    tempW.write(waitR[i])
                                else:
                                    tempW.write(waitR[i].rstrip())

                elif lenStart != len(self.sPoints):
                    self.startFrame.setScene(self.backStart)
                    self.startFrame.setSceneRect(QRectF(self.startFrame.viewport().rect()))
                    self.trueSim = self.curSceneS
                    self.holdStart = self.myScaler(self.curSceneS, self.widther, self.heightr)

                    with open(self.startFile, 'r') as tempR:
                        waitR = tempR.readlines()
                        waitR = waitR[:-1]
                        with open(self.startFile, 'w') as tempW:
                            for i in range(len(waitR)):
                                if i != len(waitR) - 1:
                                    tempW.write(waitR[i])
                                else:
                                    tempW.write(waitR[i].rstrip())
            flaggy = 1

        if flaggy == 0:
            return False
        else:
            return True

    def startPress(self, x, y):
        self.sx = x
        self.sy = y

        lineAn = "\n" * (not self.indicateNoPrior)
        writery = lineAn + " "+ str(int(x * self.trueRatio)) + " " + str(int(y * self.trueRatio))
        with open(self.startFile, 'a') as holer:
            holer.write(writery)

        whold, reassignPic, self.backStart, self.curSceneS = self.setPoint(self.trueSim, x, y, Qt.green)
        self.startFrame.setScene(whold)
        self.startFrame.setSceneRect(QRectF(self.startFrame.viewport().rect()))
        self.trueSim = reassignPic
        self.holdStart = self.myScaler(reassignPic, self.widther, self.heightr)

    def endPress(self, x, y):
        self.ex = x
        self.ey = y

        lineAn = "\n" * (not self.indicateNoPrior)
        writery = lineAn + " " + str(int(x * self.trueRatio)) + " " + str(int(y * self.trueRatio))
        with open(self.endFile, 'a') as holer:
            holer.write(writery)

        whold, reassignPic, self.backEnd, self.curSceneE = self.setPoint(self.trueEim, x, y, Qt.green)
        self.endFrame.setScene(whold)
        self.endFrame.setSceneRect(QRectF(self.endFrame.viewport().rect()))
        self.trueEim = reassignPic
        self.holdEnd = self.myScaler(reassignPic, self.widther, self.heightr)

        self.indicatePersist = 1

    def persist(self):
        self.indicatePersist = 0
        if self.indicateNoPrior != 1:
            self.sPoints = np.append(self.sPoints, np.array([[self.sx * self.trueRatio, self.sy * self.trueRatio]],
                                                        dtype=np.float64), axis=0)
            self.ePoints = np.append(self.ePoints, np.array([[self.ex * self.trueRatio, self.ey * self.trueRatio]],
                                                        dtype=np.float64), axis=0)
        else:
            self.sPoints = np.array([[self.sx * self.trueRatio, self.sy * self.trueRatio]],
                                    dtype=np.float64)
            self.ePoints = np.array([[self.ex * self.trueRatio, self.ey * self.trueRatio]],
                                    dtype=np.float64)
            self.indicateNoPrior = 0

        whold, reassignPic, reP, _ = self.setPoint(self.trueSim, self.sx, self.sy, Qt.blue)
        self.startFrame.setScene(whold)
        self.startFrame.setSceneRect(QRectF(self.startFrame.viewport().rect()))
        self.trueSim = reassignPic
        self.holdStart = self.myScaler(reassignPic, self.widther, self.heightr)

        whold, reassignPic, reP, _ = self.setPoint(self.trueEim, self.ex, self.ey, Qt.blue)
        self.endFrame.setScene(whold)
        self.endFrame.setSceneRect(QRectF(self.endFrame.viewport().rect()))
        self.trueEim = reassignPic
        self.holdEnd = self.myScaler(reassignPic, self.widther, self.heightr)

        if self.indicateDela == 1:
            self.indicateDela = 0
            self.showDelauney()

    def setPoint(self, img, x, y, col):
        whold = QGraphicsScene()
        temp = QGraphicsScene()
        myP = QPainter()
        scaledPic = self.myScaler(img)
        scaledBef = self.myScaler(img)
        temp.addPixmap(scaledPic)
        myP.begin(scaledPic)
        myP.setPen(col)
        myP.setBrush(col)
        myP.drawEllipse(QPoint(x, y), 3, 3)
        myP.end()
        whold.addPixmap(scaledPic)

        return whold, scaledPic, temp, scaledBef

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = Consumer()
    currentForm.show()
    currentApp.exec_()