import PyQt6
import PyQt6.QtCore
import PyQt6.QtGui
from PyQt6.QtGui import QPixmap, QIcon
import PyQt6.QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget, QListWidgetItem, QScrollArea
import cv2
import sys
from domainLayer.main import executeStatisticsGeneration
from pathlib import Path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.sceneSetted = False
        self.videoSetted = False
        self.team1Setted = False
        self.team2Setted = False
        self.nTeams = 1

        self.argsDict = {
            "videoPath": None,
            "team1Path": None,
            "team2Path": None,
            "courtSide": None,
            "scenePointsPath": None
        }
        self.setWindowTitle("Basketball Statistics Generator")
        
        self.resolution = PyQt6.QtGui.QGuiApplication.primaryScreen().size()
        self.setGeometry(0, 0, self.resolution.width()//2, self.resolution.height()//2)

        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)

        # set horizontal layout
        mainLayout = QVBoxLayout()
        mainWidget.setLayout(mainLayout)

        # set buttons
        statisticsGeneratorButton = QPushButton("Statistics Generator")
        statisticsGeneratorButton.clicked.connect(self.openStatisticsGenerator)
        mainLayout.addWidget(statisticsGeneratorButton)

        statisticsViewerButton = QPushButton("Statistics Viewer")
        statisticsViewerButton.clicked.connect(self.openStatisticsViewer)
        mainLayout.addWidget(statisticsViewerButton)            


    def openStatisticsGenerator(self):
        self.takeCentralWidget()

        statisticsGeneratorWindow = QWidget()
        self.setCentralWidget(statisticsGeneratorWindow)

        statisticsGeneratorLayout = QVBoxLayout()
        statisticsGeneratorWindow.setLayout(statisticsGeneratorLayout)

        sceneLabel = QLabel("Insert scene image:")
        statisticsGeneratorLayout.addWidget(sceneLabel)

        newSceneInsertButton = QPushButton("If new scene, insert video:")
        newSceneInsertButton.clicked.connect(self.insertNewScene)
        statisticsGeneratorLayout.addWidget(newSceneInsertButton)

        # 2 more buttons selecting right side court or left side court
        self.courtSideLabel = QLabel("Choose court side:")
        statisticsGeneratorLayout.addWidget(self.courtSideLabel)
        self.courtSideLabel.hide()

        self.rightSideButton = QPushButton("Right Side")
        self.rightSideButton.clicked.connect(self.setRightSide)
        statisticsGeneratorLayout.addWidget(self.rightSideButton)
        self.rightSideButton.hide()

        self.leftSideButton = QPushButton("Left Side")
        self.leftSideButton.clicked.connect(self.setLeftSide)
        statisticsGeneratorLayout.addWidget(self.leftSideButton)
        self.leftSideButton.hide()
        
        
        self.existentSceneInsertButton = QPushButton("Select existent scene")
        self.existentSceneInsertButton.clicked.connect(self.selectExistentScene)
        statisticsGeneratorLayout.addWidget(self.existentSceneInsertButton)
        self.existentSceneInsertButton.show()

        self.videoLabel = QLabel("Insert video to analyze:")
        statisticsGeneratorLayout.addWidget(self.videoLabel)

        self.videoInsertButton = QPushButton("Insert Video")
        self.videoInsertButton.clicked.connect(self.insertVideo)
        statisticsGeneratorLayout.addWidget(self.videoInsertButton)
        self.videoInsertButton.show()




        nTeamsLabel = QLabel("Choose number of teams:")
        statisticsGeneratorLayout.addWidget(nTeamsLabel)

        oneTeamButton = QPushButton("1 Team")
        oneTeamButton.clicked.connect(self.setOneTeam)
        statisticsGeneratorLayout.addWidget(oneTeamButton)

        twoTeamsButton = QPushButton("2 Teams")
        twoTeamsButton.clicked.connect(self.setTwoTeams)
        statisticsGeneratorLayout.addWidget(twoTeamsButton)

        self.team1Label = QLabel("Insert team 1 image:")
        self.team1Label.hide()
        statisticsGeneratorLayout.addWidget(self.team1Label)

        self.team1InsertButton = QPushButton("Insert Team 1")
        self.team1InsertButton.clicked.connect(self.insertTeam1)
        self.team1InsertButton.hide()
        statisticsGeneratorLayout.addWidget(self.team1InsertButton)

        self.team2Label = QLabel("Insert team 2 image:")
        self.team2Label.hide()
        statisticsGeneratorLayout.addWidget(self.team2Label)

        self.team2InsertButton = QPushButton("Insert Team 2")
        self.team2InsertButton.clicked.connect(self.insertTeam2)
        self.team2InsertButton.hide()
        statisticsGeneratorLayout.addWidget(self.team2InsertButton)

        statisticsGenerateButton = QPushButton("Generate Statistics")
        statisticsGenerateButton.clicked.connect(self.generateStatistics)
        statisticsGeneratorLayout.addWidget(statisticsGenerateButton)

    def openStatisticsViewer(self):
        print("Opening statistics viewer...")

    def setRightSide(self):
        self.argsDict["courtSide"] = "right"
        self.courtSideLabel.hide()
        self.rightSideButton.hide()
        self.leftSideButton.hide()
    
    def setLeftSide(self):
        self.argsDict["courtSide"] = "left"
        self.courtSideLabel.hide()
        self.rightSideButton.hide()
        self.leftSideButton.hide()

    def selectExistentScene(self):
        self.videoLabel.show()
        self.videoInsertButton.show()

        # open scrollable list of scenes, being the scenes images from a folder
        self.listWidget = QListWidget()

        sceneFolder = Path("database/scenes")
        # iterate over each folder in scenesFolder
        scenePaths = [scenePath for scenePath in sceneFolder.iterdir()]      
        for scenePath in scenePaths:
            scenePathImg = scenePath / "firstFrame.png"
            scene = QListWidgetItem() 
            pixmap = QPixmap(str(scenePathImg))
            icon = QIcon(pixmap)
            scene.setIcon(icon)
            scene.setText(str(scenePath))
            self.listWidget.addItem(scene)
        
        self.listWidget.show()
        self.listWidget.itemClicked.connect(self.sceneSelected)

        print("Selecting existent scene...")
    
    def sceneSelected(self, item):
        # print path of item
        scenePath = Path(item.text())
        self.argsDict["scenePointsPath"] = str(scenePath)
        print("Scene selected: ", scenePath)

    def setOneTeam(self):
        self.team1Label.hide()
        self.team1InsertButton.hide()
        self.team2Label.hide()
        self.team2InsertButton.hide()
    
    def setTwoTeams(self):
        self.team1Label.show()
        self.team1InsertButton.show()
        self.team2Label.show()
        self.team2InsertButton.show()

    def insertVideo(self):
        videoPath, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if videoPath:
            self.argsDict["videoPath"] = videoPath
            self.videoSetted = True
        else:
            print("No video selected")
    
    def insertNewScene(self):
        videoPath, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if videoPath:
            self.argsDict["videoPath"] = videoPath
            self.videoLabel.hide()
            self.videoInsertButton.hide()
            self.courtSideLabel.show()
            self.rightSideButton.show()
            self.leftSideButton.show()
            self.existentSceneInsertButton.hide()

            self.videoSetted = True
            self.sceneSetted = True
        else:
            print("No video selected")
    
    def insertTeam1(self):
        team1Path, _ = QFileDialog.getOpenFileName(self, "Select Team 1", "", "Image Files (*.png *.jpg)")
        if team1Path:
            self.argsDict["team1Path"] = team1Path
            self.team1Setted = True
        else:
            print("No team 1 selected")
    
    def insertTeam2(self):
        team2Path, _ = QFileDialog.getOpenFileName(self, "Select Team 2", "", "Image Files (*.png *.jpg)")
        if team2Path:
            self.argsDict["team2Path"] = team2Path
            self.team2Setted = True
        else:
            print("No team 2 selected")
    
    def generateStatistics(self):
        print("Generating statistics...")
        executeStatisticsGeneration(self.argsDict)
            
def initializeApp():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())