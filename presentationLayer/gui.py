import PyQt6
import PyQt6.QtCore
import PyQt6.QtGui
from PyQt6.QtGui import QPixmap, QIcon
import PyQt6.QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QListWidgetItem, QScrollArea
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

        self.mainMenu()



    def mainMenu(self):
        self.takeCentralWidget()

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


    def openStatisticsViewer(self):
        self.takeCentralWidget()

        statisticsViewerWindow = QWidget()
        self.setCentralWidget(statisticsViewerWindow)



    def openStatisticsGenerator(self):
        self.takeCentralWidget()

        statisticsGeneratorWindow = QWidget()
        self.setCentralWidget(statisticsGeneratorWindow)

        statisticsGeneratorLayout = QVBoxLayout()
        statisticsGeneratorWindow.setLayout(statisticsGeneratorLayout)

        topLayout = QHBoxLayout()

        backButton = QPushButton("Back")
        backButton.clicked.connect(self.backToMain)
        topLayout.addWidget(backButton)

        topLayout.addStretch()

        resetButton = QPushButton("Reset")
        resetButton.clicked.connect(self.resetParameters)
        topLayout.addWidget(resetButton)

        statisticsGeneratorLayout.addLayout(topLayout)

        statisticsGeneratorLayout.addStretch()

        self.sceneSelectionLayout = QVBoxLayout()
        sceneLabel = QLabel("Insert scene image:")
        self.sceneSelectionLayout.addWidget(sceneLabel)

        buttonsLayout = QHBoxLayout()
        self.newSceneInsertButton = QPushButton("If new scene, insert video:")
        self.newSceneInsertButton.clicked.connect(self.insertNewScene)
        buttonsLayout.addWidget(self.newSceneInsertButton)

        self.existentSceneInsertButton = QPushButton("Select existent scene")
        self.existentSceneInsertButton.clicked.connect(self.selectExistentScene)
        buttonsLayout.addWidget(self.existentSceneInsertButton)

        self.sceneSelectionLayout.addLayout(buttonsLayout)

        # 2 more buttons selecting right side court or left side court

        courtSideLayout = QVBoxLayout()
        courtSideLabel = QLabel("Choose court side:")
        courtSideLayout.addWidget(courtSideLabel)

        courtSideButtonsLayout = QHBoxLayout()
        self.rightSideButton = QPushButton("Left Side")
        self.rightSideButton.clicked.connect(self.setLeftSide)
        courtSideButtonsLayout.addWidget(self.rightSideButton)
        #rightSideButton.hide()

        self.leftSideButton = QPushButton("Right Side")
        self.leftSideButton.clicked.connect(self.setRightSide)
        courtSideButtonsLayout.addWidget(self.leftSideButton)
        #leftSideButton.hide()
        
        courtSideLayout.addLayout(courtSideButtonsLayout)
        self.sceneSelectionLayout.addLayout(courtSideLayout)
        
        statisticsGeneratorLayout.addLayout(self.sceneSelectionLayout)
        
        statisticsGeneratorLayout.addStretch()

        videoInsertLayout = QVBoxLayout()
        videoLabel = QLabel("Insert video to analyze:")
        videoInsertLayout.addWidget(videoLabel)

        self.videoInsertButton = QPushButton("Insert Video")
        self.videoInsertButton.clicked.connect(self.insertVideo)
        videoInsertLayout.addWidget(self.videoInsertButton)
        
        statisticsGeneratorLayout.addLayout(videoInsertLayout)

        statisticsGeneratorLayout.addStretch()
        
        nTeamsLayout = QVBoxLayout()
        nTeamsLabel = QLabel("Choose number of teams:")
        nTeamsLayout.addWidget(nTeamsLabel)

        nTeamsButtonsLayout = QHBoxLayout()
        self.noTeamButton = QPushButton("No Teams")
        self.noTeamButton.clicked.connect(self.setNoTeams)
        nTeamsButtonsLayout.addWidget(self.noTeamButton)


        self.twoTeamsButton = QPushButton("2 Teams")
        self.twoTeamsButton.clicked.connect(self.setTwoTeams)
        nTeamsButtonsLayout.addWidget(self.twoTeamsButton)

        nTeamsLayout.addLayout(nTeamsButtonsLayout)
        statisticsGeneratorLayout.addLayout(nTeamsLayout)

        equipmentLayout = QVBoxLayout()
        equipmentLabel = QLabel("If 2 teams, insert equipment for each team:")
        equipmentLayout.addWidget(equipmentLabel)
        
        equipmentButtonsLayout = QHBoxLayout()
        self.team1InsertButton = QPushButton("Insert Team 1")
        self.team1InsertButton.clicked.connect(self.insertTeam1)
        equipmentButtonsLayout.addWidget(self.team1InsertButton)

        self.team2InsertButton = QPushButton("Insert Team 2")
        self.team2InsertButton.clicked.connect(self.insertTeam2)
        equipmentButtonsLayout.addWidget(self.team2InsertButton)

        equipmentLayout.addLayout(equipmentButtonsLayout)
        statisticsGeneratorLayout.addLayout(equipmentLayout)

        statisticsGeneratorLayout.addStretch()

        statisticsGenerateButton = QPushButton("Generate Statistics")
        statisticsGenerateButton.clicked.connect(self.generateStatistics)
        statisticsGeneratorLayout.addWidget(statisticsGenerateButton)


    def openStatisticsViewer(self):
        print("Opening statistics viewer...")

    def setRightSide(self):
        self.argsDict["courtSide"] = "right"
        self.rightSideButton.setEnabled(False)
        self.leftSideButton.setEnabled(False)
    
    def setLeftSide(self):
        self.argsDict["courtSide"] = "left"
        self.rightSideButton.setEnabled(False)
        self.leftSideButton.setEnabled(False)

    def selectExistentScene(self):

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
        self.listWidget.itemDoubleClicked.connect(self.sceneSelected)

        print("Selecting existent scene...")
    
    def sceneSelected(self, item):
        # print path of item
        scenePath = Path(item.text())
        self.argsDict["scenePointsPath"] = str(scenePath)

        self.rightSideButton.setEnabled(False)
        self.leftSideButton.setEnabled(False)
        self.newSceneInsertButton.setEnabled(False)
        self.existentSceneInsertButton.setEnabled(False)

        # close qlistwidget when clicking on item
        self.listWidget.hide()
        self.listWidget.close()

        print("Scene selected: ", scenePath)

    def setNoTeams(self):
        self.team1InsertButton.setEnabled(False)
        self.team2InsertButton.setEnabled(False)
    
    def setTwoTeams(self):
        self.team1InsertButton.setEnabled(True)
        self.team2InsertButton.setEnabled(True)

    def insertVideo(self):
        videoPath, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if videoPath:
            self.argsDict["videoPath"] = videoPath
            self.videoSetted = True

            self.videoInsertButton.setEnabled(False)

        else:
            print("No video selected")
    
    def insertNewScene(self):
        videoPath, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if videoPath:
            self.argsDict["videoPath"] = videoPath
            
            self.newSceneInsertButton.setEnabled(False)
            self.existentSceneInsertButton.setEnabled(False)
            self.videoInsertButton.setEnabled(False)
            
        else:
            print("No video selected")
    
    def insertTeam1(self):
        team1Path, _ = QFileDialog.getOpenFileName(self, "Select Team 1", "", "Image Files (*.png *.jpg)")
        if team1Path:
            self.argsDict["team1Path"] = team1Path
            self.team1Setted = True

            self.noTeamButton.setEnabled(False)
            self.twoTeamsButton.setEnabled(False)

            if self.team1Setted and self.team2Setted:
                self.team1InsertButton.setEnabled(False)
                self.team2InsertButton.setEnabled(False)

        else:
            print("No team 1 selected")
    
    def insertTeam2(self):
        team2Path, _ = QFileDialog.getOpenFileName(self, "Select Team 2", "", "Image Files (*.png *.jpg)")
        if team2Path:
            self.argsDict["team2Path"] = team2Path
            self.team2Setted = True

            self.noTeamButton.setEnabled(False)
            self.twoTeamsButton.setEnabled(False)

            if self.team1Setted and self.team2Setted:
                self.team1InsertButton.setEnabled(False)
                self.team2InsertButton.setEnabled(False)

        else:
            print("No team 2 selected")
    
    def generateStatistics(self):
        print("Generating statistics...")
        executeStatisticsGeneration(self.argsDict)

    def resetParameters(self):
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

        self.rightSideButton.setEnabled(True)
        self.leftSideButton.setEnabled(True)
        self.newSceneInsertButton.setEnabled(True)
        self.existentSceneInsertButton.setEnabled(True)
        self.videoInsertButton.setEnabled(True)
        self.noTeamButton.setEnabled(True)
        self.twoTeamsButton.setEnabled(True)
        self.team1InsertButton.setEnabled(True)
        self.team2InsertButton.setEnabled(True)


    
    def backToMain(self):
        self.mainMenu()
            
def initializeApp():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())