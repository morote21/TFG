import PyQt6
import PyQt6.QtCore
import PyQt6.QtGui
from PyQt6.QtGui import QPixmap, QIcon
import PyQt6.QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QListWidgetItem, QScrollArea, QSizePolicy
import cv2
import sys
from domainLayer.main import executeStatisticsGeneration
from pathlib import Path
import json

def clearLayout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.sceneSetted = False
        self.videoSetted = False
        self.team1Setted = False
        self.team2Setted = False
        self.nTeams = 1

        self.statisticsToShow = False
        self.statisticsSelected = None

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

        mainLayout.addStretch()

        generatorButtonLayout = QHBoxLayout()
        generatorButtonLayout.addStretch()

        # set buttons
        statisticsGeneratorButton = QPushButton("Statistics Generator")
        statisticsGeneratorButton.clicked.connect(self.openStatisticsGenerator)
        statisticsGeneratorButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        statisticsGeneratorButton.setMinimumSize(200, 75)  
        statisticsGeneratorButton.setMaximumSize(200, 75)  
        generatorButtonLayout.addWidget(statisticsGeneratorButton)

        generatorButtonLayout.addStretch()

        mainLayout.addLayout(generatorButtonLayout)

        mainLayout.addStretch()

        viewerButtonLayout = QHBoxLayout()
        viewerButtonLayout.addStretch()

        statisticsViewerButton = QPushButton("Statistics Viewer")
        statisticsViewerButton.clicked.connect(self.openStatisticsViewer)
        statisticsViewerButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        statisticsViewerButton.setMinimumSize(200, 75)  
        statisticsViewerButton.setMaximumSize(200, 75)  
        viewerButtonLayout.addWidget(statisticsViewerButton)

        viewerButtonLayout.addStretch()

        mainLayout.addLayout(viewerButtonLayout)

        mainLayout.addStretch()

        

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


    
    def openStatisticsViewer(self):
        print("Opening statistics viewer...")
        self.takeCentralWidget()

        statisticsViewerWindow = QWidget()
        self.setCentralWidget(statisticsViewerWindow)

        statisticsViewerLayout = QVBoxLayout()
        statisticsViewerWindow.setLayout(statisticsViewerLayout)

        topLayout = QHBoxLayout()

        backButton = QPushButton("Back")
        backButton.clicked.connect(self.backToMain)
        topLayout.addWidget(backButton)

        topLayout.addStretch()

        statisticsButton = QPushButton("Open statistics")
        statisticsButton.clicked.connect(self.selectStatistics)
        topLayout.addWidget(statisticsButton)

        statisticsViewerLayout.addLayout(topLayout)

        #statisticsViewerLayout.addStretch()

        
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)

        statisticsWidget = QWidget()
        self.vertStatisticsLayout = QVBoxLayout()
        statisticsWidget.setLayout(self.vertStatisticsLayout)

        scrollArea.setWidget(statisticsWidget)

        self.gridStatisticsLayout = QGridLayout()
        self.vertStatisticsLayout.addLayout(self.gridStatisticsLayout)

        statisticsViewerLayout.addWidget(scrollArea)

        self.horizontalStatsLayout = QHBoxLayout()
        self.vertStatisticsLayout.addLayout(self.horizontalStatsLayout)

        self.team1statsLayout = QVBoxLayout()
        self.horizontalStatsLayout.addLayout(self.team1statsLayout)

        self.team2statsLayout = QVBoxLayout()
        self.horizontalStatsLayout.addLayout(self.team2statsLayout)

        self.statisticsToShow = False

        #numericalStats = QLabel("Team stats")
        #self.vertStatisticsLayout.addWidget(numericalStats)



    def selectStatistics(self):
        # open scrollable list of statistics, being the statistics the path of a folder
        self.listWidget = QListWidget()

        statisticsFolder = Path("database/games")
        # iterate over each folder in scenesFolder
        statisticsPaths = [statisticsPath for statisticsPath in statisticsFolder.iterdir()]
        for statisticsPath in statisticsPaths:
            statistics = QListWidgetItem()
            statistics.setText(str(statisticsPath))
            self.listWidget.addItem(statistics)

        self.listWidget.show()
        self.listWidget.itemDoubleClicked.connect(self.statisticsSelection)
    
    def statisticsSelection(self, item):

        clearLayout(self.gridStatisticsLayout)
        clearLayout(self.team1statsLayout)
        clearLayout(self.team2statsLayout)

        # print path of item
        statisticsPath = Path(item.text())
        self.statisticsSelected = str(statisticsPath)

        self.listWidget.hide()
        self.listWidget.close()

        print("Statistics selected: ", statisticsPath)

        temporalGrid = QGridLayout()
        # iterate through images in statisticsPath and add them to the grid, taking into account that not all of the elemnts in the path are images
        statisticsImages = [statisticsImage for statisticsImage in statisticsPath.iterdir() if statisticsImage.suffix == ".png"]
        
        # read json in statisticsPath
        jsonStatisticsPath = statisticsPath / "statistics.json"
        statisticsDict = None
        with open(jsonStatisticsPath) as f:
            statisticsDict = json.load(f)


        if len(statisticsImages) == 6:    
            for statisticsImage in statisticsImages:
                if statisticsImage.name == "Team1ShotTrack.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 0, 0)

                elif statisticsImage.name == "Team2ShotTrack.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 0, 1)

                elif statisticsImage.name == "Team1MotionHeatmap.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 1, 0)
                
                elif statisticsImage.name == "Team2MotionHeatmap.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 1, 1)

                elif statisticsImage.name == "Team1ShotHeatmap.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 2, 0)
                
                elif statisticsImage.name == "Team2ShotHeatmap.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 2, 1)
            
            # vertical layout for numerical statistics team 1
            self.team1statsLayout.addWidget(QLabel("Stats Team 1:"))
            self.team1statsLayout.addWidget(QLabel("FGA: " + str(statisticsDict["Team1"]["FGA"])))
            self.team1statsLayout.addWidget(QLabel("FGM: " + str(statisticsDict["Team1"]["FGM"])))
            self.team1statsLayout.addWidget(QLabel("3PA: " + str(statisticsDict["Team1"]["3PA"])))
            self.team1statsLayout.addWidget(QLabel("3PM: " + str(statisticsDict["Team1"]["3PM"])))

            # vertical layout for numerical statistics team 2
            self.team2statsLayout.addWidget(QLabel("Stats Team 2:"))
            self.team2statsLayout.addWidget(QLabel("FGA: " + str(statisticsDict["Team2"]["FGA"])))
            self.team2statsLayout.addWidget(QLabel("FGM: " + str(statisticsDict["Team2"]["FGM"])))
            self.team2statsLayout.addWidget(QLabel("3PA: " + str(statisticsDict["Team2"]["3PA"])))
            self.team2statsLayout.addWidget(QLabel("3PM: " + str(statisticsDict["Team2"]["3PM"])))


        elif len(statisticsImages) == 3:
            for statisticsImage in statisticsImages:
                if statisticsImage.name == "Team1ShotTrack.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 0, 0)

                elif statisticsImage.name == "Team1MotionHeatmap.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 1, 0)
                
                elif statisticsImage.name == "Team1ShotHeatmap.png":
                    pixmap = QPixmap(str(statisticsImage))
                    label = QLabel()
                    label.setPixmap(pixmap)
                    self.gridStatisticsLayout.addWidget(label, 2, 0)
            
            # vertical layout for numerical statistics team 1
            self.team1statsLayout.addWidget(QLabel("Stats Team 1:"))
            self.team1statsLayout.addWidget(QLabel("FGA: " + str(statisticsDict["Team1"]["FGA"])))
            self.team1statsLayout.addWidget(QLabel("FGM: " + str(statisticsDict["Team1"]["FGM"])))
            self.team1statsLayout.addWidget(QLabel("3PA: " + str(statisticsDict["Team1"]["3PA"])))
            self.team1statsLayout.addWidget(QLabel("3PM: " + str(statisticsDict["Team1"]["3PM"])))


        self.statisticsToShow = True




    def resetViewer(self):
        self.statisticsToShow = False
        self.statisticsSelected = None

        self.gridStatisticsLayout = QGridLayout()

            
def initializeApp():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())