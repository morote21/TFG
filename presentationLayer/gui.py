import PyQt6
import PyQt6.QtCore
import PyQt6.QtGui
import PyQt6.QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout, QPushButton, QLabel
import cv2
import sys
from domainLayer.main import executeStatisticsGeneration

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.argsDict = {
            "videoPath": None,
            "team1Path": None,
            "team2Path": None
        }
        self.setWindowTitle("Basketball Statistics Generator")
        
        self.resolution = PyQt6.QtGui.QGuiApplication.primaryScreen().size()
        self.setGeometry(0, 0, self.resolution.width()//2, self.resolution.height()//2)

        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)

        mainLayout = QVBoxLayout()
        mainWidget.setLayout(mainLayout)

        videoLabel = QLabel("Insert video to analyze:")
        mainLayout.addWidget(videoLabel)

        videoInsertButton = QPushButton("Insert Video")
        videoInsertButton.clicked.connect(self.insertVideo)
        mainLayout.addWidget(videoInsertButton)

        team1Label = QLabel("Insert team 1 image:")
        mainLayout.addWidget(team1Label)

        team1InsertButton = QPushButton("Insert Team 1")
        team1InsertButton.clicked.connect(self.insertTeam1)
        mainLayout.addWidget(team1InsertButton)

        team2Label = QLabel("Insert team 2 image:")
        mainLayout.addWidget(team2Label)

        team2InsertButton = QPushButton("Insert Team 2")
        team2InsertButton.clicked.connect(self.insertTeam2)
        mainLayout.addWidget(team2InsertButton)

        statisticsGenerateButton = QPushButton("Generate Statistics")
        statisticsGenerateButton.clicked.connect(self.generateStatistics)
        mainLayout.addWidget(statisticsGenerateButton)
    
    def insertVideo(self):
        videoPath, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if videoPath:
            self.argsDict["videoPath"] = videoPath
        else:
            print("No video selected")
    
    def insertTeam1(self):
        team1Path, _ = QFileDialog.getOpenFileName(self, "Select Team 1", "", "Image Files (*.png *.jpg)")
        if team1Path:
            self.argsDict["team1Path"] = team1Path
        else:
            print("No team 1 selected")
    
    def insertTeam2(self):
        team2Path, _ = QFileDialog.getOpenFileName(self, "Select Team 2", "", "Image Files (*.png *.jpg)")
        if team2Path:
            self.argsDict["team2Path"] = team2Path
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