import json
import os
import cv2
from pathlib import Path

def storeStatistics(statisticsDict):
    print("Storing statistics...")

    firstFrame = statisticsDict["firstFrame"]
    scenePoints = statisticsDict["scenePoints"]

    databasePath = Path("./database")
    if not databasePath.exists():
        os.mkdir(databasePath)
    
    
    pathForScenes = databasePath / "scenes"
    if not pathForScenes.exists():
        os.mkdir(pathForScenes)
    
    # get number of files in pathForScenes
    sceneNumber = len(os.listdir(pathForScenes))
    sceneNumber = int(sceneNumber)

    if sceneNumber == 0:
        os.mkdir(pathForScenes / "scene0")
        jsonScenePoints = json.dumps(scenePoints, indent=4)
        with open(pathForScenes / "scene0/scenePoints.json", "w") as f:
            f.write(jsonScenePoints)

        cv2.imwrite(str(pathForScenes / "scene0/firstFrame.png"), firstFrame)
        

    else:
        for i in range(sceneNumber):
            if not os.path.exists(pathForScenes / f"scene{i}"): # if scene{i} does not exist, create it and store first frame and scenePoints
                os.mkdir(pathForScenes / f"scene{i}")

                jsonScenePoints = json.dumps(scenePoints, indent=4)
                with open(pathForScenes / f"scene{i}/scenePoints.json", "w") as f:
                    f.write(jsonScenePoints)

                cv2.imwrite(str(pathForScenes / f"scene{i}/firstFrame.png"), firstFrame)
                break
            


    # statisticsDict to json
    numericalStatistics = {
        "Team1" : {
            "FGA" : statisticsDict["Team1"]["FGA"],
            "3PA" : statisticsDict["Team1"]["3PA"],
        },
        "Team2" : {
            "FGA" : statisticsDict["Team2"]["FGA"],
            "3PA" : statisticsDict["Team2"]["3PA"],
        }
    }

    # store statisticsDict
    pathForStatistics = databasePath / "games"
    if not pathForStatistics.exists():
        os.mkdir(pathForStatistics)
    
    # get number of files in pathForStatistics
    gameNumber = len(os.listdir(pathForStatistics))
    gameNumber = int(gameNumber)

    if gameNumber == 0:
        os.mkdir(pathForStatistics / "game0")
        jsonStatistics = json.dumps(numericalStatistics, indent=4)
        with open(pathForStatistics / "game0/statistics.json", "w") as f:
            f.write(jsonStatistics)
        # Store heatmaps and shot track
        cv2.imwrite(str(pathForStatistics / "game0/Team1MotionHeatmap.png"), statisticsDict["Team1"]["MotionHeatmap"])
        cv2.imwrite(str(pathForStatistics / "game0/Team2MotionHeatmap.png"), statisticsDict["Team2"]["MotionHeatmap"])
        cv2.imwrite(str(pathForStatistics / "game0/Team1ShotHeatmap.png"), statisticsDict["Team1"]["ShotHeatmap"])
        cv2.imwrite(str(pathForStatistics / "game0/Team2ShotHeatmap.png"), statisticsDict["Team2"]["ShotHeatmap"])
        cv2.imwrite(str(pathForStatistics / "game0/Team1ShotTrack.png"), statisticsDict["Team1"]["ShotTrack"])
        cv2.imwrite(str(pathForStatistics / "game0/Team2ShotTrack.png"), statisticsDict["Team2"]["ShotTrack"])

    else:
        for i in range(gameNumber):
            if not os.path.exists(pathForStatistics / f"game{i}"):
                os.mkdir(pathForStatistics / f"game{i}")
                jsonStatistics = json.dumps(numericalStatistics, indent=4)
                with open(pathForStatistics / f"game{i}/statistics.json", "w") as f:
                    f.write(jsonStatistics)
                # Store heatmaps and shot track
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team1MotionHeatmap.png"), statisticsDict["Team1"]["MotionHeatmap"])
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team2MotionHeatmap.png"), statisticsDict["Team2"]["MotionHeatmap"])
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team1ShotHeatmap.png"), statisticsDict["Team1"]["ShotHeatmap"])
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team2ShotHeatmap.png"), statisticsDict["Team2"]["ShotHeatmap"])
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team1ShotTrack.png"), statisticsDict["Team1"]["ShotTrack"])
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team2ShotTrack.png"), statisticsDict["Team2"]["ShotTrack"])

                break
    
    
    print("Statistics stored!")




def loadStatistics(folder):
    print("Loading statistics...")