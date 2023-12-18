import json
import os
import cv2
from pathlib import Path
import numpy as np

TOPVIEW_POINTS = "database/topview/topview_coords.json"


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
    

    # in case that the scene is new
    if scenePoints is not None:
        scenePointsDict = {
            "courtSide": statisticsDict["courtSide"],
            "0": {
                # corner left side of basket
                "x": scenePoints[0][0],
                "y": scenePoints[0][1]
            },
            "1": {
                "x": scenePoints[1][0],
                "y": scenePoints[1][1]
            },
            "2": {
                "x": scenePoints[2][0],
                "y": scenePoints[2][1]
            },
            "3": {
                # corner right side of basket
                "x": scenePoints[3][0],
                "y": scenePoints[3][1]
            },
            "rimX": {
                "x": statisticsDict["rimPoints"][0][0],
                "y": statisticsDict["rimPoints"][0][1]
            },
            "rimY": {
                "x": statisticsDict["rimPoints"][1][0],
                "y": statisticsDict["rimPoints"][1][1]
            }
        }

        # get number of files in pathForScenes
        sceneNumber = len(os.listdir(pathForScenes))
        sceneNumber = int(sceneNumber)

        if sceneNumber == 0:
            os.mkdir(pathForScenes / "scene0")
            jsonScenePoints = json.dumps(scenePointsDict, indent=4)
            with open(pathForScenes / "scene0/scenePoints.json", "w") as f:
                f.write(jsonScenePoints)

            cv2.imwrite(str(pathForScenes / "scene0/firstFrame.png"), firstFrame)
            

        else:
            for i in range(1, sceneNumber+1, 1):
                if not os.path.exists(pathForScenes / f"scene{i}"): # if scene{i} does not exist, create it and store first frame and scenePoints
                    os.mkdir(pathForScenes / f"scene{i}")

                    jsonScenePoints = json.dumps(scenePointsDict, indent=4)
                    with open(pathForScenes / f"scene{i}/scenePoints.json", "w") as f:
                        f.write(jsonScenePoints)

                    cv2.imwrite(str(pathForScenes / f"scene{i}/firstFrame.png"), firstFrame)
                    break
            
    # check if there is attribute "Team2" in statisticsDict
    if "Team2" in statisticsDict:
        # statisticsDict to json
        numericalStatistics = {
            "Team1" : {
                "FGA" : statisticsDict["Team1"]["FGA"],
                "3PA" : statisticsDict["Team1"]["3PA"],
                "FGM" : statisticsDict["Team1"]["FGM"],
                "3PM" : statisticsDict["Team1"]["3PM"],
            },
            "Team2" : {
                "FGA" : statisticsDict["Team2"]["FGA"],
                "3PA" : statisticsDict["Team2"]["3PA"],
                "FGM" : statisticsDict["Team2"]["FGM"],
                "3PM" : statisticsDict["Team2"]["3PM"],
            }
        }
    
    else:
        numericalStatistics = {
            "Team1" : {
                "FGA" : statisticsDict["Team1"]["FGA"],
                "3PA" : statisticsDict["Team1"]["3PA"],
                "FGM" : statisticsDict["Team1"]["FGM"],
                "3PM" : statisticsDict["Team1"]["3PM"],
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
        cv2.imwrite(str(pathForStatistics / "game0/Team1ShotHeatmap.png"), statisticsDict["Team1"]["ShotHeatmap"])
        cv2.imwrite(str(pathForStatistics / "game0/Team1ShotTrack.png"), statisticsDict["Team1"]["ShotTrack"])
        
        if "Team2" in statisticsDict:
            cv2.imwrite(str(pathForStatistics / "game0/Team2MotionHeatmap.png"), statisticsDict["Team2"]["MotionHeatmap"])
            cv2.imwrite(str(pathForStatistics / "game0/Team2ShotHeatmap.png"), statisticsDict["Team2"]["ShotHeatmap"])
            cv2.imwrite(str(pathForStatistics / "game0/Team2ShotTrack.png"), statisticsDict["Team2"]["ShotTrack"])

    else:
        for i in range(1, gameNumber + 1, 1):
            if not os.path.exists(pathForStatistics / f"game{i}"):
                os.mkdir(pathForStatistics / f"game{i}")
                jsonStatistics = json.dumps(numericalStatistics, indent=4)
                with open(pathForStatistics / f"game{i}/statistics.json", "w") as f:
                    f.write(jsonStatistics)
                # Store heatmaps and shot track
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team1MotionHeatmap.png"), statisticsDict["Team1"]["MotionHeatmap"])
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team1ShotHeatmap.png"), statisticsDict["Team1"]["ShotHeatmap"])
                cv2.imwrite(str(pathForStatistics / f"game{i}/Team1ShotTrack.png"), statisticsDict["Team1"]["ShotTrack"])
                
                if "Team2" in statisticsDict:
                    cv2.imwrite(str(pathForStatistics / f"game{i}/Team2MotionHeatmap.png"), statisticsDict["Team2"]["MotionHeatmap"])
                    cv2.imwrite(str(pathForStatistics / f"game{i}/Team2ShotHeatmap.png"), statisticsDict["Team2"]["ShotHeatmap"])
                    cv2.imwrite(str(pathForStatistics / f"game{i}/Team2ShotTrack.png"), statisticsDict["Team2"]["ShotTrack"])

                break
    
    
    print("Statistics stored!")


def readTopviewPoints(side):
    topviewPoints = []
    with open(TOPVIEW_POINTS, "r") as f:
        topviewPointsJson = json.load(f)
        if side == "left":
            topviewPointsSide = topviewPointsJson["left"]
            topviewPoints.append((topviewPointsSide["topLeft"]["x"], topviewPointsSide["topLeft"]["y"]))
            topviewPoints.append((topviewPointsSide["topRight"]["x"], topviewPointsSide["topRight"]["y"]))
            topviewPoints.append((topviewPointsSide["bottomRight"]["x"], topviewPointsSide["bottomRight"]["y"]))
            topviewPoints.append((topviewPointsSide["bottomLeft"]["x"], topviewPointsSide["bottomLeft"]["y"]))
        else:
            topviewPointsSide = topviewPointsJson["right"]
            topviewPoints.append((topviewPointsSide["bottomRight"]["x"], topviewPointsSide["bottomRight"]["y"]))
            topviewPoints.append((topviewPointsSide["bottomLeft"]["x"], topviewPointsSide["bottomLeft"]["y"]))
            topviewPoints.append((topviewPointsSide["topLeft"]["x"], topviewPointsSide["topLeft"]["y"]))
            topviewPoints.append((topviewPointsSide["topRight"]["x"], topviewPointsSide["topRight"]["y"]))

    return  np.array(topviewPoints, np.int32)


def readScenePoints(scenePath):
    scenePoints = []
    rimPoints = []
    path = Path(scenePath) / "scenePoints.json"
    side = None
    with open(path, "r") as f:
        scenePointsJson = json.load(f)

        scenePoints.append((scenePointsJson["0"]["x"], scenePointsJson["0"]["y"]))
        scenePoints.append((scenePointsJson["1"]["x"], scenePointsJson["1"]["y"]))
        scenePoints.append((scenePointsJson["2"]["x"], scenePointsJson["2"]["y"]))
        scenePoints.append((scenePointsJson["3"]["x"], scenePointsJson["3"]["y"]))

        rimPoints.append((scenePointsJson["rimX"]["x"], scenePointsJson["rimX"]["y"]))
        rimPoints.append((scenePointsJson["rimY"]["x"], scenePointsJson["rimY"]["y"]))

    return np.array(scenePoints, np.int16), np.array(rimPoints, np.int16), scenePointsJson["courtSide"]



def loadStatistics(folder):
    print("Loading statistics...")