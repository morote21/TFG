from domain_layer.player_recognition.team_association import Teams
import cv2


class DomainController:

    def __init__(self, video, top_view, team1_img, team2_img, n_players):
        self.video = video
        self.top_view = top_view
        self.teams = Teams(team1_img, team2_img, n_players)
        self.n_players = n_players

    def get_first_frame(self):
        video = cv2.VideoCapture(self.video)
        ret, frame = video.read()
        return ret, frame
