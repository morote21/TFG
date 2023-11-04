import os
import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class TopviewWindow(QWidget):
    def __init__(self, path):
        super().__init__()

        self.image_label = QLabel(self)
        self.pixmap = QPixmap(path)
        self.image_label.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(),
                    self.pixmap.height())

        self.image_label.mousePressEvent = self.get_coords_init
        self.image_label.mouseReleaseEvent = self.get_coords_end

        self.show()

    def get_coords_init(self, event):
        x = event.pos().x()
        y = event.pos().y()
        print(x, y)

    def get_coords_end(self, event):
        x = event.pos().x()
        y = event.pos().y()
        print(x, y)


class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.topview_image = None
        self.video = None
        self.tw_window = None

        self.setWindowTitle("Statistics Generator")

        layout = QVBoxLayout()

        button = QPushButton("Upload Image")
        button.setCheckable(False)
        button.clicked.connect(self.load_topview)

        layout.addWidget(QLabel("Insert top-view image:"))
        layout.addWidget(button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def the_button_was_clicked(self):
        print("The button was clicked")

    def load_topview(self):
        options = QFileDialog.Options()
        topview_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", options=options)

        if topview_path is not None:
            self.tw_window = TopviewWindow(topview_path)
            self.tw_window.show()


def main():
    app = QApplication([])  # Initialize app
    main_window = MainApp()  # Main window stablished
    main_window.show()
    app.exec_()


if __name__ == '__main__':
    main()
