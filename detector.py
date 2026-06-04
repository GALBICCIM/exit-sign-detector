import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from ultralytics import YOLO


class SafetySignDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("안내표지판 검출기 (macOS)")
        self.setGeometry(100, 100, 1000, 600)

        self.image_path = None
        self.cv_image = None
        self.annotated_frame = None  # 추론 결과 이미지를 저장할 변수 추가
        self.model = None

        self.init_model()
        self.init_ui()

    def init_model(self):
        model_path = "best.pt"
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(
                f"모델 로드 실패. '{model_path}' 파일이 존재하는지 확인하십시오. 에러: {e}"
            )

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # 이미지 디스플레이 영역
        image_layout = QHBoxLayout()

        self.lbl_original = QLabel("원본 이미지를 불러오십시오.")
        self.lbl_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_original.setStyleSheet(
            "border: 1px solid #777; background-color: #eee;"
        )
        # 라벨이 이미지 크기에 의해 강제 확장되는 것을 방지
        self.lbl_original.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )

        self.lbl_result = QLabel("검출 결과가 여기에 표시됩니다.")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setStyleSheet("border: 1px solid #777; background-color: #eee;")
        # 라벨이 이미지 크기에 의해 강제 확장되는 것을 방지
        self.lbl_result.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )

        image_layout.addWidget(self.lbl_original)
        image_layout.addWidget(self.lbl_result)
        layout.addLayout(image_layout, stretch=1)

        # 컨트롤 영역
        btn_layout = QHBoxLayout()

        self.btn_load = QPushButton("이미지 불러오기")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setMinimumHeight(40)

        self.btn_detect = QPushButton("안내표지판 검출")
        self.btn_detect.clicked.connect(self.detect_signs)
        self.btn_detect.setMinimumHeight(40)
        self.btn_detect.setEnabled(False)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_detect)
        layout.addLayout(btn_layout)

        main_widget.setLayout(layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            self.image_path = file_name
            path_array = np.fromfile(file_name, np.uint8)
            self.cv_image = cv2.imdecode(path_array, cv2.IMREAD_COLOR)
            self.annotated_frame = None  # 새 이미지를 불러오면 이전 결과 초기화

            if self.cv_image is not None:
                self.display_image(self.cv_image, self.lbl_original)
                self.btn_detect.setEnabled(True)
                self.lbl_result.setText("검출 대기 중...")
                self.lbl_result.clear()
            else:
                QMessageBox.warning(self, "오류", "이미지 디코딩에 실패했습니다.")

    def detect_signs(self):
        if self.model is None:
            QMessageBox.critical(self, "오류", "'best.pt' 모델 파일이 없습니다.")
            return

        if self.cv_image is None:
            return

        results = self.model.predict(source=self.cv_image, conf=0.25, save=False)

        # 추론 결과를 인스턴스 변수에 저장하여 resizeEvent에서 재사용 가능하게 함
        self.annotated_frame = results[0].plot()

        self.display_image(self.annotated_frame, self.lbl_result)

    def display_image(self, img, label):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        q_img = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)

        # width(), height()가 0 이하일 때 발생할 수 있는 오류 방지
        if label.width() > 0 and label.height() > 0:
            scaled_pixmap = pixmap.scaled(
                label.width(),
                label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        # 창 크기 변경 시 원본 및 결과 이미지 모두 레이아웃 크기에 맞춰 재조정
        if self.cv_image is not None:
            self.display_image(self.cv_image, self.lbl_original)
        if self.annotated_frame is not None:
            self.display_image(self.annotated_frame, self.lbl_result)

        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SafetySignDetector()
    window.show()
    sys.exit(app.exec())
