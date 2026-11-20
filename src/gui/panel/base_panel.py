"""모든 패널의 공통 베이스: 제목(접기 버튼) + content_layout + 그룹박스/버튼 생성 헬퍼."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton


class BasePanel(QWidget):
    """서브클래스는 setup_content()에서 self.content_layout에 위젯을 채운다."""

    def __init__(self, title="Panel", collapsible=True):
        super().__init__()
        self.title = title
        self.collapsible = collapsible
        self.is_collapsed = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        if self.collapsible:
            self.header_button = QPushButton(f"▼ {self.title}")
            self.header_button.setCheckable(True)
            self.header_button.setChecked(True)
            self.header_button.clicked.connect(self.toggle_collapse)
            self.header_button.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    border: 1px solid #555;
                    background-color: #404040;
                    color: white;
                    padding: 8px;
                    font-weight: bold;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QPushButton:checked {
                    background-color: #606060;
                }
            """)
            layout.addWidget(self.header_button)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.setup_content()
        layout.addWidget(self.content_widget)

    def setup_content(self):
        pass

    def toggle_collapse(self):
        """content 영역 접기/펼치기."""
        if not self.collapsible:
            return
        self.is_collapsed = not self.is_collapsed
        self.header_button.setText(f"{'▶' if self.is_collapsed else '▼'} {self.title}")
        self.content_widget.setMaximumHeight(0 if self.is_collapsed else 16777215)
        self.content_widget.setVisible(not self.is_collapsed)

    def emit_status(self, message):
        """상태 메시지를 콘솔에 출력한다."""
        print(f"[{self.__class__.__name__}] {message}")

    def create_group_box(self, title, layout_type="vertical"):
        """Returns: (QGroupBox, 그 안의 QVBoxLayout 또는 QHBoxLayout)"""
        group_box = QGroupBox(title)
        layout = QVBoxLayout(group_box) if layout_type == "vertical" else QHBoxLayout(group_box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        return group_box, layout

    def create_button_horizontal(self, buttons_config):
        """buttons_config: [{"text", "callback"?, "style"?, "height"?}, ...]

        Returns: (QHBoxLayout, [QPushButton, ...])
        """
        layout = QHBoxLayout()
        layout.setSpacing(5)
        buttons = []
        for config in buttons_config:
            button = QPushButton(config["text"])
            if "callback" in config:
                button.clicked.connect(config["callback"])
            if "style" in config:
                button.setStyleSheet(config["style"])
            if "height" in config:
                button.setMinimumHeight(config["height"])
            layout.addWidget(button)
            buttons.append(button)
        return layout, buttons
