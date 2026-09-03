"""Persistent, user-editable colours for the AIBrain desktop interface."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..utils.logging import PROJECT_ROOT

COLOUR_FILE = PROJECT_ROOT / ".cache" / "aibrain.color.json"
DEFAULT_COLOURS = {
    "background": "#09131c",
    "panel": "#0d202b",
    "panel_border": "#1d3b49",
    "text": "#dceaf1",
    "title": "#f2f8fb",
    "muted": "#88a0ae",
    "accent": "#15536b",
    "accent_hover": "#1b6d8b",
    "accent_text": "#ffffff",
    "disabled": "#26353b",
    "disabled_text": "#82939a",
    "status_ready_background": "#164831",
    "status_ready_text": "#8df3ba",
    "status_caution_background": "#4a3b19",
    "status_caution_text": "#ffe08b",
    "user_bubble": "#143e50",
    "assistant_bubble": "#10212d",
    "world_bubble": "#28203b",
    "world_border": "#5c4b86",
    "renderer_background": "#071018",
    "overlay": "#0c1e2a",
}


def _valid_colour(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    colour = QColor(value)
    return colour.name(QColor.NameFormat.HexRgb) if colour.isValid() else None


def load_colours() -> dict[str, str]:
    """Load valid custom colours while retaining safe defaults for omissions."""
    colours = DEFAULT_COLOURS.copy()
    try:
        payload = json.loads(COLOUR_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return colours
    if not isinstance(payload, dict):
        return colours
    for name in colours:
        value = _valid_colour(payload.get(name))
        if value is not None:
            colours[name] = value
    return colours


def save_colours(colours: dict[str, str]) -> None:
    """Atomically persist the supported palette in the requested cache file."""
    COLOUR_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: _valid_colour(colours.get(name)) or default
        for name, default in DEFAULT_COLOURS.items()
    }
    temporary = COLOUR_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(COLOUR_FILE)


def stylesheet(colours: dict[str, str]) -> str:
    """Render the complete semantic application stylesheet from the palette."""
    return f"""
QMainWindow {{ background: {colours['background']}; color: {colours['text']}; }}
QWidget {{ font: 10pt 'Segoe UI'; color: {colours['text']}; }}
QLabel#title {{ font-size: 22px; font-weight: 650; color: {colours['title']}; }}
QLabel#muted {{ color: {colours['muted']}; }}
QLabel#mode {{ background: {colours['accent']}; color: {colours['accent_text']}; border-radius: 9px; padding: 4px 8px; font-weight: 700; }}
QFrame#runtimeCard, QFrame#actionCard, QFrame#composeCard, QGroupBox {{ background: {colours['panel']}; border: 1px solid {colours['panel_border']}; border-radius: 9px; }}
QFrame#metricCard {{ background: {colours['panel']}; border: 1px solid {colours['panel_border']}; border-radius: 9px; }}
QFrame#runtimeCard, QFrame#actionCard {{ padding: 4px; }}
QLabel#cardLabel {{ color: {colours['muted']}; font-size: 9px; font-weight: 700; }}
QLabel#cardValue {{ color: {colours['title']}; font-size: 18px; font-weight: 700; }}
QLabel#statusBadge_neutral, QLabel#statusBadge_ready, QLabel#statusBadge_caution {{ border-radius: 8px; padding: 4px 8px; font-weight: 700; }}
QLabel#statusBadge_neutral {{ background: {colours['accent']}; color: {colours['accent_text']}; }}
QLabel#statusBadge_ready {{ background: {colours['status_ready_background']}; color: {colours['status_ready_text']}; }}
QLabel#statusBadge_caution {{ background: {colours['status_caution_background']}; color: {colours['status_caution_text']}; }}
QProgressBar {{ background: {colours['background']}; border: 1px solid {colours['panel_border']}; border-radius: 5px; text-align: center; min-height: 16px; }}
QProgressBar::chunk {{ background: {colours['accent_hover']}; border-radius: 4px; }}
QGroupBox {{ margin-top: 10px; padding: 10px 8px 6px 8px; font-weight: 650; color: {colours['title']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QLabel#overlay {{ background: {colours['overlay']}; border: 1px solid {colours['panel_border']}; border-radius: 8px; padding: 8px; color: {colours['text']}; }}
QLabel#userBubble, QLabel#assistantBubble {{ border-radius: 10px; padding: 10px; margin: 3px 0; }}
QLabel#userBubble {{ background: {colours['user_bubble']}; }}
QLabel#assistantBubble {{ background: {colours['assistant_bubble']}; border: 1px solid {colours['panel_border']}; }}
QLabel#worldBubble {{ background: {colours['world_bubble']}; border: 1px solid {colours['world_border']}; border-radius: 10px; padding: 10px; margin: 3px 0; }}
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTreeWidget {{ background: {colours['panel']}; border: 1px solid {colours['panel_border']}; border-radius: 6px; padding: 6px; color: {colours['text']}; alternate-background-color: {colours['background']}; }}
QHeaderView::section {{ background: {colours['accent']}; color: {colours['accent_text']}; border: 0; border-right: 1px solid {colours['panel_border']}; padding: 7px 9px; font-weight: 650; }}
QTreeWidget::item {{ min-height: 28px; padding: 4px; }}
QTreeWidget::item:selected {{ background: {colours['accent_hover']}; color: {colours['accent_text']}; }}
QTabWidget::pane {{ background: {colours['background']}; border: 1px solid {colours['panel_border']}; border-radius: 6px; }}
QTabBar::tab {{ background: {colours['panel']}; color: {colours['muted']}; border: 1px solid {colours['panel_border']}; padding: 7px 12px; }}
QTabBar::tab:selected {{ background: {colours['accent']}; color: {colours['accent_text']}; }}
QPushButton, QToolButton {{ background: {colours['accent']}; border: none; border-radius: 6px; padding: 7px 12px; color: {colours['accent_text']}; font-weight: 600; }}
QPushButton:hover, QToolButton:hover {{ background: {colours['accent_hover']}; }}
QPushButton:disabled, QToolButton:disabled {{ background: {colours['disabled']}; color: {colours['disabled_text']}; }}
QScrollArea {{ background: transparent; }} QSplitter::handle {{ background: {colours['panel_border']}; width: 8px; }}
"""


class ColourSettingsDialog(QDialog):
    """A semantic colour picker with Qt's wheel, RGB, and hexadecimal editor."""

    def __init__(self, colours: dict[str, str], parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("AIBrain colour settings")
        self._colours = colours.copy()
        layout = QVBoxLayout(self)
        palette = QWidget()
        self._palette_grid = QGridLayout(palette)
        self._palette_grid.setHorizontalSpacing(12)
        self._palette_grid.setVerticalSpacing(8)
        self._palette_scroll = QScrollArea()
        self._palette_scroll.setWidgetResizable(True)
        self._palette_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._palette_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._palette_scroll.setWidget(palette)
        layout.addWidget(self._palette_scroll, 1)
        self._buttons: dict[str, QPushButton] = {}
        for index, (name, value) in enumerate(self._colours.items()):
            item = QWidget()
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(name.replace("_", " ").title())
            button = QPushButton(value.upper())
            button.setMinimumWidth(112)
            button.clicked.connect(lambda checked=False, key=name: self._choose(key))
            self._buttons[name] = button
            item_layout.addWidget(label, 1)
            item_layout.addWidget(button)
            self._palette_grid.addWidget(item, index // 2, index % 2)
            self._refresh_button(name)
        controls = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        controls.accepted.connect(self.accept)
        controls.rejected.connect(self.reject)
        layout.addWidget(controls)
        self.resize(720, 500)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Fit and centre the palette inside the active display."""
        super().showEvent(event)
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.resize(
            min(720, max(360, available.width() - 40)),
            min(500, max(280, available.height() - 40)),
        )
        self.move(available.center() - self.rect().center())

    @property
    def colours(self) -> dict[str, str]:
        return self._colours.copy()

    def _choose(self, name: str) -> None:
        dialog = QColorDialog(QColor(self._colours[name]), self)
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dialog.setWindowTitle(f"Select {name.replace('_', ' ')}")
        if dialog.exec():
            self._colours[name] = dialog.currentColor().name(QColor.NameFormat.HexRgb)
            self._refresh_button(name)

    def _refresh_button(self, name: str) -> None:
        value = self._colours[name]
        self._buttons[name].setText(value.upper())
        self._buttons[name].setStyleSheet(
            f"background: {value}; color: {'#101820' if QColor(value).lightness() > 145 else '#f7fbff'};"
        )
