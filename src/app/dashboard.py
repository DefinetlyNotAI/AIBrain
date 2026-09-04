"""Small, reusable dashboard primitives shared by AIBrain Qt surfaces."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


def status_badge(text: str, tone: str = "neutral") -> QLabel:
    badge = QLabel(text.upper())
    badge.setObjectName(f"statusBadge_{tone}")
    badge.setToolTip(text)
    return badge


def metric_card(title: str, value: str = "—", detail: str = "") -> tuple[QFrame, QLabel, QLabel]:
    card = QFrame()
    card.setObjectName("metricCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    label = QLabel(title)
    label.setObjectName("cardLabel")
    value_label = QLabel(value)
    value_label.setObjectName("cardValue")
    detail_label = QLabel(detail)
    detail_label.setObjectName("muted")
    detail_label.setWordWrap(True)
    layout.addWidget(label)
    layout.addWidget(value_label)
    layout.addWidget(detail_label)
    return card, value_label, detail_label


def metric_grid(parent: QWidget) -> QGridLayout:
    grid = QGridLayout(parent)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(10)
    return grid


def progress_card(title: str) -> tuple[QFrame, QProgressBar, QLabel]:
    card = QFrame()
    card.setObjectName("metricCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    label = QLabel(title)
    label.setObjectName("cardLabel")
    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setTextVisible(True)
    detail = QLabel()
    detail.setObjectName("muted")
    detail.setWordWrap(True)
    layout.addWidget(label)
    layout.addWidget(progress)
    layout.addWidget(detail)
    return card, progress, detail
