"""
Centralized stylesheet for the Scientific Discovery Engine GUI.

This module defines color palettes, fonts, and QSS stylesheets for consistent
widget styling across the application.
"""

# --- Color Palette ---
# Chosen for a modern, clean, and professional look.
# Based on a cool, blue-centric theme.
PRIMARY_COLOR = "#007BFF"  # A strong, friendly blue
PRIMARY_COLOR_LIGHT = "#409CFF"
PRIMARY_COLOR_DARK = "#0056B3"
SECONDARY_COLOR = "#6C757D"  # Neutral gray for secondary info
SUCCESS_COLOR = "#28A745"   # Green for success, completion
DANGER_COLOR = "#DC3545"    # Red for errors, cancellation
WARNING_COLOR = "#FFC107"   # Yellow for warnings
INFO_COLOR = "#17A2B8"      # Teal for informational content

BACKGROUND_COLOR_LIGHT = "#F8F9FA" # Very light gray for backgrounds
BACKGROUND_COLOR_DARK = "#E9ECEF"  # Slightly darker gray
TEXT_COLOR_LIGHT = "#F8F9FA"
TEXT_COLOR_DARK = "#212529"      # High-contrast dark gray for text
BORDER_COLOR = "#DEE2E6"         # Light gray for borders

# --- Typography ---
FONT_FAMILY = "Segoe UI, Arial, sans-serif"
FONT_SIZE_NORMAL = "14px"
FONT_SIZE_LARGE = "16px"
FONT_SIZE_XLARGE = "20px"
FONT_SIZE_TITLE = "24px"

# --- Main Stylesheet ---
def get_main_stylesheet():
    """
    Returns the main QSS stylesheet for the entire application.
    """
    return f"""
        /* --- General Window --- */
        QMainWindow, QWidget {{
            background-color: {BACKGROUND_COLOR_LIGHT};
            font-family: {FONT_FAMILY};
            color: {TEXT_COLOR_DARK};
        }}

        /* --- GroupBox --- */
        QGroupBox {{
            font-size: {FONT_SIZE_LARGE};
            font-weight: bold;
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            margin-top: 1em;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 10px;
            background-color: {BACKGROUND_COLOR_LIGHT};
        }}

        /* --- Labels --- */
        QLabel {{
            font-size: {FONT_SIZE_NORMAL};
        }}
        QLabel#titleLabel {{
            font-size: {FONT_SIZE_TITLE};
            font-weight: bold;
            color: {PRIMARY_COLOR_DARK};
            padding-bottom: 10px;
        }}
        QLabel#headerLabel {{
            font-size: {FONT_SIZE_XLARGE};
            font-weight: bold;
            color: {TEXT_COLOR_DARK};
            padding-bottom: 5px;
        }}
        QLabel#descriptionLabel {{
            font-size: {FONT_SIZE_NORMAL};
            color: {SECONDARY_COLOR};
            padding-bottom: 15px;
        }}


        /* --- Buttons --- */
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR_LIGHT};
            font-weight: bold;
            font-size: {FONT_SIZE_NORMAL};
            padding: 10px 15px;
            border-radius: 5px;
            border: none;
        }}
        QPushButton:hover {{
            background-color: {PRIMARY_COLOR_LIGHT};
        }}
        QPushButton:pressed {{
            background-color: {PRIMARY_COLOR_DARK};
        }}
        QPushButton:disabled {{
            background-color: {SECONDARY_COLOR};
            color: #CCCCCC;
        }}
        QPushButton#cancelButton {{
            background-color: {WARNING_COLOR};
        }}
        QPushButton#cancelButton:hover {{
            background-color: #FFD34E;
        }}
        QPushButton#cancelButton:pressed {{
            background-color: #E0A800;
        }}
        QPushButton#finishButton {{
            background-color: {SUCCESS_COLOR};
        }}
         QPushButton#finishButton:hover {{
            background-color: #2EBF4E;
        }}
        QPushButton#finishButton:pressed {{
            background-color: #218838;
        }}


        /* --- Input Fields --- */
        QComboBox, QSpinBox, QLineEdit, QTextEdit {{
            font-size: {FONT_SIZE_NORMAL};
            padding: 8px;
            border: 1px solid {BORDER_COLOR};
            border-radius: 5px;
            background-color: #FFFFFF;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid {BORDER_COLOR};
        }}
        QTextEdit, QLineEdit {{
            selection-background-color: {PRIMARY_COLOR};
            selection-color: {TEXT_COLOR_LIGHT};
        }}

        /* --- Progress Bar --- */
        QProgressBar {{
            border: 1px solid {BORDER_COLOR};
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            color: {TEXT_COLOR_DARK};
        }}
        QProgressBar::chunk {{
            background-color: {SUCCESS_COLOR};
            border-radius: 4px;
        }}

        /* --- Tab Widget --- */
        QTabWidget::pane {{
            border: 1px solid {BORDER_COLOR};
            border-top: none;
            border-radius: 0 0 5px 5px;
        }}
        QTabBar::tab {{
            background: {BACKGROUND_COLOR_DARK};
            border: 1px solid {BORDER_COLOR};
            border-bottom: none;
            padding: 8px 15px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            font-weight: bold;
        }}
        QTabBar::tab:selected {{
            background: {BACKGROUND_COLOR_LIGHT};
            border-bottom: 1px solid {BACKGROUND_COLOR_LIGHT};
        }}
        QTabBar::tab:!selected:hover {{
            background: #DCDFE2;
        }}

        /* --- Table Widget --- */
        QTableWidget {{
            gridline-color: {BORDER_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 5px;
        }}
        QHeaderView::section {{
            background-color: {BACKGROUND_COLOR_DARK};
            padding: 5px;
            border: 1px solid {BORDER_COLOR};
            font-weight: bold;
        }}
    """
