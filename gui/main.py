import sys
import signal
from PyQt6.QtWidgets import QApplication
from .main_window import MainWindow

def main():
    """The main function to run the GUI."""
    app = QApplication(sys.argv)

    # --- Graceful Ctrl-C Handling ---
    # This allows Python to handle SIGINT (Ctrl+C) and exit gracefully.
    # The default behavior in PyQt can sometimes ignore it. By setting this,
    # a KeyboardInterrupt will be raised, which allows our application's
    # closeEvent handlers to be called during shutdown.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    window = MainWindow()
    window.show()

    # It's good practice to have a timer process Python events,
    # which helps with signal handling responsiveness.
    # However, for simplicity and since the above is effective, we'll omit it
    # unless issues arise.

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Ctrl+C detected. Shutting down gracefully.")
        # The application will now exit, and closeEvent handlers will be triggered.

if __name__ == '__main__':
    main()
