# main.py
import sys
from PyQt5.QtWidgets import QApplication
from pet_window import DesktopPet
from config import CONFIG

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    pet = DesktopPet()
    pet.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()