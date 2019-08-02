from qtpy.QtWidgets import QMessageBox

#message box
class msgbox(QMessageBox):
    def __init__(self, msg='None', type='info'):
        super(msgbox, self).__init__()
        if type == 'warn':
            self.setIcon(QMessageBox.Warning)
            self.setWindowTitle("Warning")
        elif type == 'error':
            self.setIcon(QMessageBox.Critical)
            self.setWindowTitle("Error")
        else:
            self.setIcon(QMessageBox.Information)
            self.setWindowTitle("Information")
        self.setText(msg)
        self.setStandardButtons(QMessageBox.Ok)
        self.exec_()