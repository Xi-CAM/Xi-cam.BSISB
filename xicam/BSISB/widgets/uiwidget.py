import os
from qtpy.QtCore import *
from qtpy.QtWidgets import *


#message box
class MsgBox(QMessageBox):
    def __init__(self, msg='None', type='info'):
        super(MsgBox, self).__init__()
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

class YesNoDialog(QMessageBox):
    def __init__(self, msg='None'):
        super(YesNoDialog, self).__init__()
        self.setIcon(QMessageBox.Question)
        self.setWindowTitle("Question")
        self.setText(msg)
        self.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        self.setDefaultButton(QMessageBox.Yes)

    def choice(self):
        return self.exec_()

def uiGetFile(caption='', dir='', filter='', options=QFileDialog.Options()):
    """
    :param caption: dialog's caption
    :param dir: The file dialog's working directory will be set to dir
    :param filter: Only files that match the given filter are shown, eg. *.txt
    :param options: options about how to run the dialog, see the QFileDialog.Option enum for more information
    :return: path, fileName, canceled
    """
    # flag:whether user clicked cancel button
    canceled = False
    filePath, _ = QFileDialog.getOpenFileName(parent=None, caption=caption, directory=dir, filter=filter, options=options)
    if filePath == '':
        path = ''
        fileName = ''
        canceled = True
    else:
        path = os.path.dirname(filePath)
        path += '/'
        fileName = os.path.basename(filePath)
    return path, fileName, canceled

def uiSaveFile(caption='', dir='', filter='', options=QFileDialog.Options()):
    """
    :param caption: dialog's caption
    :param dir: The file dialog's working directory will be set to dir
    :param filter: Only files that match the given filter are shown, eg. *.txt
    :param options: options about how to run the dialog, see the QFileDialog.Option enum for more information
    :return: path, fileName, canceled
    """
    # flag:whether user clicked cancel button
    canceled = False
    filePath, _ = QFileDialog.getSaveFileName(parent=None, caption=caption, directory=dir, filter=filter, options=options)
    if filePath == '':
        path = ''
        fileName = ''
        canceled = True
    else:
        path = os.path.dirname(filePath)
        path += '/'
        fileName = os.path.basename(filePath)
    return path, fileName, canceled

def uiGetDir(caption='', dir='', options=QFileDialog.ShowDirsOnly):
    """
    :param caption: dialog's caption
    :param dir: The file dialog's working directory will be set to dir
    :param options: options about how to run the dialog, see the QFileDialog.Option enum for more information
    :return: path, fileName, canceled
    """
    canceled = False
    path = QFileDialog.getExistingDirectory(parent=None, caption=caption, directory=dir, options=options)
    if path == '':
        canceled = True
    else:
        path += '/'
    return path, canceled
