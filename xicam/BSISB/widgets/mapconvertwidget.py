import os
import sys
import numpy as np
from glob import glob
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from xicam.BSISB.widgets.uiwidget import MsgBox, YesNoDialog, uiGetFile, uiGetDir, uiSaveFile
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from lbl_ir.data_objects import ir_map
from lbl_ir.io_tools.read_omnic import read_and_convert, read_npy

class mapToH5(QSplitter):
    def __init__(self):
        super(mapToH5, self).__init__()

        self.setOrientation(Qt.Vertical)
        self.imageview = MapViewWidget()
        self.spectra = SpectraPlotWidget()

        self.imageview_and_toolbar = QSplitter()
        self.imageview_and_toolbar.setOrientation(Qt.Horizontal)
        self.toolbar_and_text = QSplitter()
        self.toolbar_and_text.setOrientation(Qt.Vertical)
        # define tool bar
        self.toolBar = QWidget()
        self.toollayout = QGridLayout()
        self.toolBar.setLayout(self.toollayout)
        # define infobox
        self.info = QWidget()
        self.info.setLayout(QVBoxLayout())
        self.infoBox = QTextEdit()
        self.info.layout().addWidget(QLabel('Status Info:'))
        self.info.layout().addWidget(self.infoBox)
        # add tool bar buttons
        self.openMapBtn = QToolButton()
        self.openMapBtn.setText('Open Map')
        self.openNpyBtn = QToolButton()
        self.openNpyBtn.setText('Open Npy')
        self.saveBtn = QToolButton()
        self.saveBtn.setText('Save HDF5')
        self.batchBtn = QToolButton()
        self.batchBtn.setText('Batch Process')
        # define sample name input and checkbox
        self.sampleName = QLineEdit()
        self.sampleName.setText('None')
        self.T2AConvert = QCheckBox()
        self.T2AConvert.setText('Auto T->A')
        self.T2AConvert.setChecked(True)
        # Assemble widgets
        self.toollayout.addWidget(self.openMapBtn)
        self.toollayout.addWidget(self.openNpyBtn)
        self.toollayout.addWidget(self.saveBtn)
        self.toollayout.addWidget(self.batchBtn)
        self.toollayout.addWidget(QLabel('Sample Name:'))
        self.toollayout.addWidget(self.sampleName)
        self.toollayout.addWidget(self.T2AConvert)
        self.toollayout.setAlignment(Qt.AlignVCenter)
        # Assemble widgets
        self.toolbar_and_text.addWidget(self.toolBar)
        self.toolbar_and_text.addWidget(self.info)
        self.imageview_and_toolbar.addWidget(self.toolbar_and_text)
        self.imageview_and_toolbar.addWidget(self.imageview)
        self.imageview_and_toolbar.setSizes([1, 1000])  # adjust initial splitter size
        self.addWidget(self.imageview_and_toolbar)
        self.addWidget(self.spectra)
        self.setSizes([1000, 1000])

        # Connect signals
        self.imageview.sigShowSpectra.connect(self.spectra.showSpectra)
        self.spectra.sigEnergyChanged.connect(self.imageview.setEnergy)
        self.openMapBtn.clicked.connect(self.openBtnClicked)
        self.openNpyBtn.clicked.connect(self.openNpy)
        self.saveBtn.clicked.connect(self.saveBtnClicked)
        self.batchBtn.clicked.connect(self.batchBtnClicked)
        # Constants
        self.path = os.path.dirname(sys.path[1])
        self.minYLimit = 5
        self.epsilon = 1e-10
        self.fileFormat = 'map'

    def openNpy(self):
        self.fileFormat = 'npy'
        self.T2AConvert.setChecked(True)
        self.openBtnClicked()

    def openBtnClicked(self):
        # open omnic map file
        if self.fileFormat == 'map':
            self.filePath, self.fileName, canceled = uiGetFile('Open map file', self.path, "Omnic Map Files (*.map)")
        elif self.fileFormat == 'npy':
            self.filePath, self.fileName, canceled = uiGetFile('Open npy file', self.path, "Numpy array Files (*.npy)")
        if canceled:
            self.infoBox.setText('Open file canceled.')
            return
        # set sample_id
        if self.sampleName.text() == 'None':
            sample_info = ir_map.sample_info(sample_id=self.fileName[:-4])
        else:
            sample_info = ir_map.sample_info(sample_id=self.sampleName.text())
        #try to open omnic map file
        try:
            if self.fileFormat == 'map':
                self.irMap = read_and_convert(self.filePath + self.fileName, sample_info=sample_info)
            elif self.fileFormat == 'npy':
                self.irMap = read_npy(self.filePath + self.fileName, sample_info=sample_info)
        except Exception as error:
            self.infoBox.setText(error.args[0] + f'\nFailed to open file: {self.fileName}.')
        else:
            if self.fileFormat == 'map':
                # check whether to perform T->A conversion
                spec0 = self.irMap.imageCube[0, 0, :]
                maxSpecY = np.max(spec0)
                if (not self.T2AConvert.isChecked()) and (maxSpecY >= self.minYLimit):
                    userMsg = YesNoDialog(f'max(Y) of the first spectrum is greater than {self.minYLimit}, \
                    while the "Auto T->A" box is not checked. \nPlease make sure data format is in absorbance.\
                    \nDo you want to perform "Auto T->A" conversion?')
                    userChoice = userMsg.choice()
                    if userChoice == QMessageBox.Yes:
                        self.T2AConvert.setChecked(True)
                        self.irMap.imageCube = -np.log10(self.irMap.imageCube / 100 + self.epsilon)
                        self.irMap.data = -np.log10(self.irMap.data / 100 + self.epsilon)
                        self.infoBox.setText(f'User chooses to perform T->A conversion in {self.fileName}.')
                    else:
                        self.infoBox.setText(f'User chooses not to perform T->A conversion in {self.fileName}.')
                elif maxSpecY >= self.minYLimit:
                    self.irMap.imageCube = -np.log10(self.irMap.imageCube / 100 + self.epsilon)
                    self.irMap.data = -np.log10(self.irMap.data / 100 + self.epsilon)
                    self.infoBox.setText(f'T->A conversion is performed in {self.fileName}.')
                else:
                    self.infoBox.setText(f"{self.fileName}'s datatype is absorbance. \nT->A conversion is not performed.")

            self.dataCube = np.moveaxis(np.flipud(self.irMap.imageCube), -1, 0)
            # set up required data/properties in self.imageview
            row, col = self.irMap.imageCube.shape[0], self.irMap.imageCube.shape[1]
            wavenumbers = self.irMap.wavenumbers
            rc2ind = {tuple(x[1:]): x[0] for x in self.irMap.ind_rc_map}
            self.updateImage(row, col, wavenumbers, rc2ind, self.dataCube)
            # set up required data/properties in self.spectra
            self.spectra.wavenumbers = self.imageview.wavenumbers
            self.spectra.rc2ind = self.imageview.rc2ind
            self.spectra._data = self.irMap.data

    def updateImage(self, row, col, wavenumbers, rc2ind, dataCube):
        self.imageview.row, self.imageview.col = row, col
        self.imageview.wavenumbers = wavenumbers
        self.imageview.rc2ind = rc2ind
        self.imageview._data = dataCube
        self.imageview._image = self.imageview._data[0]
        self.imageview.setImage(img=dataCube)

    def saveBtnClicked(self):
        if hasattr(self, 'irMap') and (self.filePath != ''):
            h5Name = self.fileName[:-4] + '.h5'
            try:
                self.irMap.write_as_hdf5(self.filePath + h5Name)
                MsgBox(f'Map to HDF5 conversion complete! \nFile Location: {self.filePath + h5Name}')
                self.infoBox.setText(f'HDF5 File Location: {self.filePath + h5Name}')
            except Exception as error:
                MsgBox(error.args[0], 'error')
                saveFilePath, saveFileName, canceled = uiSaveFile('Save HDF5 file', self.path, "HDF5 Files (*.h5)")
                if not canceled:
                    # make sure the saveFileName end with .h5
                    if not saveFileName.endswith('h5'):
                        saveFileName = saveFileName.split('.')[0] + '.h5'
                    # save file
                    try:
                        self.irMap.write_as_hdf5(saveFilePath + saveFileName)
                        MsgBox(f'Map to HDF5 conversion complete! \nFile Location: {saveFilePath + saveFileName}')
                        self.infoBox.setText(f'HDF5 File Location: {saveFilePath + saveFileName}')
                    except Exception:
                        MsgBox(error.args[0] + '\nSave HDF5 file failed.', 'error')
                        self.infoBox.setText(f'Save HDF5 file failed.')
                else:
                    self.infoBox.setText(f'Save file canceled. No HDF5 file was saved.')
        else:
            MsgBox(f"IR map object doesn't exist or file path is incorrect. \nPlease open an Omnic map file first.")

    def batchBtnClicked(self):
        folderPath, canceled = uiGetDir('Select a folder', self.path)
        if canceled:
            self.infoBox.setText('Open folder canceled.')
            return

        filePaths = glob(folderPath + '*.map')
        # if no map file was found
        if not filePaths:
            MsgBox('No .map file was found.\nPlease select another folder')
            self.infoBox.setText('No .map file was found in the selected folder.')
            return

        #try to use thread
        # mapConverter = BatchMapConverter(self.T2AConvert.isChecked(), self.sampleName.text(),\
        #                                  self.epsilon, self.minYLimit, filePaths)
        # mapConverter.sigText.connect(lambda x:self.infoBox.setText(x))
        # mapConverter.sigImage.connect(lambda x:self.updateImage(*x))
        # mapConverter.sigT2A.connect(lambda x:self.T2AConvert.setChecked(x))
        # mapConverter.start()

        # ToDo : change this Long loop to thread
        n_files = len(filePaths)
        for i, filePath in enumerate(filePaths):
            fileName = os.path.basename(filePath)
            # set sample_id
            if self.sampleName.text() == 'None':
                sample_info = ir_map.sample_info(sample_id=fileName[:-4])
            else:
                sample_info = ir_map.sample_info(sample_id=self.sampleName.text())
            # try open omnic map and show image
            try:
                irMap = read_and_convert(filePath, sample_info=sample_info)
            except Exception as error:
                self.infoBox.setText(error.args[0] + f'\nFailed to open file: {fileName}.')
                MsgBox(error.args[0] + f'\nFailed to open file: {fileName}.')
                break
            else:
                # check whether to perform T->A conversion
                spec0 = irMap.imageCube[0, 0, :]
                maxSpecY = np.max(spec0)
                if (not self.T2AConvert.isChecked()) and (maxSpecY >= self.minYLimit):
                    userMsg = YesNoDialog(f'max(Y) of the first spectrum is greater than {self.minYLimit}, \
                    while the "Auto T->A" box is not checked. \nPlease make sure data format is in absorbance.\
                    \nDo you want to perform "Auto T->A" conversion?')
                    # get user choice
                    userMsg.addButton(QMessageBox.YesToAll)
                    userChoice = userMsg.choice()
                    if userChoice == QMessageBox.YesToAll: # set 'auto T->A' on
                        self.T2AConvert.setChecked(True)
                    if (userChoice == QMessageBox.YesToAll) or (userChoice == QMessageBox.Yes):
                        irMap.imageCube = -np.log10(irMap.imageCube / 100 + self.epsilon)
                        irMap.data = -np.log10(irMap.data / 100 + self.epsilon)
                        self.infoBox.setText(f'User chooses to perform T->A conversion in {fileName}.')
                    else:
                        self.infoBox.setText(f'User chooses not to perform T->A conversion in {fileName}.')
                elif maxSpecY >= self.minYLimit:
                    irMap.imageCube = -np.log10(irMap.imageCube / 100 + self.epsilon)
                    irMap.data = -np.log10(irMap.data / 100 + self.epsilon)
                    self.infoBox.setText(f'T->A conversion is performed in {fileName}.')
                else:
                    self.infoBox.setText(
                        f"{fileName}'s datatype is absorbance. \nT->A conversion is not performed.")

                dataCube = np.moveaxis(np.flipud(irMap.imageCube), -1, 0)
                # set up required data/properties in self.imageview and show image
                row, col = irMap.imageCube.shape[0], irMap.imageCube.shape[1]
                wavenumbers = irMap.wavenumbers
                rc2ind = {tuple(x[1:]): x[0] for x in irMap.ind_rc_map}
                self.updateImage(row, col, wavenumbers, rc2ind, dataCube)

                # save hdf5
                h5Name = fileName[:-4] + '.h5'
                try:
                    irMap.write_as_hdf5(folderPath + h5Name)
                    self.infoBox.setText(f'#{i+1} out of {n_files} maps HDF5-conversion complete! \
                    \nHDF5 File Location: {folderPath + h5Name}')
                except Exception as error:
                    MsgBox(error.args[0], 'error')
                    break
            QApplication.processEvents()
        MsgBox('All file conversion complete!')

class BatchMapConverter(QThread):
    sigText = Signal(str)
    sigImage = Signal(object)
    sigT2A = Signal(bool)

    def __init__(self, T2AConvertStatus, sampleName, epsilon, minYLimit, filePaths):
        super(BatchMapConverter, self).__init__()

        self.epsilon = epsilon
        self.minYLimit = minYLimit
        self.T2AConvertStatus = T2AConvertStatus
        self.filePaths = filePaths
        self.sampleName = sampleName

    def __del__(self):
        self.wait()

    def run(self):

        n_files = len(self.filePaths)
        for i, filePath in enumerate(self.filePaths):
            folderPath = os.path.dirname(filePath) + '/'
            fileName = os.path.basename(filePath)
            # set sample_id
            if self.sampleName == 'None':
                sample_info = ir_map.sample_info(sample_id=fileName[:-4])
            else:
                sample_info = ir_map.sample_info(sample_id=self.sampleName)
            # try open omnic map and show image
            try:
                irMap = read_and_convert(filePath, sample_info=sample_info)
            except Exception as error:
                self.sigText.emit(error.args[0] + f'\nFailed to open file: {fileName}.')
                MsgBox(error.args[0] + f'\nFailed to open file: {fileName}.')
                break
            else:
                # check whether to perform T->A conversion
                spec0 = irMap.imageCube[0, 0, :]
                maxSpecY = np.max(spec0)
                if (not self.T2AConvertStatus) and (maxSpecY >= self.minYLimit):
                    userMsg = YesNoDialog(f'max(Y) of the first spectrum is greater than {self.minYLimit}, \
                            while the "Auto T->A" box is not checked. \nPlease make sure data format is in absorbance.\
                            \nDo you want to perform "Auto T->A" conversion?')
                    # get user choice
                    userMsg.addButton(QMessageBox.YesToAll)
                    userChoice = userMsg.choice()
                    if userChoice == QMessageBox.YesToAll:  # set 'auto T->A' on
                        self.sigT2A.emit(True)
                    if (userChoice == QMessageBox.YesToAll) or (userChoice == QMessageBox.Yes):
                        irMap.imageCube = -np.log10(irMap.imageCube / 100 + self.epsilon)
                        irMap.data = -np.log10(irMap.data / 100 + self.epsilon)
                        self.sigText.emit(f'User chooses to perform T->A conversion in {fileName}.')
                    else:
                        self.sigText.emit(f'User chooses not to perform T->A conversion in {fileName}.')
                elif maxSpecY >= self.minYLimit:
                    irMap.imageCube = -np.log10(irMap.imageCube / 100 + self.epsilon)
                    irMap.data = -np.log10(irMap.data / 100 + self.epsilon)
                    self.sigText.emit(f'T->A conversion is performed in {fileName}.')
                else:
                    self.sigText.emit(f"{fileName}'s datatype is absorbance. \nT->A conversion is not performed.")

                dataCube = np.moveaxis(np.flipud(irMap.imageCube), -1, 0)
                # set up required data/properties in self.imageview and show image
                row, col = irMap.imageCube.shape[0], irMap.imageCube.shape[1]
                wavenumbers = irMap.wavenumbers
                rc2ind = {tuple(x[1:]): x[0] for x in irMap.ind_rc_map}
                self.sigImage.emit((row, col, wavenumbers, rc2ind, dataCube))

                # save hdf5
                h5Name = fileName[:-4] + '.h5'
                try:
                    irMap.write_as_hdf5(folderPath + h5Name)
                    self.sigText.emit(f'#{i + 1} out of {n_files} maps HDF5-conversion complete! \
                            \nHDF5 File Location: {folderPath + h5Name}')
                except Exception as error:
                    MsgBox(error.args[0], 'error')
                    break
            # QTimer.singleShot(0, lambda: self.infoBox.setText(f'Start processing #{i + 1} out of {n_files} files.'))
        # MsgBox('All file conversion complete!')
        self.quit()

