"""
Created on Tue Dec 18 16:14:43 2018

@author: Liang Chen
"""

import spectral.io.envi as envi
import numpy as np
import re


def read_envi(hdr_file):
    """Load an ENVI map from the .hdr header file

    Parameters:
    -----------
    hdr_file: string
        file path for a .hdr file

    Returns:
    --------
    out : object
        a spectral.io.bipfile.BipFile object

    Examples:
    ---------
    >>> hdr_file = os.path.join(test_data_home, 'test_envi.hdr')
    >>> img = envi.open(hdr_file)
    (17, 32, 1738)
    """

    # read in parameter list in the header file
    params = []

    with open(hdr_file, 'r', encoding='utf-8', errors='ignore') as header:
        for line in header:
            params.append(line.split('=')[0].strip())

    # if parameter 'byte order' is not found, append 'byte order = 0' to the header file
    if 'byte order' not in params:
        with open(hdr_file, 'a', encoding='utf-8', errors='ignore') as header:
            header.write('byte order = 0\n')

    img = envi.open(hdr_file)

    return img


def read_binary(fileObj, byteType='uint8', size=1):
    """A helper function to readin values from a binary file

    Parameters:
    -----------
    fileObj : object 
        a binary file object

    byteType : string, optional, default 'uint8'
        the type of readin values

    size : int, optional
        the number of bytes to readin. Default is 1.

    Returns:
    --------
    out : a value or a tuple of values or a string
        the readout value from the bytes
    """
    import struct

    typeNames = {
        'int8': ('b', struct.calcsize('b')),
        'uint8': ('B', struct.calcsize('B')),
        'int16': ('h', struct.calcsize('h')),
        'uint16': ('H', struct.calcsize('H')),
        'int32': ('i', struct.calcsize('i')),
        'uint32': ('I', struct.calcsize('I')),
        'int64': ('q', struct.calcsize('q')),
        'uint64': ('Q', struct.calcsize('Q')),
        'float': ('f', struct.calcsize('f')),
        'double': ('d', struct.calcsize('d')),
        'char': ('s', struct.calcsize('s'))}

    if size == 1:
        return struct.unpack(typeNames[byteType][0], fileObj.read(typeNames[byteType][1]))[0]
    elif size > 1:
        return struct.unpack(typeNames[byteType][0] * size, fileObj.read(typeNames[byteType][1] * size))
    else:
        return None


def read_spa(spa_file):
    """Load a spectrum from a .spa file

    Parameters:
    -----------
    spa_file: string 
        file path for a .spa file

    Returns:
    --------
    wavenumbers : float array 
        the wavenumber values of the spectrum (x-axis)

    spectrum : float array
        the transmission/reflection/absorption coefficient
               at wavenumber of the spectrum (y-axis)

    title : string
        the title of the spectrum

    comment : string
        the comment section of the spectrum, if it exists

    Examples:
    ---------
    >>> spa_file = os.path.join(test_data_home, 'test_data0001.spa')
    >>> wavenumbers, spectrum, title, comment = read_spa(spa_file)
    >>> print(title)
    C:\\Users\\lchen43\\Documents\\CDIPS_2017\\lbl-ir\\test_data\\test_data.map - Spectrum #1  
    Position (X,Y): 499.51, 5715.69  Thu Apr 22 13:57:44 2010 (GMT-07:00)
    >>> print(comment)
    Split map from: C:\\Users\\lchen43\\Documents\\CDIPS_2017\\lbl-ir\\test_data\\test_data.map
    X Range: 499.51, 654.51
    Y Range: 5715.69, 5795.69
    Position (X,Y): 499.51, 5715.69
    Thu Apr 22 13:57:44 2010 (GMT-07:00)
    """

    with open(spa_file, 'rb') as f:
        f.seek(30)
        readChar = read_binary(f, 'uint8', 255)
        title = ''.join([chr(i) for i in filter(lambda x: x > 0, readChar)])

        f.seek(564)
        spectrumPts = read_binary(f, 'int32')

        f.seek(576)
        maxWavenum = read_binary(f, 'float')
        minWavenum = read_binary(f, 'float')
        wavenumbers = np.linspace(maxWavenum, minWavenum, spectrumPts)

        # The starting byte location of the spectrum data is stored in the
        # header. It immediately follows a flag value of 3.
        # If there is a comment section, the flag value is 27 and it should be in front of the flag value of 3.
        f.seek(338)
        Flag = 0
        while (Flag != 3 and Flag != 27):
            Flag = read_binary(f, 'uint16')

        if Flag == 3:  # no comment section, look for data starting position
            dataPosition = read_binary(f, 'uint16')
            f.seek(dataPosition)
            spectrum = read_binary(f, 'float', spectrumPts)
            comment = ''
        elif Flag == 27:  # there is a comment section, look for comment starting position
            commentPosition = read_binary(f, 'uint16')
            # move forward 14 bytes, that's the data starting position
            f.seek(14, 1)
            dataPosition = read_binary(f, 'uint16')
            commentLength = dataPosition - commentPosition

            f.seek(commentPosition)
            readChar = read_binary(f, 'uint8', commentLength)
            comment = ''.join([chr(i)
                               for i in filter(lambda x: x > 0, readChar)])
            f.seek(dataPosition)
            spectrum = read_binary(f, 'float', spectrumPts)
        else:
            raise Exception('The flag value of 3 or 27 cannot be found')

        spectrum = np.array(spectrum)
        return wavenumbers, spectrum, title, comment


def read_series(file_name, wavLen=1738):
    """
    read Ominc series map
    :param file_name: path of minc series map
    :param wavLen: the length of the wavenumbers vector
    :return:
    wav: the wavenumbers vector
    spectra: all spectra in the series map
    xy: all xy coordinate in the series map
    """

    wav = spectra = xy = None

    with open(file_name, 'rb') as fid:
        data = fid.read()
        # read out firstWav, lastWav, construct wav
        fid.seek(0)
        s = []
        for i in range(1000):
            s.append((read_binary(fid, 'uint32')))
        offset = s.index(wavLen) * 4
        fid.seek(offset + 12)
        val = read_binary(fid, 'float', 2)
        firstWav, lastWav = val[1], val[0]
        wav = np.linspace(firstWav, lastWav, wavLen)

        # read out num of spectra
        Chain = bytes('Spectrum', 'utf-8')
        firstByte = data.index(Chain)
        s1 = str(data[firstByte:(firstByte + 100 - 16)])
        exp = re.compile('(-?[0-9]+\.?[0-9]*)')
        tmpValues = exp.findall(s1)
        nSpectra = int(tmpValues[1])

        # read out all spectra
        spectra = np.zeros((nSpectra, wavLen))
        delta = wavLen * 4 + 96
        for i in range(nSpectra):
            fid.seek(firstByte + delta * i + 80)
            spectra[i, :] = read_binary(fid, 'float', wavLen)

        # find xy positions
        chain = bytes('Position', 'utf-8')
        firstPos = data.index(chain)
        secondPos = data[(firstPos + 1):].index(chain)
        secondPos += firstPos + 1
        fid.seek(secondPos + 48)
        val = read_binary(fid, 'float', 2 * nSpectra)
        xy = np.zeros((nSpectra, 2))
        xy[:, 0], xy[:, 1] = val[::2], val[1::2]

    return wav, spectra, xy


if __name__ == "__main__":
    import os

    test_data_home = '../../test_irdata/'

    hdr_file = os.path.join(test_data_home, 'test_envi.hdr')
    img = read_envi(hdr_file)
    print('====Envi file====')
    print(img.shape)

    spa_file = os.path.join(test_data_home, 'test_data0001.spa')
    wavenumbers, spectrum, title, comment = read_spa(spa_file)
    print('====' + title + '====')
    print(comment)

    map_file = os.path.join(test_data_home, 'typeII-010_12x9.map')
    wavenumbers, spectra, xy = read_series(map_file)
    print('====Series map====')
    print(spectra.shape)
    print(xy[:3,:])
    print(spectra[:4, -3:])
