# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:34:27 2019

@author: lchen43
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from lbl_ir.data_objects.ir_map import val2ind

class ispectrum:
    
    def __init__(self, wavenumbers, y):
        self.f, self.ax = plt.subplots(figsize=(8,4))
        self.wavenumbers, self.y = wavenumbers, y
        self.ax.plot(wavenumbers, y)
        plt.xlim([4000,500])
        plt.title('Click on the spectrum to select peaks; right click to cancel selections');
        plt.show()
        
        self.pos = []
        self.connect()
    
    def connect(self):
        '''connect to all the events 
        '''
        self.cidpress = self.f.canvas.mpl_connect('button_press_event', self.onclick)
        self.cidmotion = self.f.canvas.mpl_connect('motion_notify_event', self.hover)
        self.cidenter_axes = self.f.canvas.mpl_connect('axes_enter_event', self.in_axes)
        self.cidleave_axes = self.f.canvas.mpl_connect('axes_leave_event', self.leave_axes)
    
    def disconnect(self):
        '''disconnect all the stored connection ids
        '''
        self.f.canvas.mpl_disconnect(self.cidpress)
        self.f.canvas.mpl_disconnect(self.cidmotion)
        self.f.canvas.mpl_disconnect(self.cidenter_axes)
        self.f.canvas.mpl_disconnect(self.cidleave_axes)
        return self.pos

    def onclick(self, event):
        ind = val2ind(event.xdata, self.wavenumbers)
        if event.button == 1:
            self.pos.append([self.wavenumbers[ind], self.y[ind]])
            self.ax.plot(self.wavenumbers[ind], self.y[ind],'ro')
            self.ax.text(self.wavenumbers[ind], self.y[ind]+0.03,str(len(self.pos)))
            self.ax.texts[-1], self.ax.texts[-2] = self.ax.texts[-2], self.ax.texts[-1] 
        else:
            if len(self.pos) > 0:
                self.pos.pop()
                self.ax.lines[-1].remove()
                self.ax.texts[-2].remove()

    def hover(self, event):
        ind = val2ind(event.xdata, self.wavenumbers)
        self.ax.patches[-1].set_center((self.wavenumbers[ind],self.y[ind]))
        self.ax.texts[-1].set_position((self.wavenumbers[ind],self.y[ind]+0.08))
        self.ax.texts[-1].set_text(str(self.wavenumbers[ind]) + ', ' +str(self.y[ind]))


    def in_axes(self, event): 
        self.ax.texts = []
        self.ax.patches = []
        if event.inaxes:
            ind = val2ind(event.xdata, self.wavenumbers)
            self.ax.add_patch(Circle((self.wavenumbers[ind], self.y[ind]), radius = 0.05, color = 'r'))
            if len(self.pos) > 0:
                for i in range(len(self.pos)):
                    self.ax.text(self.pos[i][0], self.pos[i][1]+0.03, str(i+1))
            self.ax.text(self.wavenumbers[ind], self.y[ind], str(len(self.ax.texts))+ '-' + str(self.wavenumbers[ind]) + ', ' +str(self.y[ind]))

    def leave_axes(self, event):
        self.ax.texts = []
        self.ax.patches = []

if __name__ == "__main__":

    import os
    from ..io_tools.map_IO import read_spa
    
    test_data_home = '../test_irdata/'
    
    spa_file = os.path.join(test_data_home, 'test_data0001.spa')
    wavenumbers, y, _, _ = read_spa(spa_file)
    spec = ispectrum(wavenumbers, y)