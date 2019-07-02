import sys
import os
from calibration import main
from filepicker import filepicker
from numpy import savetxt

from tkinter.filedialog import asksaveasfilename


# must be -defaultextension, -filetypes, -initialdir, -initialfile,
# -multiple, -parent, -title, or -typevariable
options = {'initialdir': os.curdir, 'title': 'Screen image'}
screen = filepicker(**options)
options['initialdir'] = '/'.join(screen.split('/')[:-1])
options['title'] = 'Sample image'
sample = filepicker(**options)

sys.argv = ['python', __name__,'--screen', screen, '--sample', sample]
tform = main()

options['title'] = 'Save transformation file as'
options['defaultextension'] = '.csv'

fname = asksaveasfilename(**options)
savetxt(fname, tform, delimiter=',')
