"""
Provides functionality for drawing on a canvas and copying the
contents to another canvas. Intended for controlling the display of a
digital mirror array from another monitor. To account for optical path
distortion, a transformation matrix can be provided as a
column-separated file.
"""

from __future__ import division, print_function

import argparse
import sys

import cv2
import numpy as np

from filepicker import filepicker


class Canvas:
    def __init__(self):
        # Constants
        self.white = 255
        self.black = 0

        # Hardware constants
        self.dma_dim = (1280, 800)
        self.monitor_dim = (512, 512)
        self.camera_dim = (256, 256)

        # Initial parameters
        self.drawing = False
        self.erasing = False
        self.shape = 'circle'
        self.size = 5

        self.calibration_mode = False
        self.calibration_step = 10

        self.order = 3
        self.inv_transform = np.array(
            [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
            dtype='float32')

        self.camera_img = np.zeros(self.monitor_dim, 'uint8')
        self.camera = self.camera_img.copy()

        self.dma = 255 * np.ones(self.dma_dim[::-1], 'uint8')

        self.map1 = np.zeros(self.dma.shape, dtype='float32')
        self.map2 = np.zeros(self.dma.shape, dtype='float32')

        self.mode = '(BLACK)'

    def draw(self, event, x, y, flags, param):
        """
        Mouse callback function. Handles events for drawing
        (left-click) and erasing (right-click).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        if event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True

        if event == cv2.EVENT_MOUSEMOVE or \
           (cv2.EVENT_LBUTTONDOWN or cv2.EVENT_RBUTTONDOWN):
            if self.drawing:
                if self.shape == 'circle':
                    cv2.circle(self.camera, (x, y), self.size,
                               (self.white, self.white, self.white), -1)
                    cv2.circle(self.camera_img, (x, y), self.size,
                               (self.white, self.white, self.white), -1)
                if self.shape == 'rect':
                    cv2.rectangle(self.camera, (x - self.size, y - self.size),
                                  (x + self.size, y + self.size),
                                  (self.white, self.white, self.white), -1)
                    cv2.rectangle(self.camera_img,
                                  (x - self.size, y - self.size),
                                  (x + self.size, y + self.size),
                                  (self.white, self.white, self.white), -1)
            if self.erasing:
                if self.shape == 'circle':
                    cv2.circle(self.camera, (x, y), self.size,
                               (self.black, self.black, self.black), -1)
                    cv2.circle(self.camera_img, (x, y), self.size,
                               (self.black, self.black, self.black), -1)
                if self.shape == 'rect':
                    cv2.rectangle(self.camera, (x - self.size, y - self.size),
                                  (x + self.size, y + self.size),
                                  (self.black, self.black, self.black), -1)
                    cv2.rectangle(self.camera_img,
                                  (x - self.size, y - self.size),
                                  (x + self.size, y + self.size),
                                  (self.black, self.black, self.black), -1)

        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        if event == cv2.EVENT_RBUTTONUP:
            self.erasing = False

    def load_transform(self, filename):
        """
        Mostly a wrapper around ``numpy.loadtxt`` for ``cv2`` float32
        matrix. Returns an ``skimage.transform`` and uses a (crude)
        heuristic to try to guess what kind of transformation has been
        read. It has to deal with the impedance mismatch between
        ``cv2`` and ``skimage`` datatypes, float32 and float64,
        respectively.
        """
        self.inv_transform = np.loadtxt(
            filename, delimiter=',', dtype='float32')
        self.generate_maps()

    def handle_key(self, key):
        """
        Apply action after keypress.
        """
        if key == 27:
            cv2.destroyAllWindows()
            return 1
        # Clear the screen with the BACKSPACE key
        elif key == 8:
            self.camera[:] = self.black
            self.dma[:] = self.white
        # Send image from CAMERA to DMA with the ENTER key. The keycode
        # for return will depend on the platform, so \n and \r are both
        # handled.
        elif key == ord('\n') or key == ord('\r'):
            self.calibration_mode = False
            self.dma = cv2.remap(
                src=self.camera ^ self.white,
                map1=self.map1,
                map2=self.map2,
                interpolation=cv2.INTER_AREA,
                borderValue=self.white)
            self.mode = '(SENT)'
        elif key == ord('f'):
            cv2.setWindowProperty('DMA', cv2.WND_PROP_FULLSCREEN, 1)
        elif key == ord('F'):
            cv2.setWindowProperty('DMA', cv2.WND_PROP_FULLSCREEN, 0)
        elif key == ord('r'):
            self.shape = 'rect'
        elif key == ord('c'):
            self.shape = 'circle'
        elif key == ord('s'):
            print('diameter = {} px'.format(2 * self.size + 1))
        elif key == ord('='):
            self.size += 1
        elif key == ord('-'):
            self.size -= 1
            if self.size <= 0:
                self.size = 0
        elif key == ord('@'):
            self.dma[:] = self.black  # WHITE
            self.mode = '(WHITE)'
        elif key == ord('!'):
            self.dma[:] = self.white  # BLACK
            self.mode = '(BLACK)'
        elif key == ord('o'):
            filename = filepicker()
            if filename != '':
                self.load_transform(filename)
        elif key == ord('h'):
            printdoc()
        elif key == ord('C'):
            self.dma = self.calibration_pattern()
        elif key == ord('T'):
            self.camera = self.calibration_pattern(
                dim=self.monitor_dim, img=self.camera, screen_fraction=0.7)
            self.camera ^= self.white
        if self.calibration_mode:
            if key == ord('W'):
                M = np.float32([[1, 0, 0], [0, 1, -self.calibration_step]])
                self.dma = cv2.warpAffine(
                    self.dma, M, self.dma_dim, borderMode=cv2.BORDER_WRAP)
            elif key == ord('A'):
                M = np.float32([[1, 0, -self.calibration_step], [0, 1, 0]])
                self.dma = cv2.warpAffine(
                    self.dma, M, self.dma_dim, borderMode=cv2.BORDER_WRAP)
            elif key == ord('S'):
                M = np.float32([[1, 0, 0], [0, 1, self.calibration_step]])
                self.dma = cv2.warpAffine(
                    self.dma, M, self.dma_dim, borderMode=cv2.BORDER_WRAP)
            elif key == ord('D'):
                M = np.float32([[1, 0, self.calibration_step], [0, 1, 0]])
                self.dma = cv2.warpAffine(
                    self.dma, M, self.dma_dim, borderMode=cv2.BORDER_WRAP)
            elif key == ord('=') or key == ord('-'):
                self.dma = self.calibration_pattern()
            # Clear the screen with the BACKSPACE key
            elif key == 8:
                self.calibration_mode = False
                self.mode = ''
        return 0

    def generate_maps(self):
        # src, dst, tform = _warp_test()
        x, y = np.meshgrid(
            np.arange(self.dma_dim[0]), np.arange(self.dma_dim[1]))
        powers = (x**(i - j) * y**j
                  for i in range(self.order + 1)
                  for j in range(i + 1))
        mapx = np.zeros(self.dma_dim[::-1], dtype='float32')
        mapy = np.zeros(self.dma_dim[::-1], dtype='float32')
        for i, p in enumerate(powers):
            mapx += self.inv_transform[0, i] * p
            mapy += self.inv_transform[1, i] * p
        self.map1, self.map2 = cv2.convertMaps(
            map1=mapx, map2=mapy, dstmap1type=cv2.CV_16SC2,
            nninterpolation=False)

    def calibration_pattern(self,
                            nx=4,
                            ny=4,
                            screen_fraction=0.3,
                            dim=None,
                            img=None):
        self.calibration_mode = True
        self.mode = '(CALIBRATION)'
        if dim is None:
            dim = self.dma_dim
        if img is None:
            img = self.dma
        width, height = dim
        spacing = int(screen_fraction * min(dim) / (min(nx, ny) - 1))
        pattern = np.zeros(dim[::-1], dtype=img.dtype)
        for i in range(nx):
            for j in range(ny):
                if self.size != 0:
                    cv2.circle(
                        pattern,
                        (i * spacing + (width - spacing * (nx - 1)) // 2,
                         j * spacing + (height - spacing * (ny - 1)) // 2),
                        self.size, (self.white, self.white, self.white),
                        -1,
                        lineType=cv2.LINE_AA)
                else:
                    pattern[j * spacing + self.size, i * spacing +
                            self.size] = self.white
        return pattern ^ self.white


def printdoc():
    """
    Print important keyboard shortcuts.
    """
    doc = {
        'BACKSPACE': 'Clear all screens',
        'f/F': 'Fullscreen/restore the DMA window size',
        'ENTER': 'Send image to DMA',
        'c/r': 'Set brush shape to circle/rectangle (default is circle)',
        's': 'Print size of brush',
        '=/-': 'Increase/decrease brush size',
        '!/@': 'Set DMA to black/white',
        'o': 'Open new transformation file',
        'h': 'Show this help dialog',
        'left-click/right-click': 'Draw/erase with brush.'
    }
    from operator import itemgetter
    title = 'Keyboard shortcuts'
    space = max(map(len, doc.keys()))
    print(title)
    # Sort help by description.
    for k, v in sorted(doc.items(), key=itemgetter(1)):
        print('{:>{s}} : {}'.format(k, v, s=space))


def _warp_test():
    src = np.array([[351., 135.], [511., 135.], [671., 135.], [831., 135.],
                    [351., 231.], [511., 231.], [671., 231.], [831., 231.],
                    [351., 327.], [511., 327.], [671., 327.], [831., 327.],
                    [351., 424.], [511., 424.], [671., 424.], [831., 424.]])

    dst = np.array([[440., 449.], [319., 457.], [199., 465.], [79., 473.],
                    [437., 311.], [317., 322.], [196., 330.], [75., 341.],
                    [435., 173.], [314., 184.], [192., 194.], [71., 203.],
                    [430., 39.], [310., 46.], [189., 53.], [68., 63.]])

    tform = np.array([[
        7.00374748e+02, -7.30131577e-01, 1.53052776e-02, -3.04479066e-05,
        -5.03457332e-05, -8.34824240e-05, 2.43427833e-08, -5.38033585e-08,
        1.82036704e-07, -5.14826599e-08
    ], [
        6.13218816e+02, 1.17450369e-01, -1.46495340e+00, -1.69348298e-04,
        2.86958525e-04, -1.49289015e-04, 8.61463811e-08, 6.86296386e-08,
        -6.59559586e-07, 5.84696922e-07
    ]])

    return src, dst, tform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process stuff.')
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--invert', type=bool, default=False)
    args, _ = parser.parse_known_args()
    canvas = Canvas()

    if args.file:
        canvas.load_transform(args.file)
    else:
        print(
            'No transformation file specified, using identity.',
            file=sys.stderr)
        canvas.generate_maps()

    cv2.namedWindow('DMA', cv2.WINDOW_NORMAL)
    cv2.namedWindow('CAMERA', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('CAMERA', canvas.draw)

    while True:
        key = cv2.waitKey(50) & 0xFF
        # ``handle_key`` returns 1 when program should quit.
        if canvas.handle_key(key):
            break
        cv2.imshow('CAMERA', canvas.camera)
        cv2.setWindowTitle(
            'CAMERA', 'CAMERA {}    SIZE {}'.format(canvas.mode,
                                                    2 * canvas.size + 1))
        cv2.imshow('DMA', canvas.dma)
