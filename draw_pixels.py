"""
Provides functionality for drawing on a canvas and copying the
contents to another canvas. Intended for controlling the display of a
digital mirror array from another monitor. To account for optical path
distortion, a transformation matrix can be provided as a
column-separated file.
"""

from __future__ import print_function
import cv2
import filepicker
import numpy as np
import skimage.transform as tf


# Constants
WHITE = 65535
BLACK = 0

# Hardware constants
dmaDim = (1200, 800)
monitorDim = (512, 512)
cameraDim = (256, 256)

# Initial parameters
DRAWING = False
ERASING = False
SHAPE = 'circle'
SIZE = 5

CALIBRATION_MODE = False

inv_transform = np.eye(3, dtype=np.float32)

camera_img = np.zeros(monitorDim, np.uint16)
camera = camera_img.copy()

dma = np.zeros(dmaDim[-1::-1], np.uint8)
dma_temp = dma.copy()


def calibration_pattern(nx=4, ny=4, s=SIZE):
    global dmaDim
    w, h = dmaDim
    # Minimum spacing
    # sx = int(w / (nx * s) - 2)
    # sy = int(h / (ny * s) - 2)

    # Maximum spacing
    # sx = int(2 * s + (w - nx * 2 * s) / (nx - 1))
    # sy = int(2 * s + (h - ny * 2 * s) / (ny - 1))

    sx = int((w - nx * 2 * s) / (nx - 1) / 2)
    sy = int((h - ny * 2 * s) / (ny - 1) / 2)

    pattern = np.zeros(dmaDim[-1::-1])
    for i in range(nx):
        for j in range(ny):
            cv2.circle(pattern,
                       (i * sx + s, j * sy + s),
                       s, (WHITE, WHITE, WHITE), -1)
    return pattern


def draw(event, x, y, flags, param):
    """
    Mouse callback function. Handles events for drawing (left-click)
    and erasing (right-click).
    """
    global DRAWING, ERASING, SHAPE, SIZE, camera

    if event == cv2.EVENT_LBUTTONDOWN:
        DRAWING = True
    if event == cv2.EVENT_RBUTTONDOWN:
        ERASING = True

    if event == cv2.EVENT_MOUSEMOVE or \
       (cv2.EVENT_LBUTTONDOWN or cv2.EVENT_RBUTTONDOWN):
        if DRAWING:
            if SHAPE == 'circle':
                cv2.circle(camera, (x, y), SIZE, (WHITE, WHITE, WHITE), -1)
                cv2.circle(camera_img, (x, y), SIZE, (WHITE, WHITE, WHITE), -1)
            if SHAPE == 'rect':
                cv2.rectangle(camera,
                              (x-SIZE, y-SIZE),
                              (x+SIZE, y+SIZE),
                              (WHITE, WHITE, WHITE), -1)
                cv2.rectangle(camera_img,
                              (x-SIZE, y-SIZE),
                              (x+SIZE, y+SIZE),
                              (WHITE, WHITE, WHITE), -1)
        if ERASING:
            if SHAPE == 'circle':
                cv2.circle(camera, (x, y), SIZE, (BLACK, BLACK, BLACK), -1)
                cv2.circle(camera_img, (x, y), SIZE, (BLACK, BLACK, BLACK), -1)
            if SHAPE == 'rect':
                cv2.rectangle(camera,
                              (x-SIZE, y-SIZE),
                              (x+SIZE, y+SIZE),
                              (BLACK, BLACK, BLACK), -1)
                cv2.rectangle(camera_img,
                              (x-SIZE, y-SIZE),
                              (x+SIZE, y+SIZE),
                              (BLACK, BLACK, BLACK), -1)

    if event == cv2.EVENT_LBUTTONUP:
        DRAWING = False
    if event == cv2.EVENT_RBUTTONUP:
        ERASING = False


def loadtransform(filename):
    """
    Mostly a wrapper around ``numpy.loadtxt`` for ``cv2`` float32
    matrix. Returns an ``skimage.transform`` and uses a (crude)
    heuristic to try to guess what kind of transformation has been
    read. It has to deal with the impedance mismatch between ``cv2``
    and ``skimage`` datatypes, float32 and float64, respectively.
    """
    params = np.loadtxt(filename, delimiter=',', dtype=np.float32)
    if len(params) == 2:
        return tf.PolynomialTransform(params)
    else:
        return tf.ProjectiveTransform(params)


def handlekey(key):
    """
    Apply action after keypress.
    """
    global camera, camera_img, dma, inv_transform, CALIBRATION_MODE, SHAPE, SIZE
    if key == 27:
        cv2.destroyAllWindows()
        return 1
    # Clear the screen with the BACKSPACE key
    elif key == 8:
        camera[:] = 0
        dma[:] = 0
    # Send image from camera to dma with the ENTER key. The keycode
    # for return will depend on the platform, so \n and \r are both
    # handled.
    elif key == ord('\n') or key == ord('\r'):
        dma = tf.warp(image=camera,
                      inverse_map=inv_transform,
                      output_shape=dmaDim[-1::-1],
                      order=0)
        # dma = tf.warp(image=cv2.resize(camera, cameraDim),
        #               inverse_map=inv_transform,
        #               output_shape=dmaDim[-1::-1],
        #               order=0)
    elif key == ord('f'):
        cv2.setWindowProperty('dma', cv2.WND_PROP_FULLSCREEN, 1)
    elif key == ord('F'):
        cv2.setWindowProperty('dma', cv2.WND_PROP_FULLSCREEN, 0)
    elif key == ord('r'):
        SHAPE = 'rect'
    elif key == ord('c'):
        SHAPE = 'circle'
    elif key == ord('s'):
        print('size = {}'.format(SIZE))
    elif key == ord('='):
        SIZE += 2
    elif key == ord('-'):
        SIZE -= 2
        if SIZE <= 1:
            SIZE = 1
    elif key == ord('@'):
        dma[:] = WHITE
    elif key == ord('!'):
        dma[:] = BLACK
    elif key == ord('o'):
        filename = filepicker.filepicker()
        if filename != '':
            inv_transform = loadtransform(filename)
    elif key == ord('h'):
        printdoc()
    elif key == ord('C'):
        CALIBRATION_MODE = True
        dma = calibration_pattern()
    if CALIBRATION_MODE:
        CALIBRATION_STEP = 10
        if key == ord('W'):
            M = np.float32([[1, 0, 0],
                            [0, 1, -CALIBRATION_STEP]])
            dma = cv2.warpAffine(dma, M, dmaDim)
            # dma = tf.warp(image=dma,
            #               inverse_map=tf.EuclideanTransform(translation=(0, 1)).inverse,
            #               output_shape=dma.shape)
        elif key == ord('A'):
            M = np.float32([[1, 0, -CALIBRATION_STEP],
                            [0, 1, 0]])
            dma = cv2.warpAffine(dma, M, dmaDim)
        elif key == ord('S'):
            M = np.float32([[1, 0, 0],
                            [0, 1, CALIBRATION_STEP]])
            dma = cv2.warpAffine(dma, M, dmaDim)
        elif key == ord('D'):
            M = np.float32([[1, 0, CALIBRATION_STEP],
                            [0, 1, 0]])
            dma = cv2.warpAffine(dma, M, dmaDim)
        elif key == ord('=') or key == ord('-'):
            dma = calibration_pattern(s=SIZE)
        elif key == ord('\n') or key == ord('\r'):
            CALIBRATION_MODE = False
    return 0


def printdoc():
    """
    Print important keyboard shortcuts.
    """
    doc = {
        'BACKSPACE': 'Clear all screens',
        'f/F': 'Fullscreen/restore the dma window size',
        'ENTER': 'Send image to DMA',
        'c/r': 'Set brush shape to circle/rectangle (default is circle)',
        's': 'Print size of brush',
        '=/-': 'Increase/decrease brush size',
        '!/@': 'Set dma to black/white',
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


def _generate_maps(tform):
    global cameraDim, dmaDim
    # src, dst, tform = _warp_test()
    x, y = np.meshgrid(np.arange(dmaDim[0]), np.arange(dmaDim[1]))
    on = np.ones(x.shape, dtype=x.dtype)
    powers = np.stack((on, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3))
    mapx = np.zeros(dmaDim[-1::-1], dtype='float32')
    mapy = np.zeros(dmaDim[-1::-1], dtype='float32')
    for i in range(10):
        mapx += tform[0, i] * powers[i]
        mapy += tform[1, i] * powers[i]
    map1, map2 = cv2.convertMaps(map1=mapx,
                                 map2=mapy,
                                 dstmap1type=cv2.CV_16SC2,
                                 nninterpolation=False)
    return map1, map2


def main():
    import argparse
    import sys

    global cameraDim, dmaDim, dma_temp, inv_transform, monitorDim 

    parser = argparse.ArgumentParser(description='Process stuff.')
    parser.add_argument('--file',
                        metavar='inverseTransformFile',
                        type=str,
                        default='')

    args, _ = parser.parse_known_args()
    if args.file != '':
        inv_transform = loadtransform(args.file)
    else:
        print('No transformation file specified, using identity.',
              file=sys.stderr)

    # # Create a black image, a window and bind the function to window
    # camera_img = np.zeros(monitorDim, np.uint16)
    # camera = camera_img.copy()

    # dma = np.zeros(dmaDim[-1::-1], np.uint8)
    # dma_temp = dma.copy()

    cv2.namedWindow('dma', cv2.WINDOW_NORMAL)
    cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('camera', draw)

    while True:
        key = cv2.waitKey(50) & 0xFF
        # ``handlekey`` returns 1 when program should quit.
        if handlekey(key):
            break
        cv2.imshow('camera', camera)
        cv2.imshow('dma', dma)


if __name__ == '__main__':
    main()
