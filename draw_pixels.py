"""
Provides functionality for drawing on a canvas and copying the
contents to another canvas. Intended for controlling the display of a
digital mirror array from another monitor. To account for optical path
distortion, a transformation matrix can be provided as a
column-separated file.
"""

from __future__ import division, print_function
import numpy as np
from skimage import transform
import cv2
import filepicker

# Constants
WHITE = 65535
BLACK = 0

# Hardware constants
DMA_DIM = (1200, 800)
MONITOR_DIM = (512, 512)
CAMERA_DIM = (256, 256)

# Initial parameters
DRAWING = False
ERASING = False
SHAPE = 'circle'
SIZE = 5

CALIBRATION_MODE = False

INV_TRANSFORM = np.eye(3, dtype=np.float32)

CAMERA_IMG = np.zeros(MONITOR_DIM, np.uint16)
CAMERA = CAMERA_IMG.copy()

DMA = np.zeros(DMA_DIM[-1::-1], np.uint8)
DMA_TEMP = DMA.copy()

MAP1 = np.zeros(DMA.shape, dtype='float32')
MAP2 = np.zeros(DMA.shape, dtype='float32')


def calibration_pattern(nx=4, ny=4, screen_fraction=0.4):
    global DMA_DIM, SIZE
    width, height = DMA_DIM

    # Minimum spacing
    # sx = int(w / (nx * s) - 2)
    # sy = int(h / (ny * s) - 2)

    # Maximum spacing
    # sx = int(2 * s + (w - nx * 2 * s) / (nx - 1))
    # sy = int(2 * s + (h - ny * 2 * s) / (ny - 1))

    # Constant spacing
    # s_xx = int(screen_fraction * width / (nx - 1))
    s_y = int(screen_fraction * height / (ny - 1))

    # sx = int((screen_fraction * w - nx * 2 * s) / (nx - 1))
    # sy = int((screen_fraction * h - ny * 2 * s) / (ny - 1))

    pattern = np.zeros(DMA_DIM[-1::-1])
    for i in range(nx):
        for j in range(ny):
            if SIZE != 0:
                cv2.circle(pattern, (i * s_y + SIZE, j * s_y + SIZE), SIZE,
                           (WHITE, WHITE, WHITE), -1)
            else:
                pattern[j * s_y + SIZE, i * s_y + SIZE] = WHITE
    return pattern


def draw(event, x, y, flags, param):
    """
    Mouse callback function. Handles events for drawing (left-click)
    and erasing (right-click).
    """
    global CAMERA, DRAWING, ERASING, SIZE, SHAPE

    if event == cv2.EVENT_LBUTTONDOWN:
        DRAWING = True
    if event == cv2.EVENT_RBUTTONDOWN:
        ERASING = True

    if event == cv2.EVENT_MOUSEMOVE or \
       (cv2.EVENT_LBUTTONDOWN or cv2.EVENT_RBUTTONDOWN):
        if DRAWING:
            if SHAPE == 'circle':
                cv2.circle(CAMERA, (x, y), SIZE, (WHITE, WHITE, WHITE), -1)
                cv2.circle(CAMERA_IMG, (x, y), SIZE, (WHITE, WHITE, WHITE), -1)
            if SHAPE == 'rect':
                cv2.rectangle(CAMERA, (x - SIZE, y - SIZE),
                              (x + SIZE, y + SIZE), (WHITE, WHITE, WHITE), -1)
                cv2.rectangle(CAMERA_IMG, (x - SIZE, y - SIZE),
                              (x + SIZE, y + SIZE), (WHITE, WHITE, WHITE), -1)
        if ERASING:
            if SHAPE == 'circle':
                cv2.circle(CAMERA, (x, y), SIZE, (BLACK, BLACK, BLACK), -1)
                cv2.circle(CAMERA_IMG, (x, y), SIZE, (BLACK, BLACK, BLACK), -1)
            if SHAPE == 'rect':
                cv2.rectangle(CAMERA, (x - SIZE, y - SIZE),
                              (x + SIZE, y + SIZE), (BLACK, BLACK, BLACK), -1)
                cv2.rectangle(CAMERA_IMG, (x - SIZE, y - SIZE),
                              (x + SIZE, y + SIZE), (BLACK, BLACK, BLACK), -1)

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
        return transform.PolynomialTransform(params)
    else:
        return transform.ProjectiveTransform(params)


def handlekey(key):
    """
    Apply action after keypress.
    """
    global CALIBRATION_MODE, CAMERA, CAMERA_IMG, DMA, INV_TRANSFORM, \
        MAP1, MAP2, SIZE, SHAPE
    if key == 27:
        cv2.destroyAllWindows()
        return 1
    # Clear the screen with the BACKSPACE key
    elif key == 8:
        CAMERA[:] = 0
        DMA[:] = 0
    # Send image from CAMERA to DMA with the ENTER key. The keycode
    # for return will depend on the platform, so \n and \r are both
    # handled.
    elif key == ord('\n') or key == ord('\r'):
        import time
        t0 = time.time()
        DMA = cv2.remap(CAMERA, MAP1, MAP2, cv2.INTER_CUBIC)
    elif key == ord('f'):
        cv2.setWindowProperty('DMA', cv2.WND_PROP_FULLSCREEN, 1)
    elif key == ord('F'):
        cv2.setWindowProperty('DMA', cv2.WND_PROP_FULLSCREEN, 0)
    elif key == ord('r'):
        SHAPE = 'rect'
    elif key == ord('c'):
        SHAPE = 'circle'
    elif key == ord('s'):
        print('diameter = {} px'.format(2 * SIZE + 1))
    elif key == ord('='):
        SIZE += 1
    elif key == ord('-'):
        SIZE -= 1
        if SIZE <= 0:
            SIZE = 0
    elif key == ord('@'):
        DMA[:] = WHITE
    elif key == ord('!'):
        DMA[:] = BLACK
    elif key == ord('o'):
        filename = filepicker.filepicker()
        if filename != '':
            INV_TRANSFORM = loadtransform(filename)
            MAP1, MAP2 = _generate_maps(INV_TRANSFORM.params)
    elif key == ord('h'):
        printdoc()
    elif key == ord('C'):
        CALIBRATION_MODE = True
        DMA = calibration_pattern()
    if CALIBRATION_MODE:
        CALIBRATION_STEP = 10
        if key == ord('W'):
            M = np.float32([[1, 0, 0], [0, 1, -CALIBRATION_STEP]])
            DMA = cv2.warpAffine(DMA, M, DMA_DIM)
        elif key == ord('A'):
            M = np.float32([[1, 0, -CALIBRATION_STEP], [0, 1, 0]])
            DMA = cv2.warpAffine(DMA, M, DMA_DIM)
        elif key == ord('S'):
            M = np.float32([[1, 0, 0], [0, 1, CALIBRATION_STEP]])
            DMA = cv2.warpAffine(DMA, M, DMA_DIM)
        elif key == ord('D'):
            M = np.float32([[1, 0, CALIBRATION_STEP], [0, 1, 0]])
            DMA = cv2.warpAffine(DMA, M, DMA_DIM)
        elif key == ord('=') or key == ord('-'):
            DMA = calibration_pattern(nx=10, ny=10)
        elif key == ord('\n') or key == ord('\r'):
            CALIBRATION_MODE = False
    return 0


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


def _generate_maps(tform):
    global CAMERA_DIM, DMA_DIM, MAP1, MAP2
    # src, dst, tform = _warp_test()
    x, y = np.meshgrid(np.arange(DMA_DIM[0]), np.arange(DMA_DIM[1]))
    on = np.ones(x.shape, dtype=x.dtype)
    powers = np.stack((on, x, y, x**2, x * y, y**2, x**3,
                       x**2 * y, x * y**2, y**3))
    mapx = np.zeros(DMA_DIM[-1::-1], dtype='float32')
    mapy = np.zeros(DMA_DIM[-1::-1], dtype='float32')
    for i in range(10):
        mapx += tform[0, i] * powers[i]
        mapy += tform[1, i] * powers[i]
    MAP1, MAP2 = cv2.convertMaps(
        map1=mapx, map2=mapy, dstmap1type=cv2.CV_16SC2, nninterpolation=False)
    return MAP1, MAP2


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


def main():
    import argparse
    import sys

    global CAMERA_DIM, DMA_DIM, DMA_TEMP, INV_TRANSFORM, MAP1, MAP2, \
        MONITOR_DIM

    parser = argparse.ArgumentParser(description='Process stuff.')
    parser.add_argument(
        '--file', metavar='inverseTransformFile', type=str, default='')

    args, _ = parser.parse_known_args()
    if args.file:
        INV_TRANSFORM = loadtransform(args.file)
        MAP1, MAP2 = _generate_maps(INV_TRANSFORM.params)
    else:
        INV_TRANSFORM = np.array([[0,1,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0]],
                                  dtype='float64')
        MAP1, MAP2 = _generate_maps(INV_TRANSFORM)
        print(
            'No transformation file specified, using identity.',
            file=sys.stderr)

    cv2.namedWindow('DMA', cv2.WINDOW_NORMAL)
    cv2.namedWindow('CAMERA', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('CAMERA', draw)

    while True:
        key = cv2.waitKey(50) & 0xFF
        # ``handlekey`` returns 1 when program should quit.
        if handlekey(key):
            break
        cv2.imshow('CAMERA', CAMERA)
        cv2.imshow('DMA', DMA)


if __name__ == '__main__':
    main()
