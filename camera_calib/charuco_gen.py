import cv2

ARUCO_DICT = cv2.aruco.DICT_6X6_250  # Dictionary ID
SQUARES_VERTICALLY = 6               # Number of squares vertically
SQUARES_HORIZONTALLY = 6             # Number of squares horizontally
SQUARE_LENGTH = 80                   # Square side length (in pixels)
MARKER_LENGTH = 40                   # ArUco marker side length (in pixels)
MARGIN_PX = 20                       # Margins size (in pixels)

IMG_SIZE = tuple(i * SQUARE_LENGTH + 2 * MARGIN_PX for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY))
OUTPUT_NAME = 'ChArUco_Marker.png'


def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, IMG_SIZE, marginSize=MARGIN_PX)
    cv2.imwrite(OUTPUT_NAME, img)

if __name__ == "__main__":
    create_and_save_new_board()
