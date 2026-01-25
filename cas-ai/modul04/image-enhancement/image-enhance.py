###
# Example picture: # https://i.pinimg.com/474x/a0/10/b3/a010b341b067d15afd3da1702facf0b5.jpg?nii=t
###

###
# Getting started:
# pip install opencv-python numpy
###

import cv2
import numpy as np

# === Funktionen für Optimierungen ===
def brightness_contrast(img, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def sharpening_filter2D(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def laplacian_sharpening(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(img - 0.6*lap)
    return sharp

def median_blur(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def gaussian_blur(img, ksize=(15,15), sigmaX=0):
    return cv2.GaussianBlur(img, ksize, sigmaX)

def inverse_colors(img):
    return cv2.bitwise_not(img)

def equalize_hist_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# === Bild laden ===
image_path = "berry-pie.jpg"
orig = cv2.imread(image_path)
assert orig is not None, "Bild konnte nicht geladen werden."

# Resize für Homogenität
max_dim = 600
h, w = orig.shape[:2]
scale = max_dim / max(h, w)
orig = cv2.resize(orig, (int(w*scale), int(h*scale)))

# Liste aller Optimierungen + Beschreibungen
enhancements = [
    ("Original", orig, "Ursprungsbild ohne Optimierung."),
    ("Brightness", brightness_contrast(orig), "Erhöht Helligkeit & Kontrast für klarere Details."),
    ("Sharpening", sharpening_filter2D(orig), "Betont Kanten, um Bildschärfe zu erhöhen."),
    ("Laplacian", laplacian_sharpening(orig), "Stärkere Kantenschärfung."),
    ("Median Blur", median_blur(orig), "Rauschreduktion bei Erhaltung von Kanten."),
    ("Gaussian Blur", gaussian_blur(orig), "Weiche Unschärfe für Rauschunterdrückung."),
    ("Inverse Colors", inverse_colors(orig), "Farbumkehrung für kreative/analytische Zwecke."),
    ("Equalize Hist.", equalize_hist_color(orig), "Verbesserte Tonwerte & Kontrast."),
]

# === Raster 2 Spalten × 4 Zeilen für den Output auf den Folien... ===
cols = 2
rows = 4
cell_w = max_dim
cell_h = max_dim
canvas = np.ones((rows*(cell_h+50), cols*(cell_w+10), 3), dtype=np.uint8) * 255  # +50 für Text

# Schriftgrössen
font = cv2.FONT_HERSHEY_DUPLEX
font_scale_title = 2.2
font_scale_desc = 0.8
font_color_title = (0,0,0)
font_color_desc = (50,50,50)
thickness_title = 2
thickness_desc = 1

for idx, (title, img, desc) in enumerate(enhancements):
    r = idx // cols
    c = idx % cols
    y = r*(cell_h+50)
    x = c*(cell_w+10)
    
    # Bild an Zellgrösse anpassen
    img_resized = cv2.resize(img, (cell_w, cell_h))
    
    # Bild platzieren
    canvas[y:y+cell_h, x:x+cell_w] = img_resized
    
    # Titel
    cv2.putText(canvas, title, (x+10, y+cell_h+40),
                font, font_scale_title, font_color_title, thickness_title, cv2.LINE_AA)
    # Beschreibung -> braucht zu viel Platz..
    #cv2.putText(canvas, desc, (x+10, y+cell_h-40),
    #            font, font_scale_desc, font_color_desc, thickness_desc, cv2.LINE_AA)
    # Beschreibung --> kann einkommentiert werden, ich muss aber etwas an Platz sparen..
    #cv2.putText(canvas, desc, (x+5, y+cell_h+35),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1)


# Anzeigen
cv2.imwrite("m04_01_image_enhence.jpg", canvas)
cv2.imshow("Folio Enhancements 2x4", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
