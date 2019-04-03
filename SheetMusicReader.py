import cv2 as cv
import sys
import numpy as np
import os


GAUSSIAN_BLUR_KERNEL = (11, 11)
THRESHOLD_MIN = 160
THRESHOLD_MAX = 255
LINES_DISTANCE_THRESHOLD = 50
LINES_ENDPOINTS_DIFFERENCE = 10


class Staff:
    def __init__(self, min_range, max_range):
        self.min_range = min_range
        self.max_range = max_range
        self.lines_location, self.lines_distance = self.get_lines_locations()

    def get_lines_locations(self):
        lines = []
        lines_distance = int((self.max_range - self.min_range) / 4)
        for i in range(5):
            lines.append(self.min_range + i * lines_distance)
        return lines, lines_distance


def getProjection(image, path):
    img = cv.imread(image)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    cv.imwrite(path + 'HSV.jpg', s)

    th, threshed = cv.threshold(s, 50, 255, cv.THRESH_BINARY_INV)
    cv.imwrite(path+ 'Threshold.jpg', threshed)

    _, cnts, _ = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    canvas = img.copy()
    newCnts = []
    for cnt in cnts:
        arclen = cv.arcLength(cnt, True)
        if cnt[0][0][0] < 2 or cnt[0][0][1] < 2 or len(cv.approxPolyDP(cnt, 0.02 * arclen, True)) != 4:
            continue
        else:
            newCnts.append(cnt)
    cv.drawContours(canvas, newCnts, -1, (0, 255, 0), 3)
    cv.imwrite(path + 'Contours.jpg', canvas)

    newCnts = sorted(newCnts, key=cv.contourArea)
    cnt = newCnts[-1]

    sheet = cv.approxPolyDP(cnt, 0.02 * arclen, True)
    pts = []
    for i in sheet:
        pts.append((i[0][0], i[0][1]))

    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
    cv.imwrite(path + 'Perspektywa.jpg', warped)
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    cv.imwrite(path + 'SzaraP.jpg', warped)
    #result = cv.adaptiveThreshold(warped, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 115, 1)
    #return result
    rgb_planes = cv.split(warped)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
        cv.imwrite(path + 'Dly.jpg', dilated_img)
        bg_img = cv.medianBlur(dilated_img, 21)
        cv.imwrite(path + 'Median.jpg', bg_img)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        result_planes.append(diff_img)

    result = cv.merge(result_planes)
    cv.imwrite(path + 'AfterShadows.jpg', result)

    _, result = cv.threshold(result, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #result = cv.adaptiveThreshold(result, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)
    #result = cv.adaptiveThreshold(~result, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    cv.imwrite(path + 'Wynik.jpg', result)
    return result


def preprocess_image(image, path):
    gray = image.copy()
    _, thresholded = cv.threshold(gray, THRESHOLD_MIN, THRESHOLD_MAX, cv.THRESH_BINARY)
    element = np.ones((3, 3))
    thresholded = cv.erode(gray, element)
    cv.imwrite(path + 'Erozja.jpg', thresholded)
    edges = cv.Canny(thresholded, 10, 100, apertureSize=3)
    cv.imwrite(path + 'Krawedzie.jpg', edges)
    return edges, thresholded


def detect_lines(hough, image, nlines):
    all_lines = set()
    width, height = image.shape
    lines_image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for result_arr in hough[:nlines]:
        rho = result_arr[0][0]
        theta = result_arr[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        shape_sum = width + height
        x1 = int(x0 + shape_sum * (-b))
        y1 = int(y0 + shape_sum * a)
        x2 = int(x0 - shape_sum * (-b))
        y2 = int(y0 - shape_sum * a)

        start = (x1, y1)
        end = (x2, y2)
        diff = y2 - y1
        if abs(diff) < LINES_ENDPOINTS_DIFFERENCE:
            all_lines.add(int((start[1] + end[1]) / 2))
    return all_lines, lines_image_color


def detect_staffs(all_lines):
    staffs = []
    lines = []
    for current_line in all_lines:
        if lines and abs(lines[-1] - current_line) > LINES_DISTANCE_THRESHOLD:
            if len(lines) >= 5:
                staffs.append((lines[0], lines[-1]))
            lines.clear()
        lines.append(current_line)

    if len(lines) >= 5:
        if abs(lines[-2] - lines[-1]) <= LINES_DISTANCE_THRESHOLD:
            staffs.append((lines[0], lines[-1]))
    return staffs


def detect_staffs2(all_lines):
    staffs = []
    lines = []
    for i in range(7):
        for j in range(1, 10, 2):
            lines.append(all_lines[i*10 + j])
        if lines[4] - lines[0] < 100:
            staff = (lines[0], lines[4])
            staffs.append(staff)
            lines.clear()
    return staffs


def detect_staffs3(all_lines):
    staffs = []
    lines = []
    for n, i in enumerate(all_lines):
        lines.append(i)
        if len(lines) > 1:
            if lines[-1] - lines[0] > 110 or n == len(all_lines):
                distance = min([lines[-2] - lines[-4], 16])
                print(distance)
                staff = (lines[-2] - 4 * distance, lines[-2])
                print(staff)
                staffs.append(staff)
                lines = [lines[-1]]
            else:
                continue
    return staffs


def get_staffs(image, path):
    processed_image, thresholded = preprocess_image(image, path)
    hough = cv.HoughLines(processed_image, 1, np.pi / 180, 100)
    all_lines, lines_image_color = detect_lines(hough, thresholded, 140)
    all_lines = sorted(all_lines)
    tmp = image.copy()
    num = 0
    new_lines = []
    for i in all_lines:
        if 50 < i < image.shape[0] - 50:
            new_lines.append(i)
            num = num + 1
            cv.line(tmp, (0, i), (tmp.shape[1]-1, i), (0, 0, 255))
    cv.imwrite(path + 'Linie.jpg', tmp)
    staffs = detect_staffs(all_lines)
    '''if len(staffs) != 7:
        if len(new_lines) == 70:
            staffs = detect_staffs2(new_lines)
        else:
            staffs = detect_staffs3(new_lines)
    #draw_staffs(lines_image_color, staffs)'''
    return [Staff(staff[0], staff[1]) for staff in staffs]


def getNotes(image, path):
    binary = cv.adaptiveThreshold(~image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    cv.imwrite(path + 'Binary.jpg', binary)

    height, width = binary.shape
    horizontalSize = int(width / 30)
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
    eroH = cv.erode(binary, horizontalStructure, (-1, -1))
    dilH = cv.dilate(eroH, horizontalStructure, (-1, -1))

    verticalSize = int(height / 30)
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
    cv.imwrite(path + 'Vertical.jpg', verticalStructure)
    eroV = cv.erode(binary, verticalStructure, (-1, -1))
    cv.imwrite(path + 'EroV.jpg', eroV)
    dilV = cv.dilate(eroV, verticalStructure, (-1, -1))
    cv.imwrite(path + 'DilV.jpg', dilV)
    return dilH, dilV


def findWindow(contour):
    x = [contour[i][0][0] for i in range(len(contour))]
    y = [contour[i][0][1] for i in range(len(contour))]
    l = min(x)
    r = max(x)
    d = min(y)
    u = max(y)
    return l-30, r+30, d-30, u+30


def getCntItem(cnt):
    return cnt[0][0][0]


def detectHalf(img, center, l, r):
    black_streak = 0
    max_black = 0
    for i in range(l-20, r+5, 1):
        if img[center, i] == 0 and img[center, i+1] == 0:
            black_streak = black_streak + 1
            if black_streak > max_black:
                max_black = black_streak
        else:
            black_streak = 0
    if max_black > 11:
        return False
    else:
        return True

def detectEights(img, r, c, topmost):
    change1 = 0
    if topmost < 10:
        h = topmost + 15
    else:
        h = 1/4*(r - 61) + 30
    for i in range(30, c-20, 1):
        if img[int(h), i] != img[int(h), i+1]:
            change1 = change1 + 1
    if change1 == 2:
        return False
    else:
        return True


def detectFull(t, b, l, r):
    #print((b-t)/(r-l))
    #print([t, b, l, r])
    if (b-t)/(r-l) < 1.1:
        return True
    return False


def findTop(img, l, r, c):
    black = 0
    limit = 5
    #print([l, r])
    if r - l < 12:
        l = l - 10
        limit = 4
    while True:
        if c + 1 == img.shape[0]:
            return c
        c = c + 1
        #print([c, l, r])
        for i in range(l, r, 1):
            if img[c][i] == 0:
                black = black + 1
            if i - l > 10 and black == 0:
                break
        if black > limit:
            return c
        else:
            black = 0


def findPosition(t, b, lines, distance):
    #print(b)
    if b > lines[4] + int(2 / 5 * distance):
        return '5'
    for i in range(len(lines)):
        #print([[lines[i] - int(4/5*distance), t, lines[i]], [lines[i], b, lines[i] + int(4/5*distance)]])
        #print([lines[i] - int(3/11*distance), int((t+b)/2), lines[i] + int(3/11*distance)])
        #print([lines[i] - int(3 / 5 * distance), t, b, lines[i] + int(3 / 5 * distance)])
        #if lines[i] - int(9/10*distance) < t < lines[i] and lines[i] < b < lines[i] + int(9/10*distance):
        #    return 'Linia numer ' + str(i)
        if lines[i] - int(1/4*distance) < int((t+b)/2) < lines[i] + int(1/4*distance) and\
                t > lines[i]-int(4/5*distance) and b < lines[i]+int(4/5*distance):
            return str(i+1)
        #if lines[i] - int(1 / 4 * distance) < int((t + b) / 2) < lines[i] + int(1 / 4 * distance) and \
        #        t > lines[i] - distance and b < lines[i] + distance:
        #   return str(i + 1)

    for i in range(len(lines)-1):
        #print([[lines[i] - int(2/5*distance), t, lines[i] + int(2/5*distance)], [lines[i+1] - int(2/5*distance), b, lines[i+1] + int(2/5*distance)]])
        #if lines[i] - int(2/5*distance) > t:
        #    return 'Pomiędzy ' + str(i-1) + ' i ' + str(i)
        #if lines[i] - int(2/5*distance) <= t <= lines[i] + int(2/5*distance) and lines[i+1] - int(2/5*distance) <= b <= lines[i+1] + int(2/5*distance):
        #    return 'Pomiędzy ' + str(i) + ' i ' + str(i+1)
        if lines[i] <= int((t+b)/2) <= lines[i+1]:
            return str(i+1) + ' i ' + str(i + 2)


def draw_staffs(image, staffs, path):
    # Draw the staffs
    width = image.shape[0]
    for staff in staffs:
        cv.line(image, (0, staff.lines_location[0]), (width, staff.lines_location[0]), (0, 255, 255), 2)
        for i in range(1, 4):
            cv.line(image, (0, staff.lines_location[i]), (width, staff.lines_location[i]), (255, 255, 0), 2)
        cv.line(image, (0, staff.lines_location[4]), (width, staff.lines_location[4]), (0, 255, 255), 2)
    cv.imwrite(path + "KoloroweLinie.jpg", image)



def writeOnImage(img, notes, klucz):
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in notes:
        cv.putText(img, i[0], (i[2], i[3]-60), font, 0.6, (255, 0, 255), 2, cv.LINE_AA)
        cv.putText(img, i[1], (i[2], i[3] - 40), font, 0.6, (255, 0, 255), 2, cv.LINE_AA)
    cv.putText(img, klucz, (50, i[3] - 40), font, 0.6, (255, 0, 255), 2, cv.LINE_AA)


def magic(img, per, path):
    staffs = get_staffs(img, path)
    i = 0
    tmpr = img.copy()
    for idx in range(len(staffs)):
        staff = staffs[idx]
        st = img[staff.lines_location[0] - 35:staff.lines_location[4] + 35, :]
        lines = [i for i in range(35, 35+staff.lines_distance*4+1, staff.lines_distance)]
        y = getNotes(st, path)[1]
        y = cv.bitwise_not(y)
        im2, contours, hierarchy = cv.findContours(y, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: getCntItem(x))
        clefContours = [[], []]
        nuty = []
        for cnt in contours[:]:
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])[1]
            botmost = tuple(cnt[cnt[:, :, 1].argmax()][0])[1]
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])[0]
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])[0]
            if cnt[0][0][0] == 0:
                continue
            if cnt[0][0][0] < 100:
                clefContours[0].append(topmost)
                clefContours[1].append(botmost)
                continue
            elif len(cnt) > 25 or botmost - topmost > 20:
                nl, nr, nd, nu = findWindow(cnt)
                nl, nr, nd, nu = max([nl, 0]), min([nr, y.shape[1] - 1]), max([nd, 0]), min([nu, y.shape[0] - 1])
                tmp = y[nd:nu, nl:nr]
                r, c = tmp.shape
                top = findTop(y, leftmost, rightmost, int((topmost + botmost) / 2))
                if detectFull(topmost, botmost, leftmost, rightmost) or botmost - topmost < int(5/4*staff.lines_distance):
                    nuta = 'Cala nuta'
                    position = findPosition(topmost, botmost, lines, staff.lines_distance)
                elif detectHalf(y, int((top+botmost)/2), leftmost, rightmost):
                    position = findPosition(top, botmost, lines, staff.lines_distance)
                    nuta = 'Polnuta'
                else:
                    if detectEights(tmp, r, c, topmost):
                        position = findPosition(top, botmost, lines, staff.lines_distance)
                        nuta = 'Osemka'
                    else:
                        position = findPosition(top, botmost, lines, staff.lines_distance)
                        nuta = 'Cwiercnuta'
                clefBot = max(clefContours[1])
                clefTop = min(clefContours[0])
                nuty.append([nuta, position, int((leftmost+rightmost)/2), staff.lines_location[0]])
                cv.line(y, (nl, top), (nr, top), (0, 0, 255))
                cv.line(y, (nl, botmost), (nr, botmost), (0, 0, 255))
            i = i + 1
        if clefBot > lines[4] + 5 or clefTop < lines[0]-5:
            klucz = 'Wiolinowy'
        else:
            klucz = 'Basowy'
        writeOnImage(per, nuty, klucz)
        cv.imwrite(path + 'Wyciete' + str(idx) + '.jpg', st)
        cv.imwrite(path + 'dilV' + str(idx) + '.jpg', y)


def goThroughFiles():
    nlyfiles = [f for f in os.listdir('Works') if os.path.isfile(os.path.join('Works', f))]
    for i in nlyfiles:
        print(i)
        path = 'Results/' + i[:len(i)-4] + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        cv.imwrite(path + 'Oryginal.jpg', cv.imread('Works/' + i))
        img = getProjection('Works/'+i, path)
        per = cv.imread(path+'AfterShadows.jpg')
        cv.imwrite(path + 'Zdjecie.jpg', img)
        magic(img, per, path)
        cv.imwrite(path + 'Napisy.jpg', per)


goThroughFiles()