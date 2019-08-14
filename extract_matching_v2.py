import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def matching(image, ref):

    methods = ['cv.TM_CCOEFF','cv.TM_CCOEFF_NORMED','cv.TM_CCORR','cv.TM_CCORR_NORMED','cv.TM_SQDIFF','cv.TM_SQDIFF_NORMED']
    w, h = ref.shape[::-1]

    for meth in methods:
        #meth = 'cv.TM_SQDIFF'
        img_ = image.copy()
        print(type(img_))
        
        #imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        method = eval(meth) 

        # Apply tewmplate Matching
        res = cv.matchTemplate(img_,ref,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        # Calculate top left location and bottom up location.

        #top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img_,top_left,bottom_right,(255,0,0),5)


        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
        # Slice image according to mathcing (ROI)
        area = img_[top_left[1]: bottom_right[1], top_left[0]:bottom_right[0]]
        #print(type(area))
    return area

def canny(image):

    # Apply Canny Edge detection. 
    edges = cv.Canny(image, 60, 20)
    #w,h = edges.shape[::-1]
    #print(w, h)
    return edges

def img_trim(img): 
    x, y = 297, 445 #자르고 싶은 지점의 x좌표와 y좌표 지정
    w, h = 167, 369 #x로부터 width, y로부터 height를 지정
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    return img_trim

if __name__ == "__main__":

    img = cv.imread('photo/2.jpg')

    img = cv.resize(img, dsize=(600, 900), interpolation=cv.INTER_AREA)

    kernel = np.ones((12,12), np.uint8)
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(3,3))
    
    # --- 노이즈 제거 ---
    img_noise = cv.morphologyEx(img, cv.MORPH_ELLIPSE, kernel1)
    


    # --- threshold ---
    ret, thresh = cv.threshold(img_noise,240,255, cv.THRESH_BINARY)
    
    # --- Canny Edge detection ---
    refer11 = canny(thresh)
    result = cv.dilate(refer11, kernel, iterations = 1)


    # ---------  컨투어 --------------------------
    contours, hierachy = cv.findContours(result, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # 컨투어 값 중 스크린 컨투어 값 찾기. 
    contour_list = []
    for cnt in contours:
        m = len(cnt)
        contour_list.append(m)
    max_ = max(contour_list)
    index_ = contour_list.index(max_)
    contour_list.remove(max_)
    screen = max(contour_list)
    contour_list.insert(index_, max_)

    cnt = contours[contour_list.index(screen)]
    epsilon = 0.001 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    img1 = img.copy()


    img_t = img_trim(img1)

    cv.drawContours(img, [approx] , -1, (0,255,0), 10)
   
    cv.imshow("img", img)

    cv.waitKey(0)
    

