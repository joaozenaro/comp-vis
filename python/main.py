import cv2

imagem = cv2.imread("tmp/lena.png")
cv2.imshow("Imagem Python", imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()