import cv2, glob

gimg = glob.glob("faceDetectionGroup/Images/*.jpg")
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for tmg in gimg:

	img = cv2.imread(tmg)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face = detect.detectMultiScale(gray, 1.16, 5)

	for (x,y,w,h) in face:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0, 200, 0), 3)
	scale_percent = 12
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

	cv2.imshow("Detect Multiple Images", img)
	cv2.waitKey(1500)
	cv2.destroyAllWindows()