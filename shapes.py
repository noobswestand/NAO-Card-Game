import numpy as np
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
import imutils
import client
import time
import math

import random
Color=['red','blue','green']
Shape=['star','square','rectangle','triangle','circle','pentagon']
Good=['Good job!','Great job!','Good work!','You are a pro at this!','You are really good at this!','Great work']

robotActive=False
sayWrong=False#If it tries to say if you showed the wrong one


class ShapeDetector:
	def __init__(self):
		pass
	def checkPerimeter(self,approx):#Checks to see if the points are around the same distance to eachother
		minDistance=-1
		maxDistance=-1
		for i in range(len(approx)):
			xy0=approx[i-1]
			xy1=approx[i]
			xxx0=xy0[0][0]
			yyy0=xy0[0][1]
			xxx1=xy1[0][0]
			yyy1=xy1[0][1]
			#cv2.line(img,(xxx0,yyy0),(xxx1,yyy1),(255,0,0),5)
			if minDistance!=-1:
				minDistance=min(minDistance,point_distance(xxx0,yyy0,xxx1,yyy1))
				maxDistance=max(maxDistance,point_distance(xxx0,yyy0,xxx1,yyy1))
			else:
				minDistance=point_distance(xxx0,yyy0,xxx1,yyy1)
				maxDistance=point_distance(xxx0,yyy0,xxx1,yyy1)
		return abs(maxDistance-minDistance)
	def checkAngles(self,approx,shape="square"):
		minAngle=-1
		maxAngle=-1
		for i in range(len(approx)):
			xy0=approx[i-2]
			xy1=approx[i-1]
			xy2=approx[i]

			xxx0=xy0[0][0]
			yyy0=xy0[0][1]
			xxx1=xy1[0][0]
			yyy1=xy1[0][1]
			xxx2=xy2[0][0]
			yyy2=xy2[0][1]
			ang=abs(point_angle(xxx0,yyy0,xxx1,yyy1)-point_angle(xxx1,yyy1,xxx2,yyy2))
			if shape=="square":
				if ang>180:
					ang-=180
			elif shape=="pentagon":
				if ang>150 and ang<250:
					ang-=150
				if ang>270:
					ang-=208
				#draw_text_shadow(xxx1,yyy1,str(round(ang)))

			if minAngle!=-1:
				minAngle=min(minAngle,ang)
				maxAngle=max(maxAngle,ang)
			else:
				minAngle=ang
				maxAngle=ang
		return round(abs(maxAngle-minAngle))



	def detect(self,c):#Sorts the shapes
		shape=""
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		x,y,w,h = cv2.boundingRect(c)
		xx=round(x+(w/2))
		yy=round(y+(h/2))
		#cv2.ellipse(img,(xx,yy),(5,5),0,0,360,(0,0,255),-1)
		
		if len(approx)==1:
			shape=""
		elif len(approx)==2:
			shape=""
		elif len(approx)==3:
			#cv2.ellipse(img,(xx,yy),(5,5),0,0,360,(0,0,255),-1)
			if self.checkPerimeter(approx)<50:
				shape="triangle"
		elif len(approx)==4:
			#draw_text_shadow(xx,yy,str(self.checkAngles(approx)))
			if self.checkAngles(approx)<20:
				ar = w / float(h)
				if ar >= 0.85 and ar <= 1.15:
					shape="square"
				else:
					shape="rectangle"
		elif len(approx)==5:
			#draw_text_shadow(xx,yy,str(self.checkAngles(approx,"pentagon")))
			if self.checkPerimeter(approx)<35 and self.checkAngles(approx,"pentagon")<35:
				shape="pentagon"
		elif len(approx) == 10:
			shape="star"
		else:#Make better circle detection
			check=False
			#cv2.ellipse(img,(xx,yy),(5,5),0,0,360,(0,0,255),-1)
			xy=approx[0]
			dist=point_distance(xy[0][0],xy[0][1],xx,yy)
			for xy in approx:
				xxx=xy[0][0]
				yyy=xy[0][1]
				#cv2.ellipse(img,(xxx,yyy),(5,5),0,0,360,255,-1)
				if abs(point_distance(xxx,yyy,xx,yy)-dist)>5:
					check=True
					break
			if check==False:
				shape="circle"

		return shape


class ColorLabeler:
	def label(self, image, c):

		blue=(100,45,15)
		green=(50,80,50)
		red=(50,50,150)
		black=(255,255,255)
		white=(0,0,0)

		M = cv2.moments(c)
		if M["m00"]!=0:
			height, width, channels = image.shape

			xx = int(M["m10"] / M["m00"])
			yy = int(M["m01"] / M["m00"])
			xx=clamp(xx,0,width-1)
			yy=clamp(yy,0,height-1)
			color=image[yy][xx]

			'''
			rgb_code_dictionary={(0,0,255):"red",(0,255,0):"green",(255,0,0):"blue"}
			colors = list(rgb_code_dictionary.keys())
			closest_colors = sorted(colors, key=lambda color: distance(color, point))
			closest_color = closest_colors[0]
			code = rgb_code_dictionary[closest_color]
			return code,0
			'''

			colorStr="black"
			colorDist=point_distance(color[0],color[1],color[2],black[0],black[1],black[2])
			colorDist2=point_distance(color[0],color[1],color[2],red[0],red[1],red[2])
			#print((colorDist,colorDist2,"  ",color))

			if colorDist2<colorDist:
				colorDist=colorDist2
				colorStr="red"
			colorDist2=point_distance(color[0],color[1],color[2],green[0],green[1],green[2])
			if colorDist2<colorDist:
				colorDist=colorDist2
				colorStr="green"
			colorDist2=point_distance(color[0],color[1],color[2],blue[0],blue[1],blue[2])
			if colorDist2<colorDist:
				colorDist=colorDist2
				colorStr="blue"
			colorDist2=point_distance(color[0],color[1],color[2],white[0],white[1],white[2])
			if colorDist2<colorDist:
				colorDist=colorDist2
				colorStr="white"
			return colorStr,colorDist
		else:
			return "null",9999


class Detector:
	def detect(self,img):
		s=[]
		resized = imutils.resize(img, width=300)
		ratio = img.shape[0] / float(resized.shape[0])
		
		sd=ShapeDetector()
		cl = ColorLabeler()

		#resized = cv2.fastNlMeansDenoising(resized,None, 65, 5, 21)
		blurred = cv2.GaussianBlur(resized, (5, 5), 0)
		gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


		thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 155, 1)
		#kernel = np.ones((5,5),np.uint8)
		#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		#cv2.imshow("Thresh", thresh)
		
		#edge=cv2.Canny(blurred,200,200)
		#cv2.imshow("Edge", edge)

		# find contours in the thresholded image
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]

		# loop over our contours
		for c in cnts:
			x,y,w,h = cv2.boundingRect(c)
			x*=ratio
			y*=ratio
			w*=ratio
			h*=ratio
			#resize contour
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			shape = sd.detect(c)
			colorT = cl.label(img, c)
			color_dist=colorT[1]
			color=colorT[0]

			area = cv2.contourArea(c)
			#draw_text_shadow(int(x),int(y),str(round(color_dist)))

			if len(c)>1 and color!="black" and color!="white" and shape!="" and area>800 and area<20000 and (color_dist<75):
				text = "{} {}".format(color, shape)
				draw_text_shadow(int(x),int(y),str(text))
				cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
				s.append(([c],color,shape))
		return s


def draw_text_shadow(x,y,txt):
	cv2.putText(img,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0))
	cv2.putText(img,txt,(x-1,y-1),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))

def point_distance(x1,y1,z1,x2,y2=None,z2=None):
	if z2==None:
		return ( (x1 - z1)**2 + (x2 - y1)**2 )**0.5
	else:
		return ( (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 )**0.5
def point_angle(x1,y1,x2,y2):
	return math.degrees(math.atan2(y1-y2, x1-x2))

def say(txt):
	global robotActive
	if robotActive==True:
		client.say(txt)
		print(txt)
	else:
		print(txt)

def clamp(val,mi,ma):
	return max(mi, min(val, ma))

def distance(c1, c2):
	(r1,g1,b1) = c1
	(r2,g2,b2) = c2
	return math.sqrt((r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) **2)

total=0
totalL={}
totalC={}
for i in range(len(Shape)):
	for a in range(len(Color)):
		totalL[Color[a]+" "+Shape[i]]=0
		totalC[Color[a]+" "+Shape[i]]=[]

ShapeTrack=[]
ShapeTrackWrong=[]
ShapeTrackWrong2=[]

fps=60
cam = cv2.VideoCapture(0)
requested=False
requestedTime=0
requestedStiff=False
answered=False



while True:
		if requested==False:
			cs=Color[random.randint(0,2)]+ " " + Shape[random.randint(0,4)]
			#cs="red circle"
			if robotActive==True:
				names=["HeadYaw","HeadPitch"]
				stiff=[1,1]
				client.motion_stiff_set(names,stiff)
			say("Could you show me the "+cs+"?")
			requested=True
			requestedTime=0
			requestedStiff=False
		start_time = time.time()


		#Snap head back to straight - so you know where it is looking
		if robotActive==True:
			if requested==True:
				requestedTime+=1
				if requestedTime>fps*3 and requestedStiff==False:
					requestedStiff=True
					names=["HeadYaw","HeadPitch"]
					angles=[0,7]
					client.motion_stiff_set(names,[1,1])
					client.motion_set(names,angles)
					time.sleep(0.5)
					client.motion_stiff_set(names,[0,0])
					print("HEAD OFF")


		#Just get webcam if robot is not active - for testing
		if robotActive==False:
			ret_val, img = cam.read()
		else:
			img=client.camera_get()

		detect=Detector()
		s=detect.detect(img)
		
		for a in s:
			totalL[a[1]+" "+a[2]]+=1
			totalC[a[1]+" "+a[2]]=a[0]
		total+=1
		

		#Every x frames, we check to see what is the dominate shapes
		if round(fps)>0 and total%round(fps)==0:
			ShapeTrack[:] = []
			for i in range(len(Shape)):
				for a in range(len(Color)):
					if totalL[Color[a]+" "+Shape[i]]>=fps-1:
						ShapeTrack.append((totalC[Color[a]+" "+Shape[i]],Color[a]+" "+Shape[i]))
					totalL[Color[a]+" "+Shape[i]]=0


		'''
		for c in ShapeTrack:
			cv2.drawContours(img, c[0], -1, (0, 255, 0), 2)
		'''

		found=False
		for a in ShapeTrack:
			if a[1]==cs:
				requested=False
				answered=True
				say(random.choice(Good))
				ShapeTrack[:] = []
				ShapeTrackWrong[:]=[]
				found=True
				time.sleep(2)
		if found==False and sayWrong==True:
			if not ShapeTrackWrong:
				ShapeTrackWrong=ShapeTrack
			elif not ShapeTrackWrong2:
				ShapeTrackWrong2=ShapeTrack
			else:
				#Find matching wrong answer
				for a in ShapeTrack:
					find = False
					for b in ShapeTrackWrong:
						find2 = False
						for c in ShapeTrackWrong2:
							if a[1]==b[1] and a[1]==c[1]:
								find = True
								find2 = True
								say("That is a "+a[1]+". Please find me a "+cs)
								break
						if find2==True:
							break
					if find == True:
						break
				#Clear wrong list
				ShapeTrackWrong[:]=[]


		#FPS
		fps=1/(time.time() - start_time)
		draw_text_shadow(0,20,str(round(fps)))
		cv2.imshow("img", img)
		if cv2.waitKey(1) == 27:
			break



if robotActive==True:
	names=["HeadYaw","HeadPitch"]
	stiff=[1,1]
	client.motion_stiff_set(names,stiff)
cv2.destroyAllWindows()
