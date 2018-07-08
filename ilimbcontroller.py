import iLimb as ilim
import serial
import time
import math as mt
serialPort = 'COM12' #windows, for linux -> /dev/ttyACM0
il = ilim.iLimbController(serialPort)
il.connect()
#first fully open the hand



def open():
	f=['thumb']
	c=['open']*len(f)
	p=[297]*len(f)
	il.control(f,c,p)
	time.sleep(2)

	fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(2)

	for j in range(len(fingers)):
		c=['open']
		p=[297]
		il.control([fingers[j]],c,p)
		time.sleep(1)

	fingers = ['thumbRotator']
	c=['open']*len(fingers)
	p=[297]*len(fingers)
	il.control(fingers,c,p)
	time.sleep(2)


def restart():
	f=['thumbRotator']
	c=['close']
	p=[297]
	il.control(f,c,p)
	time.sleep(2)

def start():
	fingers = ['thumbRotator']
	c=['open']*len(fingers)
	p=[297]*len(fingers)
	il.control(fingers,c,p)
	time.sleep(2)


	fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(2)

	for j in range(len(fingers)):
		c=['open']
		p=[297]
		il.control([fingers[j]],c,p)
		time.sleep(0.5)

	f=['thumb']
	c=['open']*len(f)
	p=[297]*len(f)
	il.control(f,c,p)
	time.sleep(2)

	f=['thumbRotator']
	c=['close']
	p=[297]
	il.control(f,c,p)
	time.sleep(2)

def verticalpowergrasp(timetosleep):

	# fingers = ['thumbRotator']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(2)


	# fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(2)

	# f=['thumb']
	# c=['open']*len(f)
	# p=[297]*len(f)
	# il.control(f,c,p)
	# time.sleep(2)

	# f=['thumbRotator']
	# c=['close']
	# p=[297]
	# il.control(f,c,p)
	# time.sleep(2)

	fingers=['index','middle','ring','little','thumb']
	# k=['position']*len(fingers)
	stepsrequired=10*(timetosleep)/2
	print(stepsrequired)
	uprange=mt.floor(390/stepsrequired)
	print(uprange)
	if(uprange==0):
		uprange=1
	# a=input("wait")

	for i in range(10,400,uprange):
		# p=[i]*len(fingers)
		# il.control(fingers,k,p)
		# time.sleep(0.03)
		k=['position']
		for j in range(len(fingers)):
			p=[i]
			il.control([fingers[j]],k,p)
			time.sleep(0.05)


def verticaltripod(timetosleep):
	# fingers = ['thumbRotator']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)


	# fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)

	# f=['thumb']
	# c=['open']*len(f)
	# p=[297]*len(f)
	# il.control(f,c,p)
	# time.sleep(0.1)

	# f=['thumbRotator']
	# c=['close']
	# p=[297]
	# il.control(f,c,p)
	# time.sleep(0.1)

	fingers=['index','middle','thumb']
	# k=['position']*len(fingers)
	stepsrequired=10*(timetosleep)/2
	uprange=mt.floor(390/stepsrequired)
	print(uprange)
	if(uprange==0):
		uprange=1
	# a=input("wait")
	# for i in range(10,400,uprange):
	# 	p=[i]*len(fingers)
	# 	il.control(fingers,k,p)
	# 	time.sleep(0.03)

	for i in range(10,400,uprange):
		# p=[i]*len(fingers)
		# il.control(fingers,k,p)
		# time.sleep(0.03)
		k=['position']
		for j in range(len(fingers)):
			p=[i]
			il.control([fingers[j]],k,p)
			time.sleep(0.05)


def verticalpinch(timetosleep):
	# fingers = ['thumbRotator']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)


	# fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)

	# f=['thumb']
	# c=['open']*len(f)
	# p=[297]*len(f)
	# il.control(f,c,p)
	# time.sleep(0.1)

	# f=['thumbRotator']
	# c=['close']
	# p=[297]
	# il.control(f,c,p)
	# time.sleep(0.1)

	fingers=['index','thumb']
	# k=['position']*len(fingers)
	stepsrequired=10*(timetosleep)/2
	uprange=mt.floor(390/stepsrequired)
	print(uprange)
	if(uprange==0):
		uprange=1
	# a=input("wait")
	# for i in range(10,400,uprange):
	# 	p=[i]*len(fingers)
	# 	il.control(fingers,k,p)
	# 	time.sleep(0.03)

	for i in range(10,400,uprange):
		# p=[i]*len(fingers)
		# il.control(fingers,k,p)
		# time.sleep(0.03)
		k=['position']
		for j in range(len(fingers)):
			p=[i]
			il.control([fingers[j]],k,p)
			time.sleep(0.05)

# il.control('wrist','anticlockwise',90)
# time.sleep(1)


# start()
# powergrasp(10)

def horizontalpowergrasp(timetosleep):

	# fingers = ['thumbRotator']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(2)


	# fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(2)

	# f=['thumb']
	# c=['open']*len(f)
	# p=[297]*len(f)
	# il.control(f,c,p)
	# time.sleep(2)

	# f=['thumbRotator']
	# c=['close']
	# p=[297]
	# il.control(f,c,p)
	# time.sleep(2)
	il.control('wrist','anticlockwise',90)
	time.sleep(1)
	fingers=['index','middle','ring','little','thumb']
	# k=['position']*len(fingers)
	stepsrequired=10*(timetosleep)/2
	uprange=mt.floor(390/stepsrequired)
	print(uprange)
	# a=input("wait")
	if(uprange==0):
		uprange=1
	# for i in range(10,400,uprange):
	# 	p=[i]*len(fingers)
	# 	il.control(fingers,k,p)
	# 	time.sleep(0.03)

	for i in range(10,400,uprange):
		# p=[i]*len(fingers)
		# il.control(fingers,k,p)
		# time.sleep(0.03)
		k=['position']
		for j in range(len(fingers)):
			p=[i]
			il.control([fingers[j]],k,p)
			time.sleep(0.05)


def horizontaltripod(timetosleep):
	# fingers = ['thumbRotator']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)


	# fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)

	# f=['thumb']
	# c=['open']*len(f)
	# p=[297]*len(f)
	# il.control(f,c,p)
	# time.sleep(0.1)

	# f=['thumbRotator']
	# c=['close']
	# p=[297]
	# il.control(f,c,p)
	# time.sleep(0.1)

	il.control('wrist','anticlockwise',90)
	time.sleep(1)
	fingers=['index','middle','thumb']
	# k=['position']*len(fingers)
	stepsrequired=10*(timetosleep)/2
	uprange=mt.floor(490/stepsrequired)
	# print(uprange)
	# a=input("wait")
	if(uprange==0):
		uprange=1
	# for i in range(10,400,uprange):
	# 	p=[i]*len(fingers)
	# 	il.control(fingers,k,p)
	# 	time.sleep(0.03)
	for i in range(10,500,uprange):
		# p=[i]*len(fingers)
		# il.control(fingers,k,p)
		# time.sleep(0.03)
		k=['position']
		for j in range(len(fingers)):
			p=[i]
			il.control([fingers[j]],k,p)
			time.sleep(0.05)


def horizontalpinch(timetosleep):
	# fingers = ['thumbRotator']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)


	# fingers = ['index','middle','ring','little']
	# c=['open']*len(fingers)
	# p=[297]*len(fingers)
	# il.control(fingers,c,p)
	# time.sleep(0.1)

	# f=['thumb']
	# c=['open']*len(f)
	# p=[297]*len(f)
	# il.control(f,c,p)
	# time.sleep(0.1)

	# f=['thumbRotator']
	# c=['close']
	# p=[297]
	# il.control(f,c,p)
	# time.sleep(0.1)
	il.control('wrist','anticlockwise',90)
	time.sleep(1)
	fingers=['index','thumb']
	# k=['position']*len(fingers)
	stepsrequired=10*(timetosleep)/2
	uprange=mt.floor(490/stepsrequired)
	# print(uprange)
	# a=input("wait")
	if(uprange==0):
		uprange=1
	# for i in range(10,400,uprange):
	# 	p=[i]*len(fingers)
	# 	il.control(fingers,k,p)
	# 	time.sleep(0.03)
	for i in range(10,500,uprange):
		# p=[i]*len(fingers)
		# il.control(fingers,k,p)
		# time.sleep(0.03)
		k=['position']
		for j in range(len(fingers)):
			p=[i]
			il.control([fingers[j]],k,p)
			time.sleep(0.05)


def anticlockwise():
	il.control('wrist','clockwise',90)
	time.sleep(1)

# start()
# verticalpowergrasp(5)

# il.setPose('openHand')

# open()

# fingers = ['thumbRotator']
# c=['close']*len(fingers)
# p=[297]*len(fingers)
# il.control(fingers,c,p)
# time.sleep(2)