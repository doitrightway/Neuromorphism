import os

def xypt(file):

	f = open(file, "rb")
	while True:
		reader = f.readline()
		# print(reader)
		if(reader[0]==35):
			# print("hello")
			pass
		else:
			f.seek(-1*len(reader),1)
			break

	# print(f.readline())

	xmask = 0x00fe
	xshift = 1
	ymask = 0x7f00
	yshift = 8
	pmask = 0x1
	pshift = 0

	x=[]
	y=[]
	t=[]
	p=[]
	# print(f.read(4))
	while (f.tell() != os.fstat(f.fileno()).st_size):
		f.seek(2,1)
		a_byte= f.read(2)
		# print(a_byte)
		addr= int.from_bytes(a_byte, byteorder='big')
		x.append(127-((addr & xmask) >> xshift))
		y.append(127-((addr & ymask) >> yshift))
		p.append(1- 2*((addr & pmask) >> pshift))
		timestamp= f.read(4)
		t.append(int.from_bytes(timestamp,byteorder='big'))
		
	output={}
	output['x']=x
	output['y']=y
	output['t']=t
	# print(output['t'])
	output['p']=p
	return output

# xypt('myfile.bin')