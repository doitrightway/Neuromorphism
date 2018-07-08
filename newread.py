import os
import filtertd as flt

def xypt(listinput):


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


	i=0
	while(i<len(listinput)):
		a_byte=listinput[i+2:i+4]
		addr= int.from_bytes(a_byte, byteorder='big')
		x.append(127-((addr & xmask) >> xshift))
		y.append(127-((addr & ymask) >> yshift))
		p.append(1- 2*((addr & pmask) >> pshift))
		timestamp= listinput[i+4:i+8]
		t.append(int.from_bytes(timestamp,byteorder='big'))
		i=i+8

	
	output={}
	output['x']=x
	output['y']=y
	output['t']=t
	output['p']=p
	return output
