import matplotlib.pyplot as plt
import numpy
import sys
import os 
import time

cont = True 
def on_close(evt):
	print("close")
	cont = False
	
last_size = 0
last_mdf = 0
avg = -1.0
fname = sys.argv[1]
if not os.path.exists(fname) :
	print ("File does not exits.\n")
	sys.exit()
	
fo = open(sys.argv[1], 'r') 
if last_size > 0 :
	f.seek(last_size)
y = []
z = []
s = []
t = []
x = []
lines = fo.readlines() 
fo.close() 
for line in lines :
	l = line.strip().split(',')
	i = float(l[1]) / 32
	s.append(i)
	x.append(float(l[0]))	
	loss = float(l[4])
	# loss2 = 13 * loss / i
	avg = float(l[5])	
	y.append( loss )
	z.append( avg )
	t.append(loss)
	
fig, ax = plt.subplots() 
	
ax.set_title('Loss Tendency of Palms Recognition Training')	
hl, = ax.plot(x, t,color='red', label='Loss')
h2, = ax.plot(x, z, color='blue', label='Avg Loss')
ax.set_xlabel('Iteration times')
ax.set_ylabel('Loss/N') 
ax.legend()
plt.show()



# plt.axis([0,6,0,6])
# plt.plot(x, s, color='yellow', label='N:Net_Size/32')
# plt.plot(x, y, color='green', label='Loss2: Loss/N')


# plt.savefig('scatter.png')