import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import pandas as pd
  
CSV_FILE = "aligned_data/051920231000/P005/May_19_run_2 copy.csv"
df = pd.read_csv(CSV_FILE)

cols, rows = df.shape

x=[]
acc_x=[]
acc_y=[]
acc_z=[]
gyro_x=[]
gyro_y=[]
gyro_z=[]

# Initialise the subplot function using number of rows and columns 
figure, axis = plt.subplots(1, 2) 

def updateAcc(i):
    axis[0].cla()
    axis[0].set_title("Acceleration") 
    acc_x.append(df['acc_x'][i])
    acc_y.append(df['acc_y'][i])
    acc_z.append(df['acc_z'][i])
    if (i > 20):
        axis[0].plot(x[i-20:],acc_x[i-20:], color='r', label='x')
        axis[0].plot(x[i-20:],acc_y[i-20:], color='g', label='y')
        axis[0].plot(x[i-20:],acc_z[i-20:], color='b', label='z')
    else:
        axis[0].plot(x,acc_x, color='r', label='x')
        axis[0].plot(x,acc_y, color='g', label='y')
        axis[0].plot(x,acc_z, color='b', label='z')

def updateGyro(i):
    axis[1].cla()
    axis[1].set_title("Gyro") 
    gyro_x.append(df['gyro_x'][i])
    gyro_y.append(df['gyro_y'][i])
    gyro_z.append(df['gyro_z'][i])
    if (i > 20):
        axis[1].plot(x[i-20:],gyro_x[i-20:], color='r', label='x')
        axis[1].plot(x[i-20:],gyro_y[i-20:], color='g', label='y')
        axis[1].plot(x[i-20:],gyro_z[i-20:], color='b', label='z')
    else:
        axis[1].plot(x,gyro_x, color='r', label='x')
        axis[1].plot(x,gyro_y, color='g', label='y')
        axis[1].plot(x,gyro_z, color='b', label='z')

def animate(frame):
    x.append(frame)
    updateAcc(frame)
    updateGyro(frame)

ani = FuncAnimation(plt.gcf(), animate, frames=cols, interval=500, repeat=False)
plt.tight_layout()

plt.show()
