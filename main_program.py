import numpy as np
import struct
import sys
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()                          # get rank
num_nodes = comm.Get_size()                     # get size
#print "num_nodes is: ",num_nodes

### initial the input file ###
fin = str(sys.argv[1])
bounds_data = np.empty([3,2],dtype=int)

x_segs = int(sys.argv[2])
y_segs = int(sys.argv[3])
z_segs = int(sys.argv[4])

if (rank == 0):
   f = open(fin, "r")                                             # get the input file
   shape = struct.unpack('3i', f.read(3*4))                       # obtain the dimensions of the file
   #print "the shape is: ",shape
   f.seek(12)
   data = struct.unpack('25000000f', f.read(25000000*4))          # the data


   #print "the segs are: ",x_segs,y_segs,z_segs

   # compute x bounds for each node
   x_bounds = np.zeros((x_segs, 2))
   for row in np.arange(0,x_segs):
       x_bounds[row,0] = x_bounds[row,0]+ shape[0]/x_segs*row
       x_bounds[row,1] = x_bounds[row,1]+ shape[0]/x_segs*(row+1)-1

   y_bounds = np.zeros((y_segs, 2))
   for row in np.arange(0,y_segs):
       y_bounds[row,0] = y_bounds[row,0]+ shape[1]/y_segs*row
       y_bounds[row,1] = y_bounds[row,1]+ shape[1]/y_segs*(row+1)-1
   #print "y_bounds is: ",y_bounds

   # compute z bounds for each node
   z_bounds = np.zeros((z_segs, 2))
   for row in np.arange(0,z_segs):
       z_bounds[row,0] = z_bounds[row,0]+ shape[2]/z_segs*row
       z_bounds[row,1] = z_bounds[row,1]+ shape[2]/z_segs*(row+1)-1
   #print "z_bounds is: ",z_bounds

   ###  sending data to each node ###
   for rank_id in np.arange(0,num_nodes-1):
       #print "current rank is: ",rank_id
       x_index = rank_id/(y_segs*z_segs)
       remainer = rank_id%(y_segs*z_segs)
       y_index = remainer/z_segs
       z_index = remainer%z_segs

       bounds_data[0] = x_bounds[x_index]        ##???
       bounds_data[1] = y_bounds[y_index]
       bounds_data[2] = z_bounds[z_index]
       print "Subvolume (",bounds_data[0][0],bounds_data[0][1],") (",bounds_data[1][0],bounds_data[1][1],") (",bounds_data[2][0],bounds_data[2][1],") is assigned to process",rank_id+1
       comm.Send([bounds_data,MPI.INT],dest=(rank_id+1))             ## sending data
else:
    ###  each node receiving data ###
   comm.Recv([bounds_data,MPI.INT], source=0)


### start to distribute the data
data_all = []
### send data ###
for i in np.arange(0,100):
    if (rank == 0):
       data_slice = data[250000*i:250000*(i+1)]
    else:
       data_slice = None
    ## broadcase the data_slice ##
    data_slice = comm.bcast(data_slice, root=0)

    if(rank != 0):
       #### store the related part ###
       if (bounds_data[2][0] <= i) and (bounds_data[2][1] >= i):
          for j in np.arange(bounds_data[1][0],bounds_data[1][1]+1):
              s_count = bounds_data[0][0] + j*500
              e_count = bounds_data[0][1] + j*500
              data_all.extend(data_slice[s_count:e_count+1])
### do the  Volume Sample Compositing for each node ###

def volume_composite(data,dim,dird,ran,val):

       # different view direction: positive or negative
       if dird == 1:
         range_d = np.arange(0,ran)
       else:
         range_d = np.arange(ran-1,-1,-1)

       # get the data shape
       if dim == 'x':
             data_3d = np.reshape(data,(ran,500,100))
             data_s_shape = (500,100)
             C_shape = (500,100,4)
       elif dim == 'y':
             data_3d = np.reshape(data,(500,ran,100))
             data_s_shape = (500,100)
             C_shape = (500,100,4)
       else:
             #print "the data size is: ",len(data)
             data_3d = np.reshape(data,(500,500,ran))
             data_s_shape = (500,500)
             C_shape = (500,500,4)

       # initial accumulated C and a
       C = np.zeros(C_shape)
       a = np.zeros(data_s_shape)

       # for loop all the slices
       for slice_id in range_d:
          if dim == 'x':
             data_s = data_3d[slice_id,:,:]        ##?

          elif dim == 'y':
             data_s = data_3d[:,slice_id,:]
          else:
             data_s = data_3d[:,:,slice_id]
             #print data_s
# compute the color
          cNorm = col.Normalize()
          jet = plt.get_cmap('jet')
          scalarMap = cm.ScalarMappable(norm=cNorm,cmap=jet)
          colorVal = scalarMap.to_rgba(data_s)
          #print 'colorval is: ', colorVal

          # initial the opacity array
          opa = np.empty(data_s.shape)
          min_data = np.nanmin(data_s)
          #print 'min_data is: ',min_data
          max_data = np.nanmax(data_s)
          #print 'max_data is: ',max_data

          # design the tent function
          k_left = 1/(val-min_data)
          k_right = 1/(max_data-val)

          # compute the opcity array
          for i in np.arange(data_s.shape[0]):
             for j in np.arange(data_s.shape[1]):
                data_p = data_s[i][j]
                if data_p < val:
                   opa[i][j] = 1-(val-data_p)*k_left
                elif data_p == val:
                   opa[i][j] = 1
                else:
                   opa[i][j] = 1-(data_p-val)*k_right
          #print 'opa is: ',opa

          # compute C' and a'(front-to-back compositing)
          C[:,:,0] = C[:,:,0] + colorVal[:,:,0]*opa*(1-a)
          C[:,:,1] = C[:,:,1] + colorVal[:,:,1]*opa*(1-a)
          C[:,:,2] = C[:,:,2] + colorVal[:,:,2]*opa*(1-a)
          C[:,:,3] = C[:,:,3] + colorVal[:,:,3]*opa*(1-a)
          a = a + opa*(1-a)
       C[:,:,3] = a
       return C

if(rank!=0):
  if (x_segs>1):
     C = volume_composite(data_all,'x', 1, 500/x_segs, 0)
  if (y_segs>1):
     C = volume_composite(data_all,'y', 1, 500/y_segs, 0)
  if (z_segs>1):
     C = volume_composite(data_all,'z', 1, 100/z_segs, 0)
else:
  C = None
C = comm.gather(C, root=0)
## the root node would composite all of them
if(rank == 0):
  if (x_segs>1):
     data_s_shape = (500,100)
     C_shape = (500,100,3)
  elif (y_segs>1):
     data_s_shape = (500,100)
     C_shape = (500,100,3)
  else:
     data_s_shape = (500,500)
     C_shape = (500,500,3)

  C_out = np.zeros(C_shape)
  a_out = np.zeros(data_s_shape)
  for i in np.arange(1,num_nodes):
          C_out[:,:,0] = C_out[:,:,0] + C[i][:,:,0]*C[i][:,:,3]*(1-a_out)
          C_out[:,:,1] = C_out[:,:,1] + C[i][:,:,1]*C[i][:,:,3]*(1-a_out)
          C_out[:,:,2] = C_out[:,:,2] + C[i][:,:,2]*C[i][:,:,3]*(1-a_out)
          a_out = a_out + C[i][:,:,3]*(1-a_out)
  ## save fig ##
  plt.imshow(C_out)
  plt.title("Result of composition")
  plt.savefig("composition_result.png")



