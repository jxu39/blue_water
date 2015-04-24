import numpy as np
import struct
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()                          # get rank
num_nodes = comm.Get_size()                     # get size
#print "num_nodes is: ",num_nodes

### initial the input file ###
fin = str(sys.argv[1])
bounds_data = np.empty([3,2],dtype=int)

if (rank == 0):
   f = open(fin, "r")                                             # get the input file
   shape = struct.unpack('3i', f.read(3*4))                       # obtain the dimensions of the file
   #print "the shape is: ",shape
   f.seek(12)
   data = struct.unpack('25000000f', f.read(25000000*4))          # the data

   x_segs = int(sys.argv[2])
   y_segs = int(sys.argv[3])
   z_segs = int(sys.argv[4])
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

### compute the sum data for each node ###
if (rank != 0):
    sum_data = np.sum(data_all)
    print "Process",rank,"has data (",bounds_data[0][0],bounds_data[0][1],") (",bounds_data[1][0],bounds_data[1][1],") (",bounds_data[2][0],bounds_data[2][1],"), mean =",sum_data
else:
    sum_data = None

### gather the sum_data ###
sum_data = comm.gather(sum_data, root=0)

### print the sum_data from root node ###
if (rank == 0):
   average = np.sum(np.delete(sum_data,0))/25000000
   print "Processor 0 receives local means:",np.delete(sum_data,0),"and the overall mean =",average




