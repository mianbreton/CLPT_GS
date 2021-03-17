# Makefile for compiling the CLPT_GS code on Linux systems

CC = gcc
CF = -O3 -Wall -Wextra -Wno-unused-parameter -Wuninitialized -Winit-self -pedantic -fopenmp -ffast-math -std=gnu11

INCLUDES = -I/data/home/mbreton/gsl-2.6/include/
LIBRARIES = -L/data/home/mbreton/gsl-2.6/lib/ -lgsl -lgslcblas -lm

##### WARNING !!!!!! MUST WRITE THIS LINE IN TERMINAL BEFORE EXECUTING THE CODE IF GSL LIBRARY NOT IN PATH #############
#export LD_LIBRARY_PATH=/data/home/mbreton/gsl-2.6/lib:$LD_LIBRARY_PATH


#FINALIZE COMPILE FLAGS
CF += $(OPTIONS) #-g


## FINALIZE 
CLPT_INC = $(INCLUDES)
CLPT_LIB =  $(LIBRARIES) 



CLPT: Code_RSD_CLPT.c Code_RSD_GS.c
	$(CC) Code_RSD_CLPT.c -o CLPT $(CF) $(CLPT_INC) $(CLPT_LIB)
	$(CC) Code_RSD_GS.c -o GS $(CF) $(CLPT_INC) $(CLPT_LIB)


clean:
	rm -f CLPT GS */*~ *~
	
lib:    
	$(CC) -shared Code_RSD_GS.c  -fpic $(CF) $(CLPT_INC) $(CLPT_LIB) -o libGS.so
