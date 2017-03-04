import csv

with open('fer2013.csv') as csvfile:
   	readCSV = csv.reader(csvfile, delimiter = ',')
   	with open('FER2013-Training.csv', 'w') as Training:
       		a = csv.writer(Training)
       		for row in readCSV:
       			if (row[2] == 'Training'):
       				del row[2]
   	    			a.writerow(row)
with open('fer2013.csv') as csvfile:
   	readCSV = csv.reader(csvfile, delimiter = ',')
   	with open('FER2013-Validation.csv', 'w') as Validation:
       		a = csv.writer(Validation)
       		for row in readCSV:
       			if (row[2] == 'PublicTest'):
       				del row[2]
   	    			a.writerow(row)
    	    			
with open('fer2013.csv') as csvfile:
   	readCSV = csv.reader(csvfile, delimiter = ',')
   	with open('FER2013-Testing.csv', 'w') as Testing:
       		a = csv.writer(Testing)
       		for row in readCSV:
       			if (row[2] == 'PrivateTest'):
       				del row[2]
   	    			a.writerow(row)