import re
import csv
f = open("train_snli.txt", "r")
with open('train_data1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    a=0
    #sentences1,sentences2,is_similar
    writer.writerow(["", "sentences1", "sentences2","is_similar"])
    for x in f:
        y=re.split('\\t+', x)
        #print(y[3])
        if(a==0):
            a=a+1
            continue
        if(float(y[3])>3.5)
        writer.writerow([a, y[1].rstrip("\n"), y[2].rstrip("\n"),1)
        if(float(y[3])<2)
        writer.writerow([a, y[1].rstrip("\n"), y[2].rstrip("\n"),0)
        a=a+1

   #print(words)
