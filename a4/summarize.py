"""
sumarize.py
"""
def main():
    l = open("/Users/purvank/Desktop/OSNA IIT/a4/collect.txt")
    la = []
    for line in l:
        la.append(line.strip('\n'))
    print ("Number of users collected: %s\n"%la[0])
    print ("Number of messsages collected: %s\n"%la[1])
    l1 = open("/Users/purvank/Desktop/OSNA IIT/a4/cluster.txt")
    la1 = []
    for line in l1:
        la1.append(line.strip('\n'))
    print("Number of communities dicovered: %s\n" %la1[0])
    print("Average number of users per community: %s\n" %la1[1])
    l2 = open("/Users/purvank/Desktop/OSNA IIT/a4/classify.txt")
    la2 = []
    for line in l2:
        la2.append(line.strip('\n'))
    print("Number of Positive Instances: %s\n"% la2[0])
    print("Number of Negative Instances: %s\n"% la2[1])
    print("Positive Class Example: %s\n"%la2[2])
    print("Negatvie Class Example: %s\n"%la2[3])
if __name__ == '__main__':
    print ("--------------summarize.py-----------------")
    main()
    print ("--------------summarize.py-----------------")