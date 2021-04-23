'''
1,sitting,

2,standing,

3,lying on back,

4,lying on right side,

5,ascending stairs,

6,descending stairs,

7,standing in an elevator still,

8,moving around in an elevator,

9,walking in a parking lot,

10,walking on a treadmill with a speed of 4 kmh,

11,walking in flat and 15 deg inclined positions,

12,running on a treadmill with a speed of 8 kmh,

13,exercising on a stepper,

14,exercising on a cross trainer,

15,cycling on an exercise bike in horizontal positions,

16,cycling on an exercise bike in vertical positions,

17,rowing,

18,jumping,

19,playing basketball

Anoamly classes including running, ascending stairs, descending stairs and rope jumping

'''
import numpy as np
path = '/media/zyx/self_supervised/DSADS/data/'
ac_class = []
for i in range(1,20):
    if i < 10:
        ac_class.append('a0'+str(i))
    else:
        ac_class.append('a'+str(i))

ac_person = []
for i in range(1,9):
        ac_person.append('p'+str(i))

ac_file = []
for i in range(1,61):
    if i < 10:
        ac_file.append('s0' + str(i))
    else:
        ac_file.append('s' + str(i))

print(ac_class)
print(ac_person)
print(ac_file)

###################normal class#########################
normal = []
normal_class = [i for i in range(19) if i not in [4,5,11,17,18]]
for i in range(14):
    for j in range(8):
        for z in range(60):
            print(ac_class[normal_class[i]] +'/'+ ac_person[j]+'/'+ ac_file[z]+ '.txt')
            data = np.loadtxt(path + ac_class[normal_class[i]] +'/'+ ac_person[j]+'/'+ ac_file[z] + '.txt',delimiter= ',')
            normal.append(data)
    print(len(normal))
normal = np.array(normal)
print(normal.shape)
np.save("/media/zyx/self_supervised/DSADS/normal.npy", normal)

###################abnormal class#########################
abnormal_class = [4,5,11,17,18]
abnormal = []
for i in range(5):
    for j in range(8):
        for z in range(60):
            print(ac_class[abnormal_class[i]] +'/'+ ac_person[j]+'/'+ ac_file[z]+ '.txt')
            data = np.loadtxt(path + ac_class[abnormal_class[i]] +'/'+ ac_person[j]+'/'+ ac_file[z] + '.txt',delimiter= ',')
            abnormal.append(data) 
    print(len(abnormal))
abnormal = np.array(abnormal)
print(abnormal.shape)
np.save("/media/zyx/self_supervised/DSADS/abnormal.npy", abnormal)