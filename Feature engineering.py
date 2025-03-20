
import os
import numpy as np

import gudhi as gd

dirlst=sorted(os.listdir(r'D:\rois_aal\Outputs\cpac\nofilt_noglobal\rois_aal'), key=str.lower)


def degree(coll,simplex):

    if (len(simplex)-1)==0:
        a1=sum(1 for elem in coll if set(simplex).issubset(set(elem)))
        return a1
    elif (len(simplex)-1)==1:
        a2=sum(1 for elem in coll if set(simplex).issubset(set(elem)))
        return a2


def spectral_feature(all_eigvalues):
    if len(all_eigvalues) != 0:
        m=len(all_eigvalues)
        eigvalues=[x for x in all_eigvalues if x!=0]
        if len(eigvalues)==0:
            return np.array([0,0,0,0,0,0,0,0,m,0,0,0])
        else:
            max1 = np.max(eigvalues)
            min1 = np.min(eigvalues)
            mean = np.round(np.mean(eigvalues), 2)
            std = np.round(np.std(eigvalues), 2)
            sum = np.sum(eigvalues)
            abs = 0
            range = max1 - min1
            for i in eigvalues:
                abs += np.absolute(i - mean)
            second_order = 0
            quasi_winer_index = 0
            spanning_tree_num = 0
            for i in eigvalues:
                second_order += i ** 2
            second_order = np.round(second_order, 2)
            num_nonzero = len(eigvalues)
            num_zero = m - num_nonzero
            for i in eigvalues:
                if i != 0:
                    quasi_winer_index += (num_nonzero + 1) / i
                    spanning_tree_num += np.log(i)
            quasi_winer_index = np.round(quasi_winer_index, 2)
            spanning_tree_num = np.round((spanning_tree_num-np.log(num_nonzero+1)), 2)
        return np.round(np.array([max1, min1, mean, std, sum, abs, range, second_order, num_zero, num_nonzero, quasi_winer_index,
                         spanning_tree_num],dtype=float),2)
    else:
        return np.zeros(12)


spectral_feature1=np.zeros((884,312))
for item in dirlst[:]:
    correlation_matrix=np.loadtxt(r'D:\rois_aal_coor_matrix\{0}.txt'.format(item))
    correlation_matrix1 = np.where(correlation_matrix > 0, correlation_matrix, 0)
    correlation_matrix2 = np.where(correlation_matrix < 0, correlation_matrix, 0)
    num=0

    for i in range(10, 71, 5):
        fil_value = i / 100
        rips_complex1 = gd.RipsComplex(distance_matrix=np.round(1 - correlation_matrix1, 2),
                                       max_edge_length=fil_value)
        simplex_tree1 = rips_complex1.create_simplex_tree(max_dimension=2)
        positive_val = simplex_tree1.get_filtration()
        positive_simplices = set()
        for v in positive_val:
            positive_simplices.add(tuple(v[0]))

        set0, set1, set2 = set(), set(), set()
        for elem in positive_simplices:
            if len(elem) == 1:
                set0.add(elem)
            elif len(elem) == 2:
                set1.add(elem)
            else:
                set2.add(elem)
        lst0,lst1=sorted(list(set0)),sorted(list(set1))
        L_0 = np.zeros((len(lst0), len(lst0)))

        for i in range(len(lst0)):
            L_0[i][i]=degree(coll=set1,simplex=lst0[i])
            for j in range(i+1,len(lst0)):
                if (lst0[i]+lst0[j]) in set1:
                    L_0[i][j]= L_0[j][i]=-1
        eigval0, eigvec0 = np.linalg.eigh(L_0)
        eigvalue0 = np.round(np.sort(eigval0),2)
        #print(eigvalue0)
        spectral_feature_array = spectral_feature(eigvalue0)
        for i in range(12):
           spectral_feature1[dirlst.index(item)][i + num * 12] = spectral_feature_array[i]
        num+=1


        L_1 = np.zeros((len(lst1), len(lst1)))
        for i in range(len(lst1)):
            L_1[i][i]=degree(coll=set2,simplex=lst1[i])+2
            for j in range(i+1,len(lst1)):
                intersection=tuple(set(lst1[i])& set(lst1[j]))
                if intersection:
                    num1=lst1[i].index(intersection[0])
                    num2=lst1[j].index(intersection[0])
                    if tuple(sorted(set(lst1[i]+lst1[j]))) not in set2:
                        if ((num1%2)==(num2%2)):
                            L_1[i][j]= L_1[j][i]=1
                        elif((num1%2)!=(num2%2)):
                            L_1[i][j] = L_1[j][i] =-1
        eigval1, eigvec1 = np.linalg.eigh(L_1)
        eigvalue1 =np.round(np.sort(eigval1),2)
        spectral_feature_array = spectral_feature(eigvalue1)
        for i in range(12):
            spectral_feature1[dirlst.index(item)][i + num * 12] = spectral_feature_array[i]
        num+=1
