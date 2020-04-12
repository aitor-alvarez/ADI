def contour_approximation(f0, time):
    contour=[]
    ind=[]
    val ={'1':70, '2':112}
    for k in range(0, len(f0)-1):
        if f0[k] >= f0[k+1]:
            c=f0[k] - f0[k + 1]
            if c < val['1']:
                contour.append('1')
                ind.append((time[k], time[k + 1]))
            elif c >= val['1'] and c <= val['2']:
                contour.append('2')
                ind.append((time[k], time[k + 1]))
            elif c > val['2']:
                contour.append('3')
                ind.append((time[k], time[k + 1]))
            else:
                continue

        else:
            c = f0[k + 1] - f0[k]
            if c < val['1']:
                contour.append('-1')
                ind.append((time[k], time[k + 1]))
            elif c >= val['1'] and c <= val['2']:
                contour.append('-2')
                ind.append((time[k], time[k + 1]))
            elif c > val['2']:
                contour.append('-3')
                ind.append((time[k], time[k + 1]))
            else:
                continue
    return contour, ind