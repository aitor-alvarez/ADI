def curve_fitting(freq, time_series, margin):
    freq_len=len(freq)-1
    i=0
    segments=[]
    time=[]

    if freq[i+1] >= freq[i]:
        ind_low =i
        ind_up = i+1
        ref = i+1
    else:
        ind_up=i
        ind_low=i+1
        ref = i + 1

    while i < freq_len:
        if ref < freq_len:
            epsilon=max(abs(freq[ind_up]-freq[ref]), abs(freq[ind_low]-freq[ref]))
            if epsilon <= margin:
                i += 1
                if freq[ref] >= freq[ind_up]:
                    ind_up = ref
                    ref = ref+1
                elif freq[ref] <= freq[ind_low]:
                    ind_low = ref
                    ref = ref+1
                else:
                    ref = ref + 1
            else:
                pitch, start_time, duration, ref, i, ind_up, ind_low=create_linear_segment(freq, time_series, ref, i, ind_up, ind_low)
                segments.append(round(pitch))
                time.append(start_time)
        else:
            pitch, start_time, duration, ref, i, ind_up, ind_low = create_linear_segment(freq, time_series, ref, i,
                                                                                         ind_up, ind_low)
            segments.append(round(pitch))
            time.append(start_time)
    return segments, time


def create_linear_segment(freq, time_series, ref, i, ind_up, ind_low):
    pitch = ((freq[ind_up] - freq[ind_low]) / 2) + freq[ind_low]
    duration = time_series[ref] - time_series[i]
    start_time = time_series[i]

    if ref < len(freq)-1:
        i = ref
        if freq[ref+1] >= freq[ref]:
            ind_low = ref
            ind_up = ref + 1;
            ref = ref + 1;
        else:
            ind_up = ref
            ind_low = ref +1
            ref = ref+1
    else:
        ref= ref+1
        i = i+1
    return pitch, start_time, duration, ref, i, ind_up, ind_low