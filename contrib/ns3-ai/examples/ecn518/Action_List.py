
def Action():
    kmin = [2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
    kmax = [128, 256, 512, 1024, 2048, 5120, 10240]
    pmax = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
    action_space = [[0] * 3] * 390
    num = 0
    for i in range(len(kmin)):
        for j in range(len(kmax)):
            if kmin[i] <= kmax[j]:
                for k in range(len(pmax)):
                    action_space[num] = kmin[i], kmax[j], pmax[k]
                    num += 1
            else:
                pass
    return action_space