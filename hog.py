import cv2
from surf import load_compressed_image, background_color, set_target
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool


def calc_hog(windowed_gray_img, window_shape, cell_shape, block_shape=[4, 4], block_stride=[1, 1]):
    """
    :param windowed_gray_img:
    :param window_shape: [y, x]
    :param brock_shape:  [y, x]
    :param brock_stride: [y, x]
    :param cell_shape:   [y, x]
    :return:
    """
    img = windowed_gray_img
    y_cell_num = int(window_shape[0] / cell_shape[0])
    x_cell_num = int(window_shape[1] / cell_shape[1])
    st = [[y, x] for y in np.arange(0, window_shape[0], cell_shape[0]) for x in np.arange(0, window_shape[1], cell_shape[1])]

    queue = mp.Queue()

    def wrapped_calc_hog_cell(q, windowed_gray_img):
        """
        :param st: start point of cell
        :return:
        """
        q.put(calc_hog_cell(windowed_gray_img))

    process = [mp.Process(target=wrapped_calc_hog_cell,
                          args=(queue, img[st_yx[0]: st_yx[0] + cell_shape[0], st_yx[1]: st_yx[1] + cell_shape[1]])) for
               st_yx in st]

    for p in process:
        p.start()

    cells_map = np.array([queue.get() for i in range(len(process))]).reshape(y_cell_num, x_cell_num, -1)

    y_block_num = int((y_cell_num - block_shape[0] + 1) / block_stride[0])
    x_block_num = int((x_cell_num - block_shape[1] + 1) / block_stride[1])

    queue = mp.Queue()
    process = []

    for y in range(y_block_num):
        for x in range(x_block_num):
            yst = y*block_stride[0]
            yend = y*block_stride[0]+block_shape[0]
            xst = x*block_stride[1]
            xend = x*block_stride[1]+block_shape[1]
            process.append(mp.Process(target=calc_hog_block, args=(queue, cells_map[yst:yend, xst:xend])))
            # cells_map[yst:yend, xst:xend] = calc_hog_block(cells_map[yst:yend, xst:xend])

    for p in process:
        p.start()

    brock_map = np.array([queue.get() for i in range(len(process))])

    return brock_map.flatten()


def calc_hog_cell(windowed_gray_img, n_bin=9):
    cell = windowed_gray_img
    dx = cv2.Sobel(src=cell, ddepth=cv2.CV_32F, dx=1, dy=0)
    dy = cv2.Sobel(src=cell, ddepth=cv2.CV_32F, dx=0, dy=1)

    magnitude = np.sqrt(dx * dx + dy * dy)
    rad = np.arctan2(dy, dx)
    rad[rad <= 0] += np.pi
    quantized = np.round(rad / (np.pi / n_bin)).astype(np.int32)
    quantized[quantized == 0] = 9
    quantized -= 1  # 1~9 to 0~8 for bincount
    hist = np.bincount(quantized.ravel(), magnitude.ravel(), n_bin)
    return hist


def calc_hog_block(queue, cells):
    norm = np.linalg.norm(cells)
    queue.put(cells.flatten() / (norm + 1e-8))

if __name__ == '__main__':
    # read image in gray-scale
    input = 'data/7.jpg'
    test_file = 'data/9.jpg'
    # input = 'data/10.jpg'
    # test_file = 'data/11.jpg'
    img = load_compressed_image(input, 800)
    print(img.shape)

    # HOG parameters
    # win_shape = [48, 120]
    cs = 8
    cell_shape = [cs, cs]
    block_shape = [8, 8]
    block_stride = [1, 1]
    bins = 9

    # set target object region
    crop_image, stend = set_target(img)
    start = stend[0]
    end = stend[1]

    train_center = np.mean(stend, axis=0)

    height = end[0] - start[0]
    width = end[1] - start[1]
    win_shape = np.array([height - height % cs, width - width % cs])
    stend[1] = stend[0] + win_shape
    win_stride = np.array([int(height/2), int(width/2)])
    # create positive data
    _win_stride = np.array([int(height/2), -int(width/2)])
    positive_stends = np.array([stend - win_stride / 6, stend - _win_stride / 6, stend + _win_stride / 6, stend + win_stride / 6]).astype(np.int32)
    positive_region = np.array([positive_stends[0][0], positive_stends[3][1]])


    positive_data = [calc_hog(windowed_gray_img=img[stend[0][0]:stend[1][0], stend[0][1]:stend[1][1]], window_shape=win_shape,
                        block_shape=block_shape, block_stride=block_stride,
                        cell_shape=cell_shape)]
    negative_data = []

    # for _stend in positive_stends:
    #     windowed = img[_stend[0][0]:_stend[1][0], _stend[0][1]:_stend[1][1]]
    #     hog = calc_hog(windowed_gray_img=windowed, window_shape=win_shape,
    #                    block_shape=block_shape, block_stride=block_stride,
    #                    cell_shape=cell_shape)
    #     positive_data.append(hog)


    y_iter = int((img.shape[0]-win_shape[0]+1) / win_stride[0])
    x_iter = int((img.shape[1]-win_shape[1]+1) / win_stride[1])

    progress = 0
    for y in range(y_iter):
        for x in range(x_iter):
    # for y in range(2):
    #     for x in range(2):
            yst = y * win_stride[0]
            yend = y * win_stride[0] + win_shape[0]
            xst = x * win_stride[1]
            xend = x * win_stride[1] + win_shape[1]
            hog = calc_hog(windowed_gray_img=img[yst:yend, xst:xend], window_shape=win_shape,
                           block_shape=block_shape, block_stride=block_stride,
                           cell_shape=cell_shape)
            if (positive_region[0] < np.array([yst, xst])).all() and (np.array([yst, xst]) < positive_region[1]).all():
                positive_data.append(hog)
            else:
                negative_data.append(hog)
            if (y * x_iter + x + 1) / y_iter * x_iter > progress:
                print('progress...{}%'.format(progress))
                progress += 10


    # svm training
    nega_size = min([1e6, len(negative_data)])
    idx = np.random.randint(0, len(negative_data), nega_size)
    negative_data = np.array(negative_data)
    training = np.concatenate((positive_data, negative_data[idx]), axis=0)
    label = np.concatenate((np.ones(len(positive_data), dtype=np.int32), np.zeros(nega_size, dtype=np.int32)), axis=0)

    # parameters = [{'kernel': ('rbf'), 'C': np.logspace(-4, 4, 9), 'gamma': np.logspace(-4, 4, 9)},
    #               {'kearnel': ('rbf'), 'C': np.logspace(-4, 4, 9)}]
    # tuned_parameters = [
    #     {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    #     {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]}]
    # tuned_parameters = [{'C': [0.1, 0.3, 0.5, 0.6, 0.8], 'kernel': ['linear']}]
    # tuned_parameters = [{'C': [0.01, 0.03, 0.05, 0.06, 0.08], 'kernel': ['linear']}]
    tuned_parameters = [{'C': [0.001, 0.003, 0.005, 0.006, 0.008], 'kernel': ['linear']}]

    # clf = grid_search.GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=-1)
    clf = svm.SVC(kernel='linear', C=1)
    # clf = svm.SVC(kernel='rbf', C=1)
    clf.fit(training, label)

    print(clf.predict(positive_data))
    print(clf.predict(negative_data[idx]))
    # print(clf.best_estimator_)
    # print(clf.grid_scores_)

    # compute decision plane
    v_plane = clf.coef_[0]
    test = [positive_data[0], positive_data[0] + 1e6*v_plane, positive_data[0] - 1e6*v_plane]
    pred = clf.predict(test)
    print('pred_vplane: {}'.format(pred))
    if pred[0] == 0:
        v_plane = -v_plane

    test_img = load_compressed_image(test_file, 800)
    color = tuple(background_color(cv2.imread(test_file, 1)))
    target = []
    _target = []
    progress = 0
    scale = np.abs(np.mean(positive_data))
    for y in range(y_iter):
        for x in range(x_iter):
            yst = y * win_stride[0]
            yend = y * win_stride[0] + win_shape[0]
            xst = x * win_stride[1]
            xend = x * win_stride[1] + win_shape[1]
            hog = calc_hog(windowed_gray_img=test_img[yst:yend, xst:xend], window_shape=win_shape,
                           block_shape=block_shape, block_stride=block_stride,
                           cell_shape=cell_shape)
            test_center = np.mean([[yst, xst], [yend, xend]], axis=0)

            if clf.predict([hog]) == 1:
                target.append([(xst, yst), (xend, yend)])

            # heuristically bias the feature vector
            # if region near the target region, the region is more likely contain target
            hog += v_plane * (scale * np.exp(-np.linalg.norm(train_center-test_center) / max(win_shape[0], win_shape[1])))
            if clf.predict([hog]) == 1:
                _target.append([(xst, yst), (xend, yend)])

            if (y * x_iter + x + 1) / y_iter * x_iter > progress:
                print('progress...{}%'.format(progress))
                progress += 10


    print('detect_num:{}'.format(len(target)))
    _ = test_img[:]
    for st in target:
        cv2.rectangle(_, st[0], st[1], color=color, thickness=-1)
    cv2.imshow("window", _)
    while True:
        key = cv2.waitKey(30)
        if key == ord('a'):
            break

    print('detect_num_heuristic:{}'.format(len(_target)))
    _ = test_img[:]
    for st in _target:
        cv2.rectangle(_, st[0], st[1], color=color, thickness=-1)

    cv2.imshow("window2", _)
    while True:
        key = cv2.waitKey(30)
        if key == ord('q'):
            break


    # hog = cv2.HOGDescriptor(win_s, block_size, block_stride, cell_size, bins)

    # res = hog.compute(img=crop_image)

    # print(res.shape)
    # svm
