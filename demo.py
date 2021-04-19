from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
import numpy as np
from utility import eval_band, eval_band_cv
from EGCSR_BS_Clustering import EGCSR_BS_Clustering
from EGCSR_BS_Ranking import EGCSR_BS_Ranking
import time

if __name__ == '__main__':
    root = 'D:\Python\HSI_Files\\'
    # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'PaviaU', 'PaviaU_gt'
    # im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    # im_, gt_ = 'Botswana', 'Botswana_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band).transpose())
    X_img = X_img.transpose().reshape((n_row, n_column, n_band))
    img_correct, gt_correct = p.get_correct(X_img, gt)
    gt_correct = p.standardize_label(gt_correct)
    X_img_2D = X_img.reshape(n_row * n_column, n_band)
    X_img_2D = minmax_scale(X_img_2D.transpose()).transpose()
    n_selected_band = 5

    algorithm = [
                 EGCSR_BS_Clustering(n_selected_band, regu_coef=1e4, n_neighbors=3, ro=0.8),
                 EGCSR_BS_Ranking(n_selected_band, regu_coef=1e4, n_neighbors=3, ro=0.8)
                 ]

    alg_key = ['EGCSR-Clustering', 'EGCSR-Ranking']
    for i in range(algorithm.__len__()):
        time_start = time.clock()
        X_new = algorithm[i].predict(X_img_2D)
        run_time = round(time.clock() - time_start, 3)
        # if X_new.shape[0] < n_selected_band:
        X_new_3D = X_new.reshape((n_row, n_column, X_new.shape[1]))
        img_correct, gt_correct = p.get_correct(X_img, gt)
        gt_correct = p.standardize_label(gt_correct)
        score = eval_band_cv(img_correct, gt_correct, times=10)
        print('%s knn:  %.4f + %.4f' % (alg_key[i], score['knn']['oa'][0], score['knn']['oa'][1]))
        print('%s svm:  %.4f + %.4f' % (alg_key[i], score['svm']['oa'][0], score['svm']['oa'][1]))
        print('-------------------------------------------')
